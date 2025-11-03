#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import time
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field, field_validator

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADActiveFrame,
    VADInactiveFrame,
    VoicemailFrame,
    STTRestartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.string import is_equivalent_basic
from pipecat.utils.text import voicemail

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Gladia, you need to `pip install pipecat-ai[gladia]`. Also, set `GLADIA_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")

# Constants for optimized behavior
DEFAULT_ON_NO_PUNCTUATION_SECONDS = 2.0
IGNORE_REPEATED_MSG_AT_START_SECONDS = 4.0
VOICEMAIL_DETECTION_SECONDS = 10.0
FALSE_INTERIM_SECONDS = 1.3


def language_to_gladia_language(language: Language) -> Optional[str]:
    BASE_LANGUAGES = {
        Language.EN: "english", Language.ES: "spanish", Language.FR: "french",
        Language.DE: "german", Language.IT: "italian", Language.PT: "portuguese",
        Language.RU: "russian", Language.JA: "japanese", Language.ZH: "chinese",
        # Add other supported languages here
    }
    result = BASE_LANGUAGES.get(language)
    if not result:
        lang_str = str(language.value).split("-")[0].lower()
        # A simple heuristic to map language codes to names if not directly in the map
        for key, value in BASE_LANGUAGES.items():
            if str(key.value).startswith(lang_str):
                return value
    return result


class GladiaSTTService(STTService):
    """
    An optimized Gladia STT service that follows high-performance patterns for real-time transcription.
    """

    class InputParams(BaseModel):
        language: Language = Language.EN_US
        model: str = "fast"
        endpointing_delay: int = 400
        allow_interruptions: bool = True
        detect_voicemail: bool = True

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "wss://api.gladia.io/audio/text/audio-transcription",
        confidence_threshold: float = 0.6,
        sample_rate: int = 16000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._url = url
        self.language = params.language
        self._model = params.model
        self.detect_voicemail = params.detect_voicemail
        self._allow_stt_interruptions = params.allow_interruptions
        self._confidence_threshold = confidence_threshold
        self._endpointing_delay = params.endpointing_delay

        self._websocket = None
        self._receive_task = None
        self._async_handler_task = None

        self._user_speaking = False
        self._bot_speaking = True
        self._vad_active = False

        self._first_message = None
        self._first_message_time = None
        self._last_interim_time = None

        self._accum_transcription = ""
        self._last_time_accum_transcription = time.time()
        self._last_time_transcription = time.time()

        self.start_time = time.time()
        self._stt_response_times = []
        self._current_speech_start_time = None
        self._audio_chunk_count = 0

        logger.info("Gladia STT Service initialized.")
        logger.info(f"  Model: {self._model}, Language: {self.language}")
        logger.info(f"  Allow Interruptions: {self._allow_stt_interruptions}, Detect Voicemail: {self.detect_voicemail}")

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_gladia_language(language)

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()
        if not self._async_handler_task:
            self._async_handler_task = self.create_monitored_task(self._async_handler)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._websocket:
            yield None
            return

        current_time = time.perf_counter()
        self._audio_chunk_count += 1
        if self._current_speech_start_time is None:
            self._current_speech_start_time = current_time
            logger.debug(f"🎤 Gladia: Starting speech detection timer at chunk #{self._audio_chunk_count}")

        await self.start_processing_metrics()
        b64_audio = base64.b64encode(audio).decode("utf-8")
        message = json.dumps({"frames": b64_audio})
        try:
            await self._websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Gladia WebSocket connection closed, attempting to reconnect.")
            await self._reconnect()
        yield None

    async def _connect(self):
        if self._websocket:
            return

        logger.info("Connecting to Gladia...")
        gladia_language = self.language_to_service_language(self.language)
        url = f"{self._url}?model={self._model}&language={gladia_language}&endpointing={self._endpointing_delay}&sample_rate={self.sample_rate}"

        try:
            self._websocket = await websockets.connect(url, extra_headers={"x-gladia-key": self._api_key})
            logger.info("Connected to Gladia.")
            if not self._receive_task:
                self._receive_task = self.create_task(self._receive_task_handler())
            await self.start_ttfb_metrics()
        except Exception as e:
            logger.error(f"Failed to connect to Gladia: {e}")
            await self.push_error(ErrorFrame(f"Gladia connection failed: {e}"))

    async def _disconnect(self):
        if self._async_handler_task:
            await self.cancel_task(self._async_handler_task)
            self._async_handler_task = None
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            logger.info("Disconnected from Gladia.")

    async def _reconnect(self):
        await self._disconnect()
        await asyncio.sleep(1)
        await self._connect()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_speaking()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_silence()
        elif isinstance(frame, VADActiveFrame):
            self._vad_active = True
        elif isinstance(frame, VADInactiveFrame):
            self._vad_active = False
        elif isinstance(frame, STTRestartFrame):
            logger.info("Restarting Gladia STT service.")
            await self._reconnect()

    async def _receive_task_handler(self):
        while self._websocket:
            try:
                message = await self._websocket.recv()
                await self._on_message(json.loads(message))
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Gladia connection closed during receive.")
                break
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Error in Gladia receive task: {error_msg}")

                # Push ErrorFrame before breaking
                error_msg_lower = error_msg.lower()
                is_fatal = any(keyword in error_msg_lower for keyword in [
                    'authentication', 'unauthorized', '401', '403', 'invalid', 'api key',
                    'quota', 'suspended'
                ])

                await self.push_error(ErrorFrame(
                    error=f"Gladia receive error: {error_msg[:200]}",
                    fatal=is_fatal
                ))
                break

    async def _on_message(self, data: Dict):
        transcript = data.get("transcription", "")
        if not transcript or not transcript.strip():
            return

        is_final = data.get("type") == "final"
        confidence = data.get("confidence", 0.0)

        await self.stop_ttfb_metrics()

        if await self._should_ignore_transcription(transcript, is_final, confidence):
            return

        if await self._detect_and_handle_voicemail(transcript):
            return
        
        timestamp = time_now_iso8601()
        language_enum = self.language

        if is_final:
            self._record_stt_performance(transcript, confidence)
            await self._on_final_transcript(transcript, timestamp, language_enum)
        else:
            await self._on_interim_transcript(transcript, timestamp, language_enum)

    async def _on_final_transcript(self, transcript, timestamp, language):
        if self._bot_speaking and self._allow_stt_interruptions:
            await self.push_frame(StartInterruptionFrame())

        full_transcript = self._accum_transcription + transcript
        frame = TranscriptionFrame(full_transcript.strip(), "", timestamp, language)
        await self.push_frame(frame)
        
        self._handle_first_message(full_transcript)
        self._last_time_transcription = time.time()
        self._accum_transcription = ""
        await self._handle_user_silence()
        await self.stop_processing_metrics()

    async def _on_interim_transcript(self, transcript, timestamp, language):
        self._last_interim_time = time.time()
        await self._handle_user_speaking()
        full_transcript = self._accum_transcription + transcript
        await self.push_frame(InterimTranscriptionFrame(full_transcript.strip(), "", timestamp, language))

    def _record_stt_performance(self, transcript, confidence):
        if self._current_speech_start_time is not None:
            elapsed = time.perf_counter() - self._current_speech_start_time
            self._stt_response_times.append(elapsed)
            logger.info(f"📊 ⚡ Gladia: ⏱️ STT Response Time: {elapsed:.3f}s, Transcript: '{transcript}', Confidence: {confidence:.2f}")
            self._current_speech_start_time = None

    async def _should_ignore_transcription(self, transcript, is_final, confidence):
        if confidence < self._confidence_threshold:
            return True
        if self._bot_speaking and not self._allow_stt_interruptions:
            return True
        if self._should_ignore_first_repeated_message(transcript):
            return True
        return False

    async def _detect_and_handle_voicemail(self, transcript):
        if not self.detect_voicemail or self._time_since_init() > VOICEMAIL_DETECTION_SECONDS:
            return False
        if voicemail.is_text_voicemail(transcript):
            await self.push_frame(VoicemailFrame(transcript))
            return True
        return False
    
    async def _async_handler(self):
        while True:
            await asyncio.sleep(0.1)
            # Placeholder for future logic, like managing timeouts.

    def _handle_first_message(self, text):
        if not self._first_message:
            self._first_message = text
            self._first_message_time = time.time()

    def _should_ignore_first_repeated_message(self, text):
        if not self._first_message or (time.time() - self._first_message_time > IGNORE_REPEATED_MSG_AT_START_SECONDS):
            return False
        return is_equivalent_basic(text, self._first_message)

    async def _handle_user_speaking(self):
        if not self._user_speaking:
            self._user_speaking = True
            await self.push_frame(UserStartedSpeakingFrame())

    async def _handle_user_silence(self):
        if self._user_speaking:
            self._user_speaking = False
            await self.push_frame(UserStoppedSpeakingFrame())

    async def _handle_bot_speaking(self):
        self._bot_speaking = True

    async def _handle_bot_silence(self):
        self._bot_speaking = False