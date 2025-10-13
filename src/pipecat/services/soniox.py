#
# pipecat/services/soniox.py
#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
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
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import websockets
except ModuleNotFoundError:
    logger.error("In order to use Soniox, you need to `pip install pipecat-ai[soniox]`")
    raise

def language_to_soniox_language(language: Language) -> Optional[str]:
    """Maps Pipecat Language enum to Soniox language codes."""
    # Soniox uses standard IETF language tags (e.g., "en", "es", "fr-CA")
    # We can mostly pass the language code through directly.
    return language.value

class SonioxSTTService(STTService):
    """
    An optimized Soniox STT service for real-time transcription.
    """

    class InputParams(BaseModel):
        language: Language = Field(default=Language.EN_US)
        model: str = "stt-rt-v2"  # Soniox's real-time model
        enable_speaker_diarization: bool = False
        enable_language_identification: bool = False
        allow_interruptions: bool = True

    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: int = 16000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._websocket = None
        self._receive_task = None
        self._connection_active = False

        self._params = params
        self._language = params.language
        self._allow_stt_interruptions = params.allow_interruptions

        # State tracking
        self._user_speaking = False
        self._bot_speaking = True
        self._vad_active = False
        self._last_final_transcript_time = 0

        # Performance tracking
        self._stt_response_times = []
        self._current_speech_start_time = None
        self._audio_chunk_count = 0

        logger.info("Soniox STT Service initialized.")
        logger.info(f"  Model: {params.model}, Language: {params.language.value}")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._websocket or not self._connection_active:
            yield None
            return

        if self._current_speech_start_time is None:
            self._current_speech_start_time = time.perf_counter()
            self._audio_chunk_count = 0
            logger.debug("🎤 Soniox: Starting speech detection timer.")

        self._audio_chunk_count += 1
        try:
            await self._websocket.send(audio)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Soniox WebSocket connection closed, attempting to reconnect.")
            await self._reconnect()
        yield None

    async def _connect(self):
        if self._websocket and self._connection_active:
            return

        logger.info("Connecting to Soniox...")
        uri = (
            f"wss://api.soniox.com/v1/speech-to-text-rt"
            f"?model={self._params.model}"
            f"&language={language_to_soniox_language(self._language)}"
            f"&sample_rate={self.sample_rate}"
            f"&enable_speaker_diarization={str(self._params.enable_speaker_diarization).lower()}"
            f"&enable_language_identification={str(self._params.enable_language_identification).lower()}"
        )

        try:
            self._websocket = await websockets.connect(uri, extra_headers={"Authorization": f"Bearer {self._api_key}"})
            self._connection_active = True
            logger.info("Connected to Soniox.")

            if not self._receive_task:
                self._receive_task = self.create_task(self._receive_task_handler())
            await self.start_ttfb_metrics()
        except Exception as e:
            logger.error(f"Failed to connect to Soniox: {e}")
            await self.push_error(ErrorFrame(f"Soniox connection failed: {e}"))

    async def _disconnect(self):
        self._connection_active = False
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
            logger.info("Disconnected from Soniox.")

    async def _reconnect(self):
        await self._disconnect()
        await asyncio.sleep(1)
        await self._connect()

    async def _receive_task_handler(self):
        while self._connection_active:
            try:
                message = await self._websocket.recv()
                await self._on_message(json.loads(message))
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Soniox connection closed during receive.")
                break
            except Exception as e:
                logger.error(f"Error in Soniox receive task: {e}")
                break

    async def _on_message(self, data: Dict):
        await self.stop_ttfb_metrics()

        words = data.get("words", [])
        if not words:
            return

        transcript = " ".join(w["text"] for w in words)
        is_final = data.get("final", False)

        if not transcript.strip():
            return

        if self._bot_speaking and not self._allow_stt_interruptions:
            logger.debug(f"Ignoring transcript: bot speaking and interruptions disabled: '{transcript}'")
            return

        await self._handle_user_speaking()
        
        timestamp = time_now_iso8601()
        language_enum = self.language

        if is_final:
            self._record_stt_performance(transcript, words)
            frame = TranscriptionFrame(transcript, "", timestamp, language_enum)
            await self.push_frame(frame)
            self._last_final_transcript_time = time.time()
            await self._handle_user_silence()
        else:
            frame = InterimTranscriptionFrame(transcript, "", timestamp, language_enum)
            await self.push_frame(frame)

    def _record_stt_performance(self, transcript, words):
        if self._current_speech_start_time:
            elapsed = time.perf_counter() - self._current_speech_start_time
            self._stt_response_times.append(elapsed)
            confidence = sum(w.get("confidence", 0) for w in words) / len(words) if words else 0
            logger.info(f"📊 ⚡ Soniox: ⏱️ STT Response Time: {elapsed:.3f}s")
            logger.info(f"   📝 Final Transcript: '{transcript}'")
            logger.info(f"   🎯 Avg. Confidence: {confidence:.2f}")
            logger.info(f"   📦 Audio chunks processed: {self._audio_chunk_count}")
            self._current_speech_start_time = None

    async def _handle_user_speaking(self):
        if not self._user_speaking:
            self._user_speaking = True
            await self.push_frame(UserStartedSpeakingFrame())
            logger.info("👤 User started speaking.")

    async def _handle_user_silence(self):
        if self._user_speaking:
            self._user_speaking = False
            await self.push_frame(UserStoppedSpeakingFrame())
            logger.info("👤 User stopped speaking.")
            
    async def _handle_bot_speaking(self):
        self._bot_speaking = True
        logger.debug("🤖 Bot started speaking.")

    async def _handle_bot_silence(self):
        self._bot_speaking = False
        logger.debug("🤖 Bot stopped speaking.")

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