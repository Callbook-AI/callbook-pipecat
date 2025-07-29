#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import base64
import json
import time
from typing import AsyncGenerator, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

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


# Constants similar to Deepgram service
DEFAULT_ON_NO_PUNCTUATION_SECONDS = 3
IGNORE_REPEATED_MSG_AT_START_SECONDS = 4
VOICEMAIL_DETECTION_SECONDS = 10
FALSE_INTERIM_SECONDS = 1.3


def language_to_gladia_language(language: Language) -> Optional[str]:
    BASE_LANGUAGES = {
        Language.AF: "af",
        Language.AM: "am",
        Language.AR: "ar",
        Language.AS: "as",
        Language.AZ: "az",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.EU: "eu",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GA: "ga",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.HY: "hy",
        Language.ID: "id",
        Language.IS: "is",
        Language.IT: "it",
        Language.JA: "ja",
        Language.JV: "jv",
        Language.KA: "ka",
        Language.KK: "kk",
        Language.KM: "km",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LO: "lo",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MN: "mn",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.MY: "my",
        Language.NE: "ne",
        Language.NL: "nl",
        Language.NO: "no",
        Language.OR: "or",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PS: "ps",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SI: "si",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SO: "so",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SU: "su",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.UZ: "uz",
        Language.VI: "vi",
        Language.ZH: "zh",
        Language.ZU: "zu",
    }

    result = BASE_LANGUAGES.get(language)

    # If not found in base languages, try to find the base language from a variant
    if not result:
        # Convert enum value to string and get the base language part (e.g. es-ES -> es)
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        # Look up the base code in our supported languages
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class GladiaSTTService(STTService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        endpointing: Optional[float] = 0.2
        maximum_duration_without_endpointing: Optional[int] = 10
        audio_enhancer: Optional[bool] = None
        words_accurate_timestamps: Optional[bool] = None
        speech_threshold: Optional[float] = 0.99

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "https://api.gladia.io/v2/live",
        confidence: float = 0.5,
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS,
        detect_voicemail: bool = True,
        allow_interruptions: bool = True,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._api_key = api_key
        self._url = url
        self.language = params.language
        self.detect_voicemail = detect_voicemail
        self._allow_stt_interruptions = allow_interruptions
        logger.debug(f"Allow interruptions: {self._allow_stt_interruptions}")
        
        self._settings = {
            "encoding": "wav/pcm",
            "bit_depth": 16,
            "sample_rate": 0,
            "channels": 1,
            "language_config": {
                "languages": [self.language_to_service_language(params.language)]
                if params.language
                else [],
                "code_switching": False,
            },
            "endpointing": params.endpointing,
            "maximum_duration_without_endpointing": params.maximum_duration_without_endpointing,
            "pre_processing": {
                "audio_enhancer": params.audio_enhancer,
                "speech_threshold": params.speech_threshold,
            },
            "realtime_processing": {
                "words_accurate_timestamps": params.words_accurate_timestamps,
            },
        }
        
        self._confidence = confidence
        self._websocket = None
        self._receive_task = None
        self._async_handler_task = None
        
        # State management
        self._user_speaking = False
        self._bot_speaking = True
        self._on_no_punctuation_seconds = on_no_punctuation_seconds
        self._vad_active = False
        
        # Message handling
        self._first_message = None
        self._first_message_time = None
        self._last_interim_time = None
        self._restarted = False
        
        # Accumulation handling
        self._accum_transcription_frames = []
        self._last_time_accum_transcription = time.time()
        self._last_time_transcription = time.time()
        self._was_first_transcript_receipt = False
        
        self.start_time = time.time()

    def _time_since_init(self):
        return time.time() - self.start_time

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_gladia_language(language)

    def _transcript_words_count(self, transcript: str):
        return len(transcript.split(" "))

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._websocket:
            return
        self._settings["sample_rate"] = self.sample_rate
        response = await self._setup_gladia()
        self._websocket = await websockets.connect(response["url"])
        self._restarted = True  # Enable message processing
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler())
        if not self._async_handler_task:
            self._async_handler_task = self.create_monitored_task(self._async_handler)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        if self._websocket:
            await self._send_stop_recording()
            await self._websocket.close()
            self._websocket = None
        if self._receive_task:
            await self.wait_for_task(self._receive_task)
            self._receive_task = None
        if self._async_handler_task:
            await self.cancel_task(self._async_handler_task)
            self._async_handler_task = None

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        if self._websocket:
            await self._websocket.close()
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        if self._async_handler_task:
            await self.cancel_task(self._async_handler_task)
            self._async_handler_task = None

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self.start_processing_metrics()
            await self._send_audio(audio)
            await self.stop_processing_metrics()
            yield None
        except Exception as e:
            logger.exception(f"{self} exception in run_stt: {e}")
            yield ErrorFrame(f"run_stt error: {e}")

    async def _setup_gladia(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self._url,
                headers={"X-Gladia-Key": self._api_key, "Content-Type": "application/json"},
                json=self._settings,
            ) as response:
                if response.ok:
                    return await response.json()
                else:
                    logger.error(
                        f"Gladia error: {response.status}: {response.text or response.reason}"
                    )
                    raise Exception(f"Failed to initialize Gladia session: {response.status}")

    async def _send_audio(self, audio: bytes):
        data = base64.b64encode(audio).decode("utf-8")
        message = {"type": "audio_chunk", "data": {"chunk": data}}
        await self._websocket.send(json.dumps(message))

    async def _send_stop_recording(self):
        if self._websocket:
            await self._websocket.send(json.dumps({"type": "stop_recording"}))

    async def _receive_task_handler(self):
        """Enhanced message handler with advanced processing."""
        if not self._restarted:
            return
            
        try:
            async for message in self._websocket:
                content = json.loads(message)
                if content["type"] == "transcript":
                    await self._handle_transcript_message(content)
        except Exception as e:
            logger.exception(f"{self} unexpected error in _receive_task_handler: {e}")

    async def _handle_transcript_message(self, content):
        """Handle transcript message with advanced logic."""
        try:
            utterance = content["data"]["utterance"]
            confidence = utterance.get("confidence", 0)
            transcript = utterance["text"]
            is_final = content["data"]["is_final"]
            
            # Extract language if available
            language = None
            if "language" in utterance:
                try:
                    language = Language(utterance["language"])
                except ValueError:
                    language = None
            
            logger.debug(f"Transcription{'' if is_final else ' interim'}: {transcript}")
            logger.debug(f"Confidence: {confidence}")
            
            if len(transcript) > 0:
                await self.stop_ttfb_metrics()

                # Check for voicemail
                if await self._detect_and_handle_voicemail(transcript):
                    return
                
                # Check if we should ignore this transcription
                if await self._should_ignore_transcription(transcript, is_final, confidence):
                    return
                
                # Process transcription based on confidence
                if confidence >= self._confidence:
                    if is_final:
                        await self._on_final_transcript_message(transcript, language, confidence)
                        self._last_time_transcription = time.time()
                    else:
                        await self._on_interim_transcript_message(transcript, language, time.time())
                        
        except Exception as e:
            logger.exception(f"{self} exception in _handle_transcript_message: {e}")

    async def _handle_user_speaking(self):
        """Handle when user starts speaking."""
        await self.push_frame(StartInterruptionFrame())
        if self._user_speaking:
            return
        self._user_speaking = True
        await self.push_frame(UserStartedSpeakingFrame())

    async def _handle_user_silence(self):
        """Handle when user stops speaking."""
        if not self._user_speaking:
            return
        self._user_speaking = False
        await self.push_frame(UserStoppedSpeakingFrame())

    async def _handle_bot_speaking(self):
        """Handle when bot starts speaking."""
        self._bot_speaking = True

    async def _handle_bot_silence(self):
        """Handle when bot stops speaking."""
        self._bot_speaking = False

    async def _async_handle_accum_transcription(self, current_time):
        """Handle accumulated transcription timeout."""
        if (current_time - self._last_time_accum_transcription > self._on_no_punctuation_seconds 
            and len(self._accum_transcription_frames)):
            logger.debug("Sending accum transcription because of timeout")
            await self._send_accum_transcriptions()

    async def _handle_false_interim(self, current_time):
        """Handle false interim results."""
        if not self._user_speaking:
            return
        if not self._last_interim_time:
            return
        if self._vad_active:
            return

        last_interim_delay = current_time - self._last_interim_time
        if last_interim_delay > FALSE_INTERIM_SECONDS:
            return

        logger.debug("False interim detected")
        await self._handle_user_silence()

    async def _async_handler(self, task_name):
        """Background task for handling timeouts and false interims."""
        while True:
            if not self.is_monitored_task_active(task_name):
                return

            await asyncio.sleep(0.1)
            
            current_time = time.time()
            await self._async_handle_accum_transcription(current_time)
            await self._handle_false_interim(current_time)

    async def _send_accum_transcriptions(self):
        """Send accumulated transcription frames."""
        if not len(self._accum_transcription_frames):
            return

        await self._handle_user_speaking()

        for frame in self._accum_transcription_frames:
            await self.push_frame(frame)
        self._accum_transcription_frames = []

        await self._handle_user_silence()
        await self.stop_processing_metrics()

    def _is_accum_transcription(self, text: str):
        """Check if text should be accumulated (lacks end punctuation)."""
        END_OF_PHRASE_CHARACTERS = ['.', '?']
        text = text.strip()
        if not text:
            return True
        return not text[-1] in END_OF_PHRASE_CHARACTERS
    
    def _append_accum_transcription(self, frame: TranscriptionFrame):
        """Add frame to accumulation buffer."""
        self._last_time_accum_transcription = time.time()
        self._accum_transcription_frames.append(frame)

    def _handle_first_message(self, text):
        """Handle first message logic."""
        if self._first_message:
            return
        self._first_message = text
        self._first_message_time = time.time()

    def _should_ignore_first_repeated_message(self, text):
        """Check if we should ignore repeated first message."""
        if not self._first_message:
            return False
        
        time_since_first_message = time.time() - self._first_message_time
        if time_since_first_message > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        return is_equivalent_basic(text, self._first_message)

    async def _on_final_transcript_message(self, transcript, language, confidence):
        """Handle final transcript message."""
        await self._handle_user_speaking()
        frame = TranscriptionFrame(transcript, "", time_now_iso8601(), language)

        self._handle_first_message(frame.text)
        self._append_accum_transcription(frame)
        self._was_first_transcript_receipt = True
        
        if not self._is_accum_transcription(frame.text):
            await self._send_accum_transcriptions()
    
    async def _on_interim_transcript_message(self, transcript, language, start_time):
        """Handle interim transcript message."""
        self._last_interim_time = time.time()
        await self._handle_user_speaking()
        await self.push_frame(
            InterimTranscriptionFrame(transcript, "", time_now_iso8601(), language)
        )

    async def _should_ignore_transcription(self, transcript, is_final, confidence):
        """Check if transcription should be ignored."""
        if not is_final and confidence < 0.7:
            logger.debug("Ignoring interim because low confidence")
            return True

        if self._transcript_words_count(transcript) == 1:
            logger.debug("Ignoring single word at start")
            return True
        
        if self._should_ignore_first_repeated_message(transcript):
            logger.debug("Ignoring repeated first message")
            return True
        
        if not self._vad_active and not is_final:
            logger.debug("Ignoring interruption because VAD inactive")
            return True
        
        logger.debug(f"Bot speaking: {self._bot_speaking}, allow_interruptions: {self._allow_stt_interruptions}")
        if self._bot_speaking and not self._allow_stt_interruptions:
            logger.debug("Ignoring interruption because allow_interruptions is False")
            return True
        
        if self._bot_speaking and self._transcript_words_count(transcript) == 1:
            logger.debug(f"Ignoring interruption because bot is speaking: {transcript}")
            return True

        return False

    async def _detect_and_handle_voicemail(self, transcript: str):
        """Detect and handle voicemail."""
        if not self.detect_voicemail:
            return False

        logger.debug(transcript)
        logger.debug(self._time_since_init())
        
        if (self._time_since_init() > VOICEMAIL_DETECTION_SECONDS 
            and self._was_first_transcript_receipt):
            return False
        
        if not voicemail.is_text_voicemail(transcript):
            return False
        
        logger.debug("Voicemail detected")
        await self.push_frame(VoicemailFrame(transcript))
        logger.debug("Voicemail pushed")
        return True

    async def start_metrics(self):
        """Start TTFB and processing metrics."""
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process various frame types for state management."""
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("Received bot started speaking on Gladia")
            await self._handle_bot_speaking()

        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("Received bot stopped speaking on Gladia")
            await self._handle_bot_silence()

        elif isinstance(frame, STTRestartFrame):
            logger.debug("Received STT Restart Frame")
            self._restarted = True
            # Reconnect websocket
            if self._websocket:
                await self._websocket.close()
                self._websocket = None
            if self._receive_task:
                await self.cancel_task(self._receive_task)
                self._receive_task = None
            # Restart connection
            response = await self._setup_gladia()
            self._websocket = await websockets.connect(response["url"])
            self._receive_task = self.create_task(self._receive_task_handler())
            return

        elif isinstance(frame, UserStartedSpeakingFrame):
            # Start metrics if external VAD has detected speech
            await self.start_metrics()
            
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # Handle user stopped speaking
            logger.debug("User stopped speaking, sending stop recording")
            await self._send_stop_recording()
        
        elif isinstance(frame, VADInactiveFrame):
            self._vad_active = False
            await self._send_stop_recording()
            
        elif isinstance(frame, VADActiveFrame):
            self._vad_active = True
