#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
from typing import AsyncGenerator, Optional

from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    StartInterruptionFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
)
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import assemblyai as aai
    from assemblyai import AudioEncoding, RealtimeTranscript, RealtimeFinalTranscript, RealtimeError, RealtimeSessionOpened
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use AssemblyAI, you need to `pip install pipecat-ai[assemblyai]`. Also, set `ASSEMBLYAI_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


# Similar to Deepgram, this allows flushing partial sentences if no punctuation is received.
DEFAULT_ON_NO_PUNCTUATION_SECONDS = 3


class AssemblyAISTTService(STTService):
    """
    Speech-to-text service using AssemblyAI's real-time transcription API.

    This service is designed to behave similarly to the DeepgramSTTService,
    including features like sentence accumulation, user speaking state management,
    and interruption handling.
    """
    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: Optional[int] = None,
        encoding: AudioEncoding = AudioEncoding("pcm_s16le"),
        language: Language = Language.EN,  # Only English is supported for Realtime
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        aai.settings.api_key = api_key
        self._transcriber: Optional[aai.RealtimeTranscriber] = None

        self._settings = {
            "encoding": encoding,
            "language_code": language.value, # AssemblyAI SDK expects the string value
        }

        # --- State management mirrored from DeepgramSTTService ---
        self._user_speaking = False
        self._bot_speaking = True
        self._on_no_punctuation_seconds = on_no_punctuation_seconds

        self._async_handler_task: Optional[asyncio.Task] = None
        self._accum_transcription_frames: list[TranscriptionFrame] = []
        self._last_time_accum_transcription = time.time()
        # --- End of state management ---

    def can_generate_metrics(self) -> bool:
        return True

    async def set_language(self, language: Language):
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language_code"] = language.value
        # Re-connect to apply new settings
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        if not self._transcriber:
            logger.warning("Transcriber not available, skipping audio frame.")
            yield None
            return

        try:
            await self.start_processing_metrics()
            # The AssemblyAI SDK handles streaming in a background thread
            self._transcriber.stream(audio)
            await self.stop_processing_metrics()
            yield None
        except Exception as e:
            logger.error(f"Error streaming audio to AssemblyAI: {e}")
            yield ErrorFrame(f"AssemblyAI stream error: {e}")

    # --- Logic mirrored from DeepgramSTTService to handle speaking states ---
    async def _handle_user_speaking(self):
        if self._user_speaking:
            return
        logger.debug("User started speaking")
        self._user_speaking = True
        await self.push_frame(StartInterruptionFrame())
        await self.push_frame(UserStartedSpeakingFrame())

    async def _handle_user_silence(self):
        if not self._user_speaking:
            return
        logger.debug("User stopped speaking")
        self._user_speaking = False
        await self.push_frame(UserStoppedSpeakingFrame())

    async def _handle_bot_speaking(self):
        self._bot_speaking = True

    async def _handle_bot_silence(self):
        self._bot_speaking = False

    # --- Sentence accumulation logic from DeepgramSTTService ---
    def _is_accum_transcription(self, text: str) -> bool:
        END_OF_PHRASE_CHARACTERS = ['.', '?', '!']
        text = text.strip()
        return not text or text[-1] not in END_OF_PHRASE_CHARACTERS

    def _append_accum_transcription(self, frame: TranscriptionFrame):
        self._last_time_accum_transcription = time.time()
        self._accum_transcription_frames.append(frame)

    async def _send_accum_transcriptions(self):
        if not self._accum_transcription_frames:
            return

        logger.debug("Sending accumulated transcriptions")
        await self._handle_user_speaking()

        for frame in self._accum_transcription_frames:
            await self.push_frame(frame)
        self._accum_transcription_frames = []

        await self._handle_user_silence()
        await self.stop_processing_metrics()

    # --- Async handler task to flush incomplete sentences, like in Deepgram ---
    async def _async_handler(self, task_name: str):
        while self.is_monitored_task_active(task_name):
            await asyncio.sleep(0.5)
            current_time = time.time()
            if current_time - self._last_time_accum_transcription > self._on_no_punctuation_seconds and self._accum_transcription_frames:
                logger.debug("Timeout detected, flushing accumulated transcriptions.")
                await self._send_accum_transcriptions()

    def _on_data(self, transcript: RealtimeTranscript):
        """Callback for handling incoming transcription data from the SDK's thread."""
        if not transcript.text:
            return

        async def handle_push():
            await self.stop_ttfb_metrics()

            if isinstance(transcript, RealtimeFinalTranscript):
                logger.debug(f"Received final transcript: '{transcript.text}'")
                frame = TranscriptionFrame(transcript.text, "", time_now_iso8601(), Language(self._settings["language_code"]))
                self._append_accum_transcription(frame)
                if not self._is_accum_transcription(transcript.text):
                    await self._send_accum_transcriptions()
            else: # Interim transcript
                logger.debug(f"Received interim transcript: '{transcript.text}'")
                await self._handle_user_speaking()
                frame = InterimTranscriptionFrame(transcript.text, "", time_now_iso8601(), Language(self._settings["language_code"]))
                await self.push_frame(frame)

        # Schedule the async processing in the main event loop
        asyncio.run_coroutine_threadsafe(handle_push(), self.get_event_loop())

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_speaking()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_silence()

    async def _connect(self):
        if self._transcriber:
            return
        
        logger.debug("Connecting to AssemblyAI...")

        def on_open(session_opened: RealtimeSessionOpened):
            logger.info(f"{self}: Connected to AssemblyAI with session ID: {session_opened.session_id}")

        def on_error(error: RealtimeError):
            logger.error(f"{self}: An error occurred: {error}")
            asyncio.run_coroutine_threadsafe(self.push_frame(ErrorFrame(str(error))), self.get_event_loop())

        def on_close():
            logger.info(f"{self}: Disconnected from AssemblyAI")

        try:
            await self.start_ttfb_metrics()
            self._transcriber = aai.RealtimeTranscriber(
                sample_rate=self.sample_rate,
                encoding=self._settings["encoding"],
                language_code=self._settings["language_code"],
                on_data=self._on_data,
                on_error=on_error,
                on_open=on_open,
                on_close=on_close,
            )
            # The connect method starts the background thread
            self._transcriber.connect()

            # Start the async handler task
            if not self._async_handler_task:
                self._async_handler_task = self.create_monitored_task(self._async_handler)

        except Exception as e:
            logger.error(f"Failed to connect to AssemblyAI: {e}")
            raise

    async def _disconnect(self):
        if self._async_handler_task:
            await self.cancel_task(self._async_handler_task)
            self._async_handler_task = None
            
        if self._transcriber:
            logger.debug("Disconnecting from AssemblyAI...")
            # The close method stops the background thread
            self._transcriber.close()
            self._transcriber = None