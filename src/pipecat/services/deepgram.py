#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import time
import asyncio
from typing import AsyncGenerator, Dict, Optional

from loguru import logger

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
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADActiveFrame,
    VADInactiveFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import STTService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

# See .env.example for Deepgram configuration needed
try:
    from deepgram import (
        AsyncListenWebSocketClient,
        DeepgramClient,
        DeepgramClientOptions,
        ErrorResponse,
        LiveOptions,
        LiveResultResponse,
        LiveTranscriptionEvents,
        SpeakOptions,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`. Also, set `DEEPGRAM_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


DEFAULT_ON_NO_PUNCTUATION_SECONDS = 3


class DeepgramTTSService(TTSService):
    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "aura-helios-en",
        sample_rate: Optional[int] = None,
        encoding: str = "linear16",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
            "encoding": encoding,
        }
        self.set_voice(voice)
        self._deepgram_client = DeepgramClient(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        options = SpeakOptions(
            model=self._voice_id,
            encoding=self._settings["encoding"],
            sample_rate=self.sample_rate,
            container="none",
        )

        try:
            await self.start_ttfb_metrics()

            response = await asyncio.to_thread(
                self._deepgram_client.speak.v("1").stream, {"text": text}, options
            )

            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            # The response.stream_memory is already a BytesIO object
            audio_buffer = response.stream_memory

            if audio_buffer is None:
                raise ValueError("No audio data received from Deepgram")

            # Read and yield the audio data in chunks
            audio_buffer.seek(0)  # Ensure we're at the start of the buffer
            chunk_size = 8192  # Use a fixed buffer size
            while True:
                await self.stop_ttfb_metrics()
                chunk = audio_buffer.read(chunk_size)
                if not chunk:
                    break
                frame = TTSAudioRawFrame(audio=chunk, sample_rate=self.sample_rate, num_channels=1)
                yield frame

                yield TTSStoppedFrame()

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
            yield ErrorFrame(f"Error getting audio: {str(e)}")


class DeepgramSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        url: str = "",
        sample_rate: Optional[int] = None,
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS,
        live_options: Optional[LiveOptions] = None,
        addons: Optional[Dict] = None,
        **kwargs,
    ):
        sample_rate = sample_rate or (live_options.sample_rate if live_options else None)
        super().__init__(sample_rate=sample_rate, **kwargs)

        default_options = LiveOptions(
            encoding="linear16",
            language=Language.EN,
            model="nova-3-general",
            channels=1,
            interim_results=True,
            smart_format=True,
            punctuate=True,
            profanity_filter=True,
            vad_events=False,
        )

        merged_options = default_options
        if live_options:
            merged_options = LiveOptions(**{**default_options.to_dict(), **live_options.to_dict()})

        # deepgram connection requires language to be a string
        if isinstance(merged_options.language, Language) and hasattr(
            merged_options.language, "value"
        ):
            merged_options.language = merged_options.language.value

        self._settings = merged_options.to_dict()
        self._addons = addons
        self._user_speaking = False
        self._bot_speaking = True
        self._on_no_punctuation_seconds = on_no_punctuation_seconds
        self._vad_active = False

        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(
                url=url,
                options={"keepalive": "true"},  # verbose=logging.DEBUG
            ),
        )

        if self.vad_enabled:
            logger.debug(f"Deepgram VAD Enabled: {self.vad_enabled}")
            self._register_event_handler("on_speech_started")
            self._register_event_handler("on_utterance_end")

        self._transcription_accum_task = None
        self._accum_transcription_frames = []
        self._last_time_accum_transcription = time.time()
        

    @property
    def vad_enabled(self):
        return self._settings["vad_events"]

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        await super().set_model(model)
        logger.info(f"Switching STT model to: [{model}]")
        self._settings["model"] = model
        await self._disconnect()
        await self._connect()

    async def set_language(self, language: Language):
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language"] = language
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._settings["sample_rate"] = self.sample_rate
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        await self._connection.send(audio)
        yield None

    async def _connect(self):
        logger.debug("Connecting to Deepgram")

        self._connection: AsyncListenWebSocketClient = self._client.listen.asyncwebsocket.v("1")

        self._connection.on(
            LiveTranscriptionEvents(LiveTranscriptionEvents.Transcript), self._on_message
        )
        self._connection.on(LiveTranscriptionEvents(LiveTranscriptionEvents.Error), self._on_error)

        if not self._transcription_accum_task:
            self._transcription_accum_task = self.create_monitored_task(self._transcription_accum)

        if self.vad_enabled:
            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.SpeechStarted),
                self._on_speech_started,
            )
            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.UtteranceEnd),
                self._on_utterance_end,
            )

        if not await self._connection.start(options=self._settings, addons=self._addons):
            logger.error(f"{self}: unable to connect to Deepgram")

    async def _disconnect(self):

        if self._transcription_accum_task:
            await self.cancel_task(self._transcription_accum_task)

        if self._connection.is_connected:
            logger.debug("Disconnecting from Deepgram")
            await self._connection.finish()

    async def start_metrics(self):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def _on_error(self, *args, **kwargs):
        error: ErrorResponse = kwargs["error"]
        logger.warning(f"{self} connection error, will retry: {error}")
        await self.stop_all_metrics()
        # NOTE(aleix): we don't disconnect (i.e. call finish on the connection)
        # because this triggers more errors internally in the Deepgram SDK. So,
        # we just forget about the previous connection and create a new one.
        await self._connect()

    async def _on_speech_started(self, *args, **kwargs):
        await self.start_metrics()
        await self._call_event_handler("on_speech_started", *args, **kwargs)

    async def _on_utterance_end(self, *args, **kwargs):
        await self._call_event_handler("on_utterance_end", *args, **kwargs)


    async def _handle_user_speaking(self):

        await self.push_frame(StartInterruptionFrame())
        if self._user_speaking == True: return

        self._user_speaking = True

        await self.push_frame(UserStartedSpeakingFrame())

    async def _handle_user_silence(self):

        if self._user_speaking == False: return

        self._user_speaking = False
        await self.push_frame(UserStoppedSpeakingFrame())

    async def _handle_bot_speaking(self):
        self._bot_speaking = True


    async def _handle_bot_silence(self):
        self._bot_speaking = False


    def _transcript_words_count(self, transcript: str):
        return len(transcript.split(" "))
    
    async def _transcription_accum(self, task_name):

        while True:
            if not self.is_monitored_task_active(task_name): return

            await asyncio.sleep(0.1)
            
            current_time = time.time()

            if current_time - self._last_time_accum_transcription > self._on_no_punctuation_seconds and len(self._accum_transcription_frames):
                logger.debug("Sending accum transcription because of timeout")
                await self._send_accum_transcriptions()

            



    async def _send_accum_transcriptions(self):

        if not len(self._accum_transcription_frames): return

        await self._handle_user_speaking()

        for frame in self._accum_transcription_frames:
            await self.push_frame(
                frame
            )
        self._accum_transcription_frames = []

        await self._handle_user_silence()
        await self.stop_processing_metrics()

    def _is_accum_transcription(self, text: str):

        END_OF_PHRASE_CHARACTERS = ['.', '?']

        text =  text.strip()

        if not text: return True

        return not text[-1] in END_OF_PHRASE_CHARACTERS
    
    def _append_accum_transcription(self, frame: TranscriptionFrame):
        self._last_time_accum_transcription = time.time()
        self._accum_transcription_frames.append(frame)

    async def _on_final_transcript_message(self, transcript, language):

        await self._handle_user_speaking()
        frame = TranscriptionFrame(transcript, "", time_now_iso8601(), language)

        self._append_accum_transcription(frame)
        if not self._is_accum_transcription(frame.text):
            await self._send_accum_transcriptions()
    
    async def _on_interim_transcript_message(self, transcript, language):

        if self._bot_speaking and self._transcript_words_count(transcript) == 1: 
            logger.debug(f"Ignoring Deepgram interruption because bot is speaking: {transcript}")
            return
        
        if not self._vad_active:
            logger.debug("Ignoring Deepgram interruption because VAD inactive")
            return
        
        await self._handle_user_speaking()
        await self.push_frame(
            InterimTranscriptionFrame(transcript, "", time_now_iso8601(), language)
        )


    async def _on_message(self, *args, **kwargs):
        result: LiveResultResponse = kwargs["result"]

        logger.debug(result)

        if len(result.channel.alternatives) == 0:
            return
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        confidence = result.channel.alternatives[0].confidence
        language = None
        if result.channel.alternatives[0].languages:
            language = result.channel.alternatives[0].languages[0]
            language = Language(language)
        if len(transcript) > 0:
            await self.stop_ttfb_metrics()

            logger.debug(f"Transcription{'' if is_final else ' interim'}: {transcript}")
            logger.debug(f"Confidence: {confidence}")

            if is_final:
                await self._on_final_transcript_message(transcript, language)
            else:
                if confidence > 0.7:
                    await self._on_interim_transcript_message(transcript, language)
                else:
                    logger.debug("Ignoring iterim because low confidence")
                

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("Received bot started speaking on deepgram")
            await self._handle_bot_speaking()

        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("Received bot started speaking on deepgram")
            await self._handle_bot_silence()

        if isinstance(frame, UserStartedSpeakingFrame) and not self.vad_enabled:
            # Start metrics if Deepgram VAD is disabled & pipeline VAD has detected speech
            await self.start_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # https://developers.deepgram.com/docs/finalize
            await self._connection.finalize()
            logger.trace(f"Triggered finalize event on: {frame.name}, {direction}")
        
        if isinstance(frame, VADInactiveFrame):
            self._vad_active = False
        elif isinstance(frame, VADActiveFrame):
            self._vad_active = True