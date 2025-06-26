#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import re
import time
import asyncio
from typing import AsyncGenerator, Dict, Optional, Callable


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
    VADInactiveFrame,
    VoicemailFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import STTService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.rex import regex_list_matches
from pipecat.utils.string import is_equivalent_basic
from pipecat.utils.text import voicemail


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
IGNORE_REPEATED_MSG_AT_START_SECONDS = 4
VOICEMAIL_DETECTION_SECONDS = 10
FALSE_INTERIM_SECONDS = 1.3


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


class DeepgramSiDetector:
    """
    Connects to Deepgram in Spanish, streams raw audio,
    and calls `callback(transcript)` whenever a final
    transcript contains "si", "sí", "si!", "sí!", etc.
    """
    def __init__(
        self,
        api_key: str,
        callback: Callable[[str], None],
        url: str = "wss://api.deepgram.com/v1/listen",
        sample_rate: int = 16_000,
    ):
        # callback to invoke on detection
        self._callback = callback
        # match standalone si or sí (case-insensitive), optional !/¡
        self._pattern = re.compile(r'\b(?:si|sí)\b[!¡]?', re.IGNORECASE)

        # build a Deepgram client
        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(
                url=url,
                options={"keepalive": "true"}
            ),
        )

        # settings for Spanish transcription
        self._settings = {
            "encoding":       "linear16",
            "language":       "es",          # Spanish
            "model":          "general",
            "sample_rate":    sample_rate,
            "interim_results": False,        # only finals
            "punctuate":      True,
        }

        self._conn = None  # will hold the websocket client
        self.start_times = set()

    async def start(self):
        """
        Open the Deepgram websocket. Must be called before sending audio.
        """
        try:
            self._conn = self._client.listen.asyncwebsocket.v("1")
            self._conn.on(
                LiveTranscriptionEvents.Transcript,
                self._on_transcript_event
            )
            await self._conn.start(options=self._settings)
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector.start: {e}")

    async def send_audio(self, chunk: bytes):
        """
        Send a chunk of raw audio (bytes) to Deepgram.
        """
        try:
            if not self._conn or not self._conn.is_connected:
                raise RuntimeError("Connection not started. Call start() first.")
            await self._conn.send(chunk)
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector.send_audio: {e}")

    async def stop(self):
        """
        Gracefully close the Deepgram websocket when you’re done sending.
        """
        try:
            if self._conn and self._conn.is_connected:
                await self._conn.finish()
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector.stop: {e}")

    async def _on_transcript_event(self, *args, **kwargs):
        """
        Internal handler for every transcription event.
        Filters down to final transcripts and applies the "si" regex.
        """
        try:
            result = kwargs.get("result")
            if not result:
                return

            alts = result.channel.alternatives
            if not alts:
                return

            transcript = alts[0].transcript.strip()
            if self._pattern.search(transcript):

                if result.start in self.start_times: return
                logger.debug("Si detected")
                
                self.start_times.add(result.start)
                
                logger.debug("Si detected")

                await self._callback(result)
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector._on_transcript_event: {e}")


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
        detect_voicemail: bool = True,  
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
        
        self.language = merged_options.language
        self.api_key = api_key
        self.detect_voicemail = detect_voicemail  

        self._settings = merged_options.to_dict()
        self._addons = addons
        self._user_speaking = False
        self._bot_speaking = True
        self._on_no_punctuation_seconds = on_no_punctuation_seconds
        self._vad_active = False

        self._first_message = None
        self._first_message_time = None
        self._last_interim_time = None


        self._setup_sibling_deepgram()


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

        self._async_handler_task = None
        self._accum_transcription_frames = []
        self._last_time_accum_transcription = time.time()
        self._last_time_transcription = time.time()
        self._was_first_transcript_receipt = False

        self.start_time = time.time()


    @property
    def vad_enabled(self):
        return self._settings["vad_events"]
    
    def _setup_sibling_deepgram(self):

        self._sibling_deepgram = None

        if self.language != 'it': return 

        self._sibling_deepgram = DeepgramSiDetector(self.api_key, self.sibling_transcript_handler)


    async def sibling_transcript_handler(self, result: LiveResultResponse):

        result_time = result.start

        if abs(self._last_time_transcription - result_time) < 0.5:
            logger.debug("Ignoring 'Si' because recent transcript")
            return
        
        await self._on_message(result=result)


    def _time_since_init(self):
        return time.time() - self.start_time

    def can_generate_metrics(self) -> bool:
        return True

    async def set_model(self, model: str):
        try:
            await super().set_model(model)
            logger.info(f"Switching STT model to: [{model}]")
            self._settings["model"] = model
            await self._disconnect()
            await self._connect()
        except Exception as e:
            logger.exception(f"{self} exception in set_model: {e}")
            raise

    async def set_language(self, language: Language):
        try:
            logger.info(f"Switching STT language to: [{language}]")
            self._settings["language"] = language
            await self._disconnect()
            await self._connect()
        except Exception as e:
            logger.exception(f"{self} exception in set_language: {e}")
            raise

    async def start(self, frame: StartFrame):
        try:
            await super().start(frame)
            self._settings["sample_rate"] = self.sample_rate
            await self._connect()

            if self._sibling_deepgram:
                await self._sibling_deepgram.start()
        except Exception as e:
            logger.exception(f"{self} exception in start: {e}")
            raise

    async def stop(self, frame: EndFrame):

        try:
            await super().stop(frame)
            await self._disconnect()

            if self._sibling_deepgram:
                await self._sibling_deepgram.stop()
        except Exception as e:
            logger.exception(f"{self} exception in stop: {e}")
            raise

    async def cancel(self, frame: CancelFrame):

        try:
            await super().cancel(frame)
            await self._disconnect()
        except Exception as e:
            logger.exception(f"{self} exception in cancel: {e}")
            raise

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            await self._connection.send(audio)
            if self._sibling_deepgram:
                await self._sibling_deepgram.send_audio(audio)
            yield None
        except Exception as e:
            logger.exception(f"{self} exception in run_stt: {e}")
            yield ErrorFrame(f"run_stt error: {e}")

    async def _connect(self):
        try:
            logger.debug("Connecting to Deepgram")

            self._connection: AsyncListenWebSocketClient = self._client.listen.asyncwebsocket.v("1")

            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.Transcript), self._on_message
            )
            self._connection.on(LiveTranscriptionEvents(LiveTranscriptionEvents.Error), self._on_error)

            if not self._async_handler_task:
                self._async_handler_task = self.create_monitored_task(self._async_handler)

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
        except Exception as e:
            logger.exception(f"{self} exception in _connect: {e}")
            raise

    async def _disconnect(self):
        try:
            if self._async_handler_task:
                await self.cancel_task(self._async_handler_task)

            if self._connection.is_connected:
                logger.debug("Disconnecting from Deepgram")
                await self._connection.finish()
        except Exception as e:
            logger.exception(f"{self} exception in _disconnect: {e}")

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

    async def _async_handle_accum_transcription(self, current_time):

        if current_time - self._last_time_accum_transcription > self._on_no_punctuation_seconds and len(self._accum_transcription_frames):
            logger.debug("Sending accum transcription because of timeout")
            await self._send_accum_transcriptions()

    async def _handle_false_interim(self, current_time):

        if not self._user_speaking: return
        if not self._last_interim_time: return
        if self._vad_active: return

        last_interim_delay = current_time - self._last_interim_time

        if last_interim_delay > FALSE_INTERIM_SECONDS: return

        logger.debug("False interim detected")

        await self._handle_user_silence()


    
    async def _async_handler(self, task_name):

        while True:
            if not self.is_monitored_task_active(task_name): return

            await asyncio.sleep(0.1)
            
            current_time = time.time()

            await self._async_handle_accum_transcription(current_time)
            await self._handle_false_interim(current_time)



            
    
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


    def _handle_first_message(self, text):

        if self._first_message: return 

        self._first_message = text
        self._first_message_time = time.time()

    def _should_ignore_first_repeated_message(self, text):

        if not self._first_message: return
        
        time_since_first_message = time.time() - self._first_message_time
        if time_since_first_message > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        return is_equivalent_basic(text, self._first_message)



    async def _on_final_transcript_message(self, transcript, language, speech_final: bool):

        await self._handle_user_speaking()
        frame = TranscriptionFrame(transcript, "", time_now_iso8601(), language)

        self._handle_first_message(frame.text)
        self._append_accum_transcription(frame)
        self._was_first_transcript_receipt = True
        if not self._is_accum_transcription(frame.text) or speech_final:
            await self._send_accum_transcriptions()
    
    async def _on_interim_transcript_message(self, transcript, language, start_time):
        
        self._last_interim_time = time.time()
        await self._handle_user_speaking()
        await self.push_frame(
            InterimTranscriptionFrame(transcript, "", time_now_iso8601(), language)
        )

    async def _should_ignore_transcription(self, result: LiveResultResponse):

        is_final = result.is_final
        confidence = result.channel.alternatives[0].confidence
        transcript = result.channel.alternatives[0].transcript
        time_start = result.channel.alternatives[0].words[0].start if result.channel.alternatives[0].words else 0
        
        if not is_final and confidence < 0.7:
            logger.debug("Ignoring iterim because low confidence")
            return True

        if time_start < 1 and self._transcript_words_count(transcript) == 1:
            logger.debug("Ignoring first message, fast greeting")
            return True
        
        if self._should_ignore_first_repeated_message(transcript):
            logger.debug("Ignoring repeated first message")
            return True
        
        if not self._vad_active and not is_final:
            logger.debug("Ignoring Deepgram interruption because VAD inactive")
            return True
        
        if self._bot_speaking and self._transcript_words_count(transcript) == 1: 
            logger.debug(f"Ignoring Deepgram interruption because bot is speaking: {transcript}")
            return True

        return False
    

    async def _detect_and_handle_voicemail(self, transcript: str):

        if not self.detect_voicemail: return False

        logger.debug(transcript)
        logger.debug(self._time_since_init())
        
        if self._time_since_init() > VOICEMAIL_DETECTION_SECONDS and self._was_first_transcript_receipt: return False
        
        if not voicemail.is_text_voicemail(transcript): return False
        
        logger.debug("Voicemail detected")

        await self.push_frame(
            VoicemailFrame(transcript)
        )

        logger.debug("Voicemail pushed")
        return True



    async def _on_message(self, *args, **kwargs):
        try:
            result: LiveResultResponse = kwargs["result"]
            
            logger.debug(result)
            
            if len(result.channel.alternatives) == 0:
                return
            
            is_final = result.is_final
            speech_final = result.speech_final
            transcript = result.channel.alternatives[0].transcript
            confidence = result.channel.alternatives[0].confidence
            start_time = result.start
            
            language = None
            if result.channel.alternatives[0].languages:
                language = result.channel.alternatives[0].languages[0]
                language = Language(language)
            
            if len(transcript) > 0:
                await self.stop_ttfb_metrics()


                if await self._detect_and_handle_voicemail(transcript):
                    return 
                
                logger.debug(f"Transcription{'' if is_final else ' interim'}: {transcript}")
                logger.debug(f"Confidence: {confidence}")
                
                if await self._should_ignore_transcription(result):
                    return
                
                if is_final:
                    await self._on_final_transcript_message(transcript, language, speech_final)
                    self._last_time_transcription = start_time
                else:
                    await self._on_interim_transcript_message(transcript, language, start_time)

        except Exception as e:
            # full traceback will be logged
            logger.exception(f"{self} unexpected error in _on_message: {e}")


    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("Received bot started speaking on deepgram")
            await self._handle_bot_speaking()

        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("Received bot stopped speaking on deepgram")
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