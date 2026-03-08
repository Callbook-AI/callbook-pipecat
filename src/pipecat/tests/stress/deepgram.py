#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import uuid
from pathlib import Path

import asyncio, statistics, time, wave, math
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

from pipecat.utils.asyncio import TaskManager
from pipecat.clocks.system_clock import SystemClock

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


DG_LOG_FILE = Path("dg_received_messages.jsonl")
DG_ERROR_LOCK = asyncio.Lock()
DG_ERRORS = 0


class DeepgramSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        callback: Callable[[float], None] | None = None,   # 🆕 callback
        url: str = "",
        sample_rate: Optional[int] = None,
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS,
        live_options: Optional[LiveOptions] = None,
        addons: Optional[Dict] = None,
        detect_voicemail: bool = True,
        **kwargs,
    ):
        
        self._callback      = callback        # 🆕 store the hook
        self._send_ts       = None            # 🆕 first-chunk timestamp
        self._final_fired   = False           # 🆕 guard
        self._instance_id   = uuid.uuid4().hex

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

        self.start_time = time.time()


    @property
    def vad_enabled(self):
        return self._settings["vad_events"]

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
        if self._send_ts is None:
            self._send_ts = time.perf_counter()
        await self._connection.send(audio)

    async def _connect(self):
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

    async def _disconnect(self):

        if self._async_handler_task:
            await self.cancel_task(self._async_handler_task)

        if self._connection.is_connected:
            logger.debug("Disconnecting from Deepgram")
            await self._connection.finish()

    async def start_metrics(self):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()

    async def _on_error(self, *args, **kwargs):
        """
        Fatal error handler.

        • increments the global DG_ERRORS counter  
        • stops all metrics, closes the websocket, cancels the async-handler  
        • *does NOT* try to reconnect – we just bubble the exception up so the
          caller can decide what to do (the benchmark will treat it as a
          failed run)
        """
        global DG_ERRORS
        error: ErrorResponse = kwargs["error"]
        logger.error(f"{self} fatal connection error: {error}")

        async with DG_ERROR_LOCK:
            DG_ERRORS += 1

        await self.stop_all_metrics()
        await self._disconnect()

        # propagate so the surrounding task finishes immediately
        raise RuntimeError(f"Deepgram connection error: {error}") from None
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
        if not self._is_accum_transcription(frame.text) or speech_final:
            await self._send_accum_transcriptions()
    
    async def _on_interim_transcript_message(self, transcript, language, start_time):
        
        self._last_interim_time = time.time()
        await self._handle_user_speaking()
        await self.push_frame(
            InterimTranscriptionFrame(transcript, "", time_now_iso8601(), language)
        )



    async def _on_message(self, *args, **kwargs):
        global DG_ERRORS
        result: LiveResultResponse = kwargs["result"]

        # fire the latency callback on the first *final* transcript (= success)
        if result.is_final and not self._final_fired and self._callback:
            self._final_fired = True
            self._callback(time.perf_counter())

        DG_LOG_FILE.parent.mkdir(exist_ok=True)
        with DG_LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(result.to_json())
            fh.write("\n")

        

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



DEEPGRAM_API_KEY = ""
AUDIO_FILE       = Path("pipecat/tests/stress/sample_16k_mono.wav")    # ~5 s of speech
CHUNK_MS         = 20                             # send 20 ms frames
SPEECH_END_SEC   = 3                          # speaker stops at 5 s


def _percentile(data, pct):
    if not data: return float("nan")
    k = max(0, min(len(data)-1, int(round((pct/100)*len(data)+0.5))-1))
    return sorted(data)[k]


async def stream_file(service: DeepgramSTTService, wav_path: Path):
    """Send the WAV in realtime (20 ms chunks)."""
    with wave.open(str(wav_path), "rb") as wf:
        bytes_per_frame = wf.getsampwidth() * wf.getnchannels()
        frames_per_chunk = int(wf.getframerate() * CHUNK_MS / 1000)
        while True:
            frames = wf.readframes(frames_per_chunk)
            if not frames:
                break
            await service.run_stt(frames)
            await asyncio.sleep(CHUNK_MS / 1000)


async def time_response(wav_path: Path) -> float:
    """Return post-speech latency in ms."""
    latency_holder = []


    def final_callback(ts: float):
        # ts is perf_counter() when final transcript arrived
        latency_holder.append((ts - start_ts - SPEECH_END_SEC) * 1000)

    dg = DeepgramSTTService(api_key=DEEPGRAM_API_KEY,
        live_options=LiveOptions(language='es', vad_events=False, model='nova-2', filler_words=True, profanity_filter=False), 
        callback=final_callback)

    clock = SystemClock()
    tm = TaskManager(); tm.set_event_loop(asyncio.get_running_loop())
    await dg.process_frame(StartFrame(clock=clock, task_manager=tm),
                           direction=FrameDirection.UPSTREAM)
    start_ts = time.perf_counter()
    await stream_file(dg, wav_path)        # realtime playback
    await dg.stop(EndFrame())              # flush / close

    return latency_holder[0] if latency_holder else float("nan")


async def run_benchmark(concurrency=50):
    tasks = [asyncio.create_task(time_response(AUDIO_FILE)) for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)
    lat = [v for v in results if v == v]

    print("Runs:        ", len(lat))
    print("Errors:      ", DG_ERRORS)
    print("\n— Post-speech latency (ms) —")
    print("Min:         ", f"{min(lat):.2f}")
    print("Max:         ", f"{max(lat):.2f}")
    print("Mean:        ", f"{statistics.mean(lat):.2f}")
    print("Median:      ", f"{statistics.median(lat):.2f}")
    print("Std dev:     ", f"{statistics.stdev(lat):.2f}" if len(lat) > 1 else "n/a")
    print("95th pct:    ", f"{_percentile(lat, 95):.2f}")
    print("99th pct:    ", f"{_percentile(lat, 99):.2f}")


if __name__ == "__main__":
    print(AUDIO_FILE)
    asyncio.run(run_benchmark())