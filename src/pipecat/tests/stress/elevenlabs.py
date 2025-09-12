
import time
import asyncio
import statistics
import uuid
import json
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Tuple, Union

from loguru import logger
from pydantic import BaseModel, model_validator

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import InterruptibleWordTTSService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio import TaskManager
from pipecat.clocks.system_clock import SystemClock
from pathlib import Path



import websockets


ELEVENLABS_MULTILINGUAL_MODELS = {
    "eleven_flash_v2_5",
    "eleven_turbo_v2_5",
}

def language_to_elevenlabs_language(language: Language) -> Optional[str]:
    BASE_LANGUAGES = {
        Language.AR: "ar",
        Language.BG: "bg",
        Language.CS: "cs",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.FI: "fi",
        Language.FIL: "fil",
        Language.FR: "fr",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.ID: "id",
        Language.IT: "it",
        Language.JA: "ja",
        Language.KO: "ko",
        Language.MS: "ms",
        Language.NL: "nl",
        Language.NO: "no",
        Language.PL: "pl",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SK: "sk",
        Language.SV: "sv",
        Language.TA: "ta",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.VI: "vi",
        Language.ZH: "zh",
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


def output_format_from_sample_rate(sample_rate: int) -> str:
    match sample_rate:
        case 16000:
            return "pcm_16000"
        case 22050:
            return "pcm_22050"
        case 24000:
            return "pcm_24000"
        case 44100:
            return "pcm_44100"
    logger.warning(
        f"ElevenLabsTTSService: No output format available for {sample_rate} sample rate"
    )
    return "pcm_16000"


def build_elevenlabs_voice_settings(
    settings: Dict[str, Any],
) -> Optional[Dict[str, Union[float, bool]]]:
    """Build voice settings dictionary for ElevenLabs based on provided settings.

    Args:
        settings: Dictionary containing voice settings parameters

    Returns:
        Dictionary of voice settings or None if required parameters are missing
    """
    voice_settings = {}
    if settings["stability"] is not None and settings["similarity_boost"] is not None:
        voice_settings["stability"] = settings["stability"]
        voice_settings["similarity_boost"] = settings["similarity_boost"]
        if settings["style"] is not None:
            voice_settings["style"] = settings["style"]
        if settings["use_speaker_boost"] is not None:
            voice_settings["use_speaker_boost"] = settings["use_speaker_boost"]
        if settings["speed"] is not None:
            voice_settings["speed"] = settings["speed"]
    else:
        if settings["style"] is not None:
            logger.warning(
                "'style' is set but will not be applied because 'stability' and 'similarity_boost' are not both set."
            )
        if settings["use_speaker_boost"] is not None:
            logger.warning(
                "'use_speaker_boost' is set but will not be applied because 'stability' and 'similarity_boost' are not both set."
            )
        if settings["speed"] is not None:
            logger.warning(
                "'speed' is set but will not be applied because 'stability' and 'similarity_boost' are not both set."
            )

    return voice_settings or None



# ── global helpers ────────────────────────────────────────────────────
LOG_FILE   = Path("received_messages.jsonl")              # all messages land here
ERROR_LOCK = asyncio.Lock()                               # protects the counter
ERRORS     = 0                                            # global error counter


class ElevenLabsTTSService(InterruptibleWordTTSService):
    class InputParams(BaseModel):
        language: Optional[Language] = None
        optimize_streaming_latency: Optional[str] = None
        stability: Optional[float] = None
        similarity_boost: Optional[float] = None
        style: Optional[float] = None
        use_speaker_boost: Optional[bool] = None
        speed: Optional[float] = None
        auto_mode: Optional[bool] = True

        @model_validator(mode="after")
        def validate_voice_settings(self):
            stability = self.stability
            similarity_boost = self.similarity_boost
            if (stability is None) != (similarity_boost is None):
                raise ValueError(
                    "Both 'stability' and 'similarity_boost' must be provided when using voice settings"
                )
            return self

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "eleven_flash_v2_5",
        url: str = "wss://api.elevenlabs.io",
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        callback,
        **kwargs,
    ):
        # Aggregating sentences still gives cleaner-sounding results and fewer
        # artifacts than streaming one word at a time. On average, waiting for a
        # full sentence should only "cost" us 15ms or so with GPT-4o or a Llama
        # 3 model, and it's worth it for the better audio quality.
        #
        # We also don't want to automatically push LLM response text frames,
        # because the context aggregators will add them to the LLM context even
        # if we're interrupted. ElevenLabs gives us word-by-word timestamps. We
        # can use those to generate text frames ourselves aligned with the
        # playout timing of the audio!
        #
        # Finally, ElevenLabs doesn't provide information on when the bot stops
        # speaking for a while, so we want the parent class to send TTSStopFrame
        # after a short period not receiving any audio.
        super().__init__(
            aggregate_sentences=True,
            push_text_frames=False,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=sample_rate,
            **kwargs,
        )

        self._api_key = api_key
        self._url = url
        self._settings = {
            "language": self.language_to_service_language(params.language)
            if params.language
            else None,
            "optimize_streaming_latency": params.optimize_streaming_latency,
            "stability": params.stability,
            "similarity_boost": params.similarity_boost,
            "style": params.style,
            "use_speaker_boost": params.use_speaker_boost,
            "speed": params.speed,
            "auto_mode": str(params.auto_mode).lower(),
        }
        self.set_model_name(model)
        self.set_voice(voice_id)
        self._output_format = ""  # initialized in start()
        self._voice_settings = self._set_voice_settings()

        # Indicates if we have sent TTSStartedFrame. It will reset to False when
        # there's an interruption or TTSStoppedFrame.
        self._started = False
        self._cumulative_time = 0

        self._receive_task = None
        self._keepalive_task = None

        self._callback = callback                   # store the callback
        self._send_ts: Optional[float] = None       # when we last sent text
        self._ttfb_reported = False                 # guard so we report only once
        self._instance_id = uuid.uuid4().hex 

    def can_generate_metrics(self) -> bool:
        return True

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_elevenlabs_language(language)

    def _set_voice_settings(self):
        return build_elevenlabs_voice_settings(self._settings)

    async def set_model(self, model: str):
        await super().set_model(model)
        logger.info(f"Switching TTS model to: [{model}]")
        await self._disconnect()
        await self._connect()

    async def _update_settings(self, settings: Mapping[str, Any]):
        prev_voice = self._voice_id
        await super()._update_settings(settings)
        if not prev_voice == self._voice_id:
            await self._disconnect()
            await self._connect()
            logger.info(f"Switching TTS voice to: [{self._voice_id}]")

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._output_format = output_format_from_sample_rate(self.sample_rate)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        if self._websocket:
            msg = {"text": " ", "flush": True}
            await self._websocket.send(json.dumps(msg))
    
    
    async def flush_audio_to_ignore(self):

        if self._started:
            logger.debug("Flushing to ignore")
            self._started = False
            await self.flush_audio()

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        await super().push_frame(frame, direction)
        if isinstance(frame, (TTSStoppedFrame, StartInterruptionFrame)):

            await self.flush_audio_to_ignore()

            self._started = False
            if isinstance(frame, TTSStoppedFrame):
                await self.add_word_timestamps([("LLMFullResponseEndFrame", 0), ("Reset", 0)])

    async def _connect(self):
        logger.debug("CONNECTING")
        await self._connect_websocket()

        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self.push_error))


        if not self._keepalive_task:
            self._keepalive_task = self.create_monitored_task(self._keepalive_task_handler)

    async def _disconnect(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

        if self._keepalive_task:
            await self.cancel_task(self._keepalive_task)
            self._keepalive_task = None

        await self._disconnect_websocket()

    async def _connect_websocket(self):
        try:
            if self._websocket:
                return

            logger.debug("Connecting to ElevenLabs")

            voice_id = self._voice_id
            model = self.model_name
            output_format = self._output_format
            url = f"{self._url}/v1/text-to-speech/{voice_id}/stream-input?model_id={model}&output_format={output_format}&auto_mode={self._settings['auto_mode']}"

            if self._settings["optimize_streaming_latency"]:
                url += f"&optimize_streaming_latency={self._settings['optimize_streaming_latency']}"

            # Language can only be used with the ELEVENLABS_MULTILINGUAL_MODELS
            language = self._settings["language"]
            if model in ELEVENLABS_MULTILINGUAL_MODELS and language is not None:
                url += f"&language_code={language}"
                logger.debug(f"Using language code: {language}")
            elif language is not None:
                logger.warning(
                    f"Language code [{language}] not applied. Language codes can only be used with multilingual models: {', '.join(sorted(ELEVENLABS_MULTILINGUAL_MODELS))}"
                )

            # Set max websocket message size to 16MB for large audio responses
            self._websocket = await websockets.connect(url, max_size=16 * 1024 * 1024)

            # According to ElevenLabs, we should always start with a single space.
            msg: Dict[str, Any] = {
                "text": " ",
                "xi_api_key": self._api_key,
            }
            if self._voice_settings:
                msg["voice_settings"] = self._voice_settings
            await self._websocket.send(json.dumps(msg))
            logger.debug("Connected")
        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from ElevenLabs")
                await self._websocket.send(json.dumps({"text": ""}))
                await self._websocket.close()
                self._websocket = None

            self._started = False
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    # replace the old _receive_messages with the one below
    async def _receive_messages(self):
        global ERRORS
        """Receive every websocket message, log it, and count errors."""
        async for raw in self._get_websocket():

            # existing TTFB logic -------------------------------------------------
            if self._send_ts and not self._ttfb_reported:
                ttfb = time.perf_counter() - self._send_ts
                self._callback(ttfb)
                self._ttfb_reported = True

            # 1️⃣  persist the raw JSON line-by-line
            LOG_FILE.parent.mkdir(exist_ok=True)
            with LOG_FILE.open("a", encoding="utf-8") as fh:
                fh.write(raw)
                fh.write("\n")
                
            try:
                payload = json.loads(raw)
                if "audio" not in payload:
                    async with ERROR_LOCK:
                        ERRORS += 1
            except json.JSONDecodeError:
                async with ERROR_LOCK:
                    ERRORS += 1

            
            

    async def _keepalive_task_handler(self, task_name):
        while True:
            if not self.is_monitored_task_active(task_name): return 

            await asyncio.sleep(10)
            try:
                await self._send_text("")
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break


    async def _send_text(self, text: str):
        if self._websocket:
            self._send_ts = time.perf_counter()
            self._ttfb_reported = False
            msg = {"text": text + " "}
            await self._websocket.send(json.dumps(msg))
            logger.debug("message sent")

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket:
                await self._connect()

            try:
                logger.debug("sending text")
                await self._send_text(text)
            except Exception as e:
                logger.debug(f"{self} error sending message: {e}")
                await self._disconnect()
                await self._connect()
                return
        except Exception as e:
            logger.error(f"{self} exception: {e}")

ELEVEN_LABS_API_KEY = ''
voice_id = ''




# ── keep your existing imports / ElevenLabsTTSService definition here ──

def _percentile(data: List[float], pct: float) -> float:
    """Return the given percentile (e.g. 95 for p95)."""
    if not data:
        return float("nan")
    k = max(0, min(len(data) - 1, int(round((pct / 100) * len(data) + 0.5)) - 1))
    return sorted(data)[k]

async def time_response(text: str) -> float:
    """Run one TTS request and return the TTFB (ms)."""
    ttfb_holder: List[float] = []

    # closure to grab the measurement instead of printing it
    def collect_ttfb(ttfb_sec: float):
        ttfb_holder.append(ttfb_sec * 1_000)  # convert → ms

    eleven = ElevenLabsTTSService(
        api_key=ELEVEN_LABS_API_KEY,
        voice_id=voice_id,
        params=ElevenLabsTTSService.InputParams(
            similarity_boost=0.75,
            stability=0.5,
            speed=1,
            language=Language.ES,
            auto_mode=False,
        ),
        callback=collect_ttfb,
    )

    clock = SystemClock()
    task_manager = TaskManager()
    task_manager.set_event_loop(asyncio.get_running_loop())

    await eleven.process_frame(StartFrame(clock=clock, task_manager=task_manager),
                               direction=FrameDirection.UPSTREAM)
    await asyncio.sleep(1)                 # keep loop alive long enough to receive
    await eleven.run_tts(text)             # _send_text() starts the timer
    await eleven.flush_audio()
    await asyncio.sleep(4)                 # wait for audio + callback
    await eleven.stop(EndFrame())

    return ttfb_holder[0] if ttfb_holder else float("nan")

async def run_benchmark(concurrency: int = 300, text: str = "Hola, ¿cómo estás?"):
    # launch N coroutines at once
    tasks = [asyncio.create_task(time_response(text)) for _ in range(concurrency)]
    results_ms = await asyncio.gather(*tasks)

    # drop failed runs (NaN) if any
    latencies = [v for v in results_ms if v == v]

    print("Runs:        ", len(latencies))
    print("Errors:      ", ERRORS)
    print("\n— Latency results (ms) —")
    print("Min:         ", f"{min(latencies):.2f}")
    print("Max:         ", f"{max(latencies):.2f}")
    print("Mean:        ", f"{statistics.mean(latencies):.2f}")
    print("Median:      ", f"{statistics.median(latencies):.2f}")
    print("Std dev:     ", f"{statistics.stdev(latencies):.2f}" if len(latencies) > 1 else "n/a")
    print("95th pct:    ", f"{_percentile(latencies, 95):.2f}")
    print("99th pct:    ", f"{_percentile(latencies, 99):.2f}")

if __name__ == "__main__":
    asyncio.run(run_benchmark())