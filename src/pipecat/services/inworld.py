#
# Inworld TTS Service
#
# WebSocket-based TTS using Inworld AI's streaming bidirectional API.
# https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech-websocket
#

import asyncio
import base64
import json
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional, Tuple

from loguru import logger
from pydantic import BaseModel

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    StartFrame,
    StartInterruptionFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import InterruptibleWordTTSService
from pipecat.transcriptions.language import Language

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Inworld TTS, you need to `pip install pipecat-ai[inworld]`. "
        "Also, set `INWORLD_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


InworldAudioEncoding = Literal["LINEAR16", "MP3", "OGG_OPUS", "ALAW", "MULAW"]


def output_encoding_from_sample_rate(sample_rate: int) -> str:
    """LINEAR16 is the only raw PCM encoding Inworld supports.
    MP3/OGG add headers that the pipeline can't handle directly.
    """
    return "LINEAR16"


def calculate_word_times(
    word_alignment: Mapping[str, Any], cumulative_time: float
) -> List[Tuple[str, float]]:
    """Convert Inworld word alignment into (word, timestamp) pairs."""
    words = word_alignment.get("words", [])
    start_times = word_alignment.get("wordStartTimeSeconds", [])

    word_times = []
    for word, start in zip(words, start_times):
        word_times.append((word, cumulative_time + start))

    return word_times


class InworldTTSService(InterruptibleWordTTSService):

    class InputParams(BaseModel):
        speaking_rate: Optional[float] = None  # 0.5 - 1.5
        temperature: Optional[float] = None    # 0.0 - 2.0
        auto_mode: Optional[bool] = True

    def __init__(
        self,
        *,
        api_key: str,
        voice_id: str,
        model: str = "v1.5",
        url: str = "wss://api.inworld.ai",
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        **kwargs,
    ):
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
        self._context_id: Optional[str] = None
        self._settings = {
            "speaking_rate": params.speaking_rate,
            "temperature": params.temperature,
            "auto_mode": params.auto_mode,
        }
        self.set_model_name(model)
        self.set_voice(voice_id)

        self._started = False
        self._cumulative_time = 0

        self._receive_task = None
        self._keepalive_task = None

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def flush_audio(self):
        if self._websocket and self._context_id:
            msg = {
                "flush_context": {},
                "contextId": self._context_id,
            }
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

            logger.debug("Connecting to Inworld TTS")

            ws_url = f"{self._url}/tts/v1/voice:streamBidirectional"

            self._websocket = await websockets.connect(
                ws_url,
                additional_headers={"authorization": f"Basic {self._api_key}"},
                max_size=16 * 1024 * 1024,
            )

            # Send create context message
            create_msg = self._build_create_message()
            await self._websocket.send(json.dumps(create_msg))

            # Wait for contextCreated response
            response = await self._websocket.recv()
            result = json.loads(response)
            ctx_created = result.get("result", {})
            if ctx_created.get("contextCreated"):
                self._context_id = ctx_created.get("contextId")
                logger.debug(f"Inworld TTS context created: {self._context_id}")
            else:
                status = ctx_created.get("status", {})
                logger.error(f"Failed to create Inworld TTS context: {status}")
                self._websocket = None

        except Exception as e:
            logger.error(f"{self} initialization error: {e}")
            self._websocket = None

    def _build_create_message(self) -> Dict[str, Any]:
        audio_config: Dict[str, Any] = {
            "audioEncoding": "LINEAR16",
            "sampleRateHertz": self.sample_rate,
        }

        if self._settings["speaking_rate"] is not None:
            audio_config["speakingRate"] = self._settings["speaking_rate"]

        create: Dict[str, Any] = {
            "voiceId": self._voice_id,
            "modelId": self.model_name,
            "audioConfig": audio_config,
            "timestampType": "WORD",
            "autoMode": bool(self._settings.get("auto_mode", True)),
        }

        if self._settings["temperature"] is not None:
            create["temperature"] = self._settings["temperature"]

        return {"create": create}

    async def _disconnect_websocket(self):
        try:
            await self.stop_all_metrics()

            if self._websocket:
                logger.debug("Disconnecting from Inworld TTS")
                if self._context_id:
                    close_msg = {
                        "close_context": {},
                        "contextId": self._context_id,
                    }
                    await self._websocket.send(json.dumps(close_msg))
                await self._websocket.close()
                self._websocket = None
                self._context_id = None

            self._started = False
        except Exception as e:
            logger.error(f"{self} error closing websocket: {e}")

    def _get_websocket(self):
        if self._websocket:
            return self._websocket
        raise Exception("Websocket not connected")

    async def _receive_messages(self):
        async for message in self._get_websocket():
            if not self._started:
                logger.debug("Ignoring message, not started")
                continue

            msg = json.loads(message)
            result = msg.get("result", {})

            if result.get("audioChunk"):
                chunk = result["audioChunk"]
                audio_content = chunk.get("audioContent")
                if audio_content:
                    await self.stop_ttfb_metrics()
                    self.start_word_timestamps()

                    audio = base64.b64decode(audio_content)
                    frame = TTSAudioRawFrame(audio, self.sample_rate, 1)
                    await self.push_frame(frame)

                # Process word alignment timestamps
                timestamp_info = chunk.get("timestampInfo", {})
                word_alignment = timestamp_info.get("wordAlignment")
                if word_alignment and word_alignment.get("words"):
                    word_times = calculate_word_times(
                        word_alignment, self._cumulative_time
                    )
                    if word_times:
                        await self.add_word_timestamps(word_times)
                        self._cumulative_time = word_times[-1][1]

            elif result.get("flushCompleted"):
                logger.debug(f"Inworld flush completed for context {result.get('contextId')}")

            elif result.get("contextClosed"):
                logger.debug(f"Inworld context closed: {result.get('contextId')}")

    async def _keepalive_task_handler(self, task_name):
        while True:
            if not self.is_monitored_task_active(task_name):
                return
            await asyncio.sleep(60)  # Inworld has 10 min inactivity timeout
            try:
                # Send empty flush as keepalive
                if self._websocket and self._context_id:
                    msg = {"flush_context": {}, "contextId": self._context_id}
                    await self._websocket.send(json.dumps(msg))
            except websockets.ConnectionClosed as e:
                logger.warning(f"{self} keepalive error: {e}")
                break

    async def _send_text(self, text: str):
        if self._websocket and self._context_id:
            msg = {
                "send_text": {
                    "text": text,
                    "flush_context": {},
                },
                "contextId": self._context_id,
            }
            logger.debug(f"{self}::Sending text: {text}")
            await self._websocket.send(json.dumps(msg))

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        try:
            if not self._websocket:
                await self._connect()

            try:
                if not self._started:
                    await self.start_ttfb_metrics()
                    yield TTSStartedFrame()
                    self._started = True
                    self._cumulative_time = 0

                await self._send_text(text)
                await self.start_tts_usage_metrics(text)
            except Exception as e:
                logger.error(f"{self} error sending message: {e}")
                yield TTSStoppedFrame()
                await self._disconnect()
                await self._connect()
                return
            yield None
        except Exception as e:
            logger.error(f"{self} exception: {e}")
