#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""AssemblyAI speech-to-text service implementation.

This module provides integration with AssemblyAI's real-time speech-to-text
WebSocket API for streaming audio transcription. It also includes the data
models for handling AssemblyAI's WebSocket messages and the base STT service
classes.
"""

import asyncio
import io
import json
import wave
from abc import abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Literal, Mapping, Optional
from urllib.parse import urlencode

from loguru import logger
from pydantic import BaseModel, Field

from pipecat import __version__ as pipecat_version
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    STTMuteFrame,
    STTUpdateSettingsFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import AIService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error('In order to use AssemblyAI, you need to `pip install "pipecat-ai[assemblyai]"`.')
    raise Exception(f"Missing module: {e}")

#
# Models
#

class Word(BaseModel):
    """Represents a single word in a transcription with timing and confidence."""
    start: int
    end: int
    text: str
    confidence: float
    word_is_final: bool = Field(..., alias="word_is_final")


class BaseMessage(BaseModel):
    """Base class for all AssemblyAI WebSocket messages."""
    type: str


class BeginMessage(BaseMessage):
    """Message sent when a new session begins."""
    type: Literal["Begin"] = "Begin"
    id: str
    expires_at: int


class TurnMessage(BaseMessage):
    """Message containing transcription data for a turn of speech."""
    type: Literal["Turn"] = "Turn"
    turn_order: int
    turn_is_formatted: bool
    end_of_turn: bool
    transcript: str
    end_of_turn_confidence: float
    words: List[Word]


class TerminationMessage(BaseMessage):
    """Message sent when the session is terminated."""
    type: Literal["Termination"] = "Termination"
    audio_duration_seconds: float
    session_duration_seconds: float


AnyMessage = BeginMessage | TurnMessage | TerminationMessage


class AssemblyAIConnectionParams(BaseModel):
    """Configuration parameters for AssemblyAI WebSocket connection."""
    sample_rate: int = 16000
    encoding: Literal["pcm_s16le", "pcm_mulaw"] = "pcm_s16le"
    formatted_finals: bool = True
    word_finalization_max_wait_time: Optional[int] = None
    end_of_turn_confidence_threshold: Optional[float] = None
    min_end_of_turn_silence_when_confident: Optional[int] = None
    max_turn_silence: Optional[int] = None


#
# Base STT Service
#

class STTService(AIService):
    """Base class for speech-to-text services."""
    def __init__(
        self,
        audio_passthrough=True,
        sample_rate: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._audio_passthrough = audio_passthrough
        self._init_sample_rate = sample_rate
        self._sample_rate = 0
        self._settings: Dict[str, Any] = {}
        self._muted: bool = False
        self._user_id: str = ""

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def set_model(self, model: str):
        self.set_model_name(model)

    async def set_language(self, language: Language):
        pass

    @abstractmethod
    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        pass

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._sample_rate = self._init_sample_rate or frame.audio_in_sample_rate

    async def _update_settings(self, settings: Mapping[str, Any]):
        logger.info(f"Updating STT settings: {self._settings}")
        for key, value in settings.items():
            if key in self._settings:
                logger.info(f"Updating STT setting {key} to: [{value}]")
                self._settings[key] = value
                if key == "language":
                    await self.set_language(value)
            elif key == "model":
                self.set_model_name(value)
            else:
                logger.warning(f"Unknown setting for STT service: {key}")

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        if self._muted:
            return

        if hasattr(frame, "user_id"):
            self._user_id = frame.user_id
        else:
            self._user_id = ""

        if not frame.audio:
            logger.warning(
                f"Empty audio frame received for STT service: {self.name} {frame.num_frames}"
            )
            return

        await self.process_generator(self.run_stt(frame.audio))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, AudioRawFrame):
            await self.process_audio_frame(frame, direction)
            if self._audio_passthrough:
                await self.push_frame(frame, direction)
        elif isinstance(frame, STTUpdateSettingsFrame):
            await self._update_settings(frame.settings)
        elif isinstance(frame, STTMuteFrame):
            self._muted = frame.mute
            logger.debug(f"STT service {'muted' if frame.mute else 'unmuted'}")
        else:
            await self.push_frame(frame, direction)


class SegmentedSTTService(STTService):
    """STT service that processes speech in segments using VAD events."""
    def __init__(self, *, sample_rate: Optional[int] = None, **kwargs):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self._content = None
        self._wave = None
        self._audio_buffer = bytearray()
        self._audio_buffer_size_1s = 0
        self._user_speaking = False

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._audio_buffer_size_1s = self.sample_rate * 2

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking(frame)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking(frame)

    async def _handle_user_started_speaking(self, frame: UserStartedSpeakingFrame):
        if frame.emulated:
            return
        self._user_speaking = True

    async def _handle_user_stopped_speaking(self, frame: UserStoppedSpeakingFrame):
        if frame.emulated:
            return
        self._user_speaking = False
        content = io.BytesIO()
        wav = wave.open(content, "wb")
        wav.setsampwidth(2)
        wav.setnchannels(1)
        wav.setframerate(self.sample_rate)
        wav.writeframes(self._audio_buffer)
        wav.close()
        content.seek(0)
        await self.process_generator(self.run_stt(content.read()))
        self._audio_buffer.clear()

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        if hasattr(frame, "user_id"):
            self._user_id = frame.user_id
        else:
            self._user_id = ""
        self._audio_buffer += frame.audio
        if not self._user_speaking and len(self._audio_buffer) > self._audio_buffer_size_1s:
            discarded = len(self._audio_buffer) - self._audio_buffer_size_1s
            self._audio_buffer = self._audio_buffer[discarded:]

#
# AssemblyAI Service
#

class AssemblyAISTTService(STTService):
    """AssemblyAI real-time speech-to-text service."""
    def __init__(
        self,
        *,
        api_key: str,
        language: Language = Language.EN,
        api_endpoint_base_url: str = "wss://streaming.assemblyai.com/v3/ws",
        connection_params: AssemblyAIConnectionParams = AssemblyAIConnectionParams(),
        vad_force_turn_endpoint: bool = True,
        **kwargs,
    ):
        super().__init__(sample_rate=connection_params.sample_rate, **kwargs)
        self._api_key = api_key
        self._language = language
        self._api_endpoint_base_url = api_endpoint_base_url
        self._connection_params = connection_params
        self._vad_force_turn_endpoint = vad_force_turn_endpoint
        self._websocket = None
        self._termination_event = asyncio.Event()
        self._received_termination = False
        self._connected = False
        self._receive_task = None
        self._audio_buffer = bytearray()
        self._chunk_size_ms = 50
        self._chunk_size_bytes = 0

    def can_generate_metrics(self) -> bool:
        return True

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._chunk_size_bytes = int(self._chunk_size_ms * self._sample_rate * 2 / 1000)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        self._audio_buffer.extend(audio)
        while len(self._audio_buffer) >= self._chunk_size_bytes:
            chunk = bytes(self._audio_buffer[: self._chunk_size_bytes])
            self._audio_buffer = self._audio_buffer[self._chunk_size_bytes :]
            await self._websocket.send(chunk)
        yield None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, UserStartedSpeakingFrame):
            await self.start_ttfb_metrics()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            if self._vad_force_turn_endpoint:
                await self._websocket.send(json.dumps({"type": "ForceEndpoint"}))
            await self.start_processing_metrics()

    def _build_ws_url(self) -> str:
        params = {
            k: str(v).lower() if isinstance(v, bool) else v
            for k, v in self._connection_params.model_dump().items()
            if v is not None
        }
        if params:
            query_string = urlencode(params)
            return f"{self._api_endpoint_base_url}?{query_string}"
        return self._api_endpoint_base_url

    async def _connect(self):
        try:
            ws_url = self._build_ws_url()
            headers = {
                "Authorization": self._api_key,
                "User-Agent": f"AssemblyAI/1.0 (integration=Pipecat/{pipecat_version})",
            }
            self._websocket = await websockets.connect(ws_url, extra_headers=headers)
            self._connected = True
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            logger.error(f"Failed to connect to AssemblyAI: {e}")
            self._connected = False
            raise

    async def _disconnect(self):
        if not self._connected or not self._websocket:
            return
        try:
            self._termination_event.clear()
            self._received_termination = False
            if len(self._audio_buffer) > 0:
                await self._websocket.send(bytes(self._audio_buffer))
                self._audio_buffer.clear()
            try:
                await self._websocket.send(json.dumps({"type": "Terminate"}))
                try:
                    await asyncio.wait_for(self._termination_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("Timed out waiting for termination message from server")
            except Exception as e:
                logger.warning(f"Error during termination handshake: {e}")
            if self._receive_task:
                await self.cancel_task(self._receive_task)
            await self._websocket.close()
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
        finally:
            self._websocket = None
            self._connected = False
            self._receive_task = None

    async def _receive_task_handler(self):
        try:
            while self._connected:
                try:
                    message = await asyncio.wait_for(self._websocket.recv(), timeout=1.0)
                    data = json.loads(message)
                    await self._handle_message(data)
                except asyncio.TimeoutError:
                    self.reset_watchdog()
                except websockets.exceptions.ConnectionClosedOK:
                    break
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
                    break
        except Exception as e:
            logger.error(f"Fatal error in receive handler: {e}")

    def _parse_message(self, message: Dict[str, Any]) -> BaseMessage:
        msg_type = message.get("type")
        if msg_type == "Begin":
            return BeginMessage.model_validate(message)
        elif msg_type == "Turn":
            return TurnMessage.model_validate(message)
        elif msg_type == "Termination":
            return TerminationMessage.model_validate(message)
        else:
            raise ValueError(f"Unknown message type: {msg_type}")

    async def _handle_message(self, message: Dict[str, Any]):
        try:
            parsed_message = self._parse_message(message)
            if isinstance(parsed_message, BeginMessage):
                logger.debug(
                    f"Session Begin: {parsed_message.id} (expires at {parsed_message.expires_at})"
                )
            elif isinstance(parsed_message, TurnMessage):
                await self._handle_transcription(parsed_message)
            elif isinstance(parsed_message, TerminationMessage):
                await self._handle_termination(parsed_message)
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_termination(self, message: TerminationMessage):
        self._received_termination = True
        self._termination_event.set()
        logger.info(
            f"Session Terminated: Audio Duration={message.audio_duration_seconds}s, "
            f"Session Duration={message.session_duration_seconds}s"
        )
        await self.push_frame(EndFrame())

    async def _handle_transcription(self, message: TurnMessage):
        if not message.transcript:
            return
        await self.stop_ttfb_metrics()
        if message.end_of_turn and (
            not self._connection_params.formatted_finals or message.turn_is_formatted
        ):
            await self.push_frame(
                TranscriptionFrame(
                    message.transcript,
                    self._user_id,
                    time_now_iso8601(),
                    self._language,
                    message,
                )
            )
            await self.stop_processing_metrics()
        else:
            await self.push_frame(
                InterimTranscriptionFrame(
                    message.transcript,
                    self._user_id,
                    time_now_iso8601(),
                    self._language,
                    message,
                )
            )