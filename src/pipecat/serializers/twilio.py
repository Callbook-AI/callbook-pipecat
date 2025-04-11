#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import os
import datetime
from typing import Optional

from pydantic import BaseModel

from pipecat.audio.utils import create_default_resampler, pcm_to_ulaw, ulaw_to_pcm
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    KeypadEntry,
    StartFrame,
    StartInterruptionFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType
from loguru import logger
import struct


class TwilioFrameSerializer(FrameSerializer):
    class InputParams(BaseModel):
        twilio_sample_rate: int = 8000  # Default Twilio rate (8kHz)
        sample_rate: Optional[int] = None  # Pipeline input rate

    def __init__(self, stream_sid: str, call_sid: str = "", params: InputParams = InputParams()):
        self._stream_sid = stream_sid
        self._params = params
        self._call_sid = call_sid

        self._twilio_sample_rate = self._params.twilio_sample_rate
        self._sample_rate = 0  # Pipeline input rate

        self._resampler = create_default_resampler()

        self._log_messages = False
        self._last_timestamp = 0

        self._high_loss_frames = False


    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.TEXT

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    @property
    def log_messages(self) -> bool:
        return self._log_messages

    @log_messages.setter
    def log_messages(self, value: bool) -> None:
        self._log_messages = value
    
    async def setup(self, frame: StartFrame):
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    def _detect_high_loss_frames(self, message: dict):

        if self._high_loss_frames: return
        
        timestamp = int(message["media"]["timestamp"])

        if not self._last_timestamp:
            self._last_timestamp = timestamp

        timestamp_diff = timestamp - self._last_timestamp
        
        if timestamp > 3000 and timestamp_diff > 150:
            self._high_loss_frames = True
        
        self._last_timestamp = timestamp


    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, StartInterruptionFrame):
            answer = {"event": "clear", "streamSid": self._stream_sid}
            return json.dumps(answer)
        elif isinstance(frame, AudioRawFrame):
            data = frame.audio

            # Output: Convert PCM at frame's rate to 8kHz μ-law for Twilio
            serialized_data = await pcm_to_ulaw(
                data, frame.sample_rate, self._twilio_sample_rate, self._resampler
            )
            payload = base64.b64encode(serialized_data).decode("utf-8")
            answer = {
                "event": "media",
                "streamSid": self._stream_sid,
                "media": {"payload": payload},
            }

            return json.dumps(answer)
        elif isinstance(frame, (TransportMessageFrame, TransportMessageUrgentFrame)):
            return json.dumps(frame.message)

    async def deserialize(self, data: str | bytes) -> Frame | None:
        message = json.loads(data)

        if message["event"] == "media":

            self._detect_high_loss_frames(message)

            if self._log_messages:
                logger.debug(message)

            payload_base64 = message["media"]["payload"]
            payload = base64.b64decode(payload_base64)

            # Input: Convert Twilio's 8kHz μ-law to PCM at pipeline input rate
            deserialized_data = await ulaw_to_pcm(
                payload, self._twilio_sample_rate, self._sample_rate, self._resampler
            )
            audio_frame = InputAudioRawFrame(
                audio=deserialized_data, num_channels=1, sample_rate=self._sample_rate
            )
            return audio_frame
        elif message["event"] == "dtmf":
            digit = message.get("dtmf", {}).get("digit")

            try:
                return InputDTMFFrame(KeypadEntry(digit))
            except ValueError as e:
                # Handle case where string doesn't match any enum value
                return None
        else:
            return None



def _simple_vad(audio_data):

    audio_threshold = 100

    total_amplitude = 0
    sample_count = len(audio_data) // 2

    for i in range(0, len(audio_data), 2):
        sample = struct.unpack_from("<h", audio_data, i)[0]
        total_amplitude += abs(sample)
    
    avg_amplitude = total_amplitude / sample_count if sample_count > 0 else 0

    return avg_amplitude > audio_threshold


# Used to test FastAPIWebsocketInputTransport::_silence_audio_stream
class TwilioFrameSerializerVAD(TwilioFrameSerializer):

    async def deserialize(self, data: str | bytes) -> Frame | None:
        
        frame = await super().deserialize(data)

        if not frame: return frame

        is_speaking = _simple_vad(frame.audio)

        logger.debug(f"Is speaking: {is_speaking}")

        if not is_speaking: return None

        return frame