#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Tuple,
)

from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.clocks.base_clock import BaseClock
from pipecat.metrics.metrics import MetricsData
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio import BaseTaskManager
from pipecat.utils.time import nanoseconds_to_str
from pipecat.utils.utils import obj_count, obj_id

if TYPE_CHECKING:
    from pipecat.observers.base_observer import BaseObserver


class KeypadEntry(str, Enum):
    """DTMF entries."""

    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    FIVE = "5"
    SIX = "6"
    SEVEN = "7"
    EIGHT = "8"
    NINE = "9"
    ZERO = "0"
    POUND = "#"
    STAR = "*"


def format_pts(pts: Optional[int]):
    return nanoseconds_to_str(pts) if pts else None


@dataclass
class Frame:
    """Base frame class."""

    id: int = field(init=False)
    name: str = field(init=False)
    pts: Optional[int] = field(init=False)
    metadata: Dict[str, Any] = field(init=False)

    def __post_init__(self):
        self.id: int = obj_id()
        self.name: str = f"{self.__class__.__name__}#{obj_count(self)}"
        self.pts: Optional[int] = None
        self.metadata: Dict[str, Any] = {}

    def __str__(self):
        return self.name


@dataclass
class SystemFrame(Frame):
    """System frames are frames that are not internally queued by any of the
    frame processors and should be processed immediately.

    """

    pass


@dataclass
class DataFrame(Frame):
    """Data frames are frames that will be processed in order and usually
    contain data such as LLM context, text, audio or images.

    """

    pass


@dataclass
class ControlFrame(Frame):
    """Control frames are frames that, similar to data frames, will be processed
    in order and usually contain control information such as frames to update
    settings or to end the pipeline.

    """

    pass


#
# Mixins
#


@dataclass
class AudioRawFrame:
    """A chunk of audio."""

    audio: bytes
    sample_rate: int
    num_channels: int
    num_frames: int = field(default=0, init=False)

    def __post_init__(self):
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))


@dataclass
class ImageRawFrame:
    """A raw image."""

    image: bytes
    size: Tuple[int, int]
    format: Optional[str]


#
# Data frames.
#


@dataclass
class OutputAudioRawFrame(DataFrame, AudioRawFrame):
    """A chunk of audio. Will be played by the output transport if the
    transport's microphone has been enabled.

    """

    def __post_init__(self):
        super().__post_init__()
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, size: {len(self.audio)}, frames: {self.num_frames}, sample_rate: {self.sample_rate}, channels: {self.num_channels})"


@dataclass
class OutputImageRawFrame(DataFrame, ImageRawFrame):
    """An image that will be shown by the transport if the transport's camera is
    enabled.

    """

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, size: {self.size}, format: {self.format})"


@dataclass
class TTSAudioRawFrame(OutputAudioRawFrame):
    """A chunk of output audio generated by a TTS service."""

    pass


@dataclass
class URLImageRawFrame(OutputImageRawFrame):
    """An output image with an associated URL. These images are usually
    generated by third-party services that provide a URL to download the image.

    """

    url: Optional[str]

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, url: {self.url}, size: {self.size}, format: {self.format})"


@dataclass
class SpriteFrame(DataFrame):
    """An animated sprite. Will be shown by the transport if the transport's
    camera is enabled. Will play at the framerate specified in the transport's
    `camera_out_framerate` constructor parameter.

    """

    images: List[OutputImageRawFrame]

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, size: {len(self.images)})"


@dataclass
class TextFrame(DataFrame):
    """A chunk of text. Emitted by LLM services, consumed by TTS services, can
    be used to send text through processors.

    """

    text: str

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, text: [{self.text}])"


@dataclass
class LLMTextFrame(TextFrame):
    """A text frame generated by LLM services."""

    pass


@dataclass
class TTSTextFrame(TextFrame):
    """A text frame generated by TTS services."""

    pass


@dataclass
class TranscriptionFrame(TextFrame):
    """A text frame with transcription-specific data. Will be placed in the
    transport's receive queue when a participant speaks.

    """

    user_id: str
    timestamp: str
    language: Optional[Language] = None

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: [{self.text}], language: {self.language}, timestamp: {self.timestamp})"


@dataclass
class InterimTranscriptionFrame(TextFrame):
    """A text frame with interim transcription-specific data. Will be placed in
    the transport's receive queue when a participant speaks.
    """

    text: str
    user_id: str
    timestamp: str
    language: Optional[Language] = None

    def __str__(self):
        return f"{self.name}(user: {self.user_id}, text: [{self.text}], language: {self.language}, timestamp: {self.timestamp})"


@dataclass
class VoicemailFrame(Frame):
    text: str

    def __str__(self):
        return F"{self.name}(text: [{self.text}])"

@dataclass
class OpenAILLMContextAssistantTimestampFrame(DataFrame):
    """Timestamp information for assistant message in LLM context."""

    timestamp: str


@dataclass
class TranscriptionMessage:
    """A message in a conversation transcript containing the role and content.

    Messages are in standard format with roles normalized to user/assistant.
    """

    role: Literal["user", "assistant"]
    content: str
    timestamp: Optional[str] = None


@dataclass
class TranscriptionUpdateFrame(DataFrame):
    """A frame containing new messages added to the conversation transcript.

    This frame is emitted when new messages are added to the conversation history,
    containing only the newly added messages rather than the full transcript.
    Messages have normalized roles (user/assistant) regardless of the LLM service used.
    Messages are always in the OpenAI standard message format, which supports both:

    Simple format:
    [
        {
            "role": "user",
            "content": "Hi, how are you?"
        },
        {
            "role": "assistant",
            "content": "Great! And you?"
        }
    ]

    Content list format:
    [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hi, how are you?"}]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Great! And you?"}]
        }
    ]

    OpenAI supports both formats. Anthropic and Google messages are converted to the
    content list format.
    """

    messages: List[TranscriptionMessage]

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, messages: {len(self.messages)})"


@dataclass
class LLMMessagesFrame(DataFrame):
    """A frame containing a list of LLM messages. Used to signal that an LLM
    service should run a chat completion and emit an LLMFullResponseStartFrame,
    TextFrames and an LLMFullResponseEndFrame. Note that the `messages`
    property in this class is mutable, and will be be updated by various
    aggregators.

    """

    messages: List[dict]


@dataclass
class LLMMessagesAppendFrame(DataFrame):
    """A frame containing a list of LLM messages that need to be added to the
    current context.

    """

    messages: List[dict]


@dataclass
class LLMMessagesUpdateFrame(DataFrame):
    """A frame containing a list of new LLM messages. These messages will
    replace the current context LLM messages and should generate a new
    LLMMessagesFrame.

    """

    messages: List[dict]


@dataclass
class LLMSetToolsFrame(DataFrame):
    """A frame containing a list of tools for an LLM to use for function calling.
    The specific format depends on the LLM being used, but it should typically
    contain JSON Schema objects.
    """

    tools: List[dict]


@dataclass
class LLMEnablePromptCachingFrame(DataFrame):
    """A frame to enable/disable prompt caching in certain LLMs."""

    enable: bool


@dataclass
class FunctionCallResultProperties:
    """Properties for a function call result frame."""

    run_llm: Optional[bool] = None
    on_context_updated: Optional[Callable[[], Awaitable[None]]] = None


@dataclass
class FunctionCallResultFrame(DataFrame):
    """A frame containing the result of an LLM function (tool) call."""

    function_name: str
    tool_call_id: str
    arguments: str
    result: Any
    properties: Optional[FunctionCallResultProperties] = None


@dataclass
class TTSSpeakFrame(DataFrame):
    """A frame that contains a text that should be spoken by the TTS in the
    pipeline (if any).

    """

    text: str


@dataclass
class TransportMessageFrame(DataFrame):
    message: Any

    def __str__(self):
        return f"{self.name}(message: {self.message})"


@dataclass
class DTMFFrame(DataFrame):
    """A DTMF button frame"""

    button: KeypadEntry


@dataclass
class InputDTMFFrame(DTMFFrame):
    """A DTMF button input"""

    pass


@dataclass
class OutputDTMFFrame(DTMFFrame):
    """A DTMF button output"""

    pass


#
# System frames
#


@dataclass
class StartFrame(SystemFrame):
    """This is the first frame that should be pushed down a pipeline."""

    clock: BaseClock
    task_manager: BaseTaskManager
    audio_in_sample_rate: int = 16000
    audio_out_sample_rate: int = 24000
    allow_interruptions: bool = False
    enable_metrics: bool = False
    enable_usage_metrics: bool = False
    observer: Optional["BaseObserver"] = None
    report_only_initial_ttfb: bool = False


@dataclass
class CancelFrame(SystemFrame):
    """Indicates that a pipeline needs to stop right away."""

    pass


@dataclass
class ErrorFrame(SystemFrame):
    """This is used notify upstream that an error has occurred downstream the
    pipeline. A fatal error indicates the error is unrecoverable and that the
    bot should exit.

    """

    error: str
    fatal: bool = False

    def __str__(self):
        return f"{self.name}(error: {self.error}, fatal: {self.fatal})"


@dataclass
class FatalErrorFrame(ErrorFrame):
    """This is used notify upstream that an unrecoverable error has occurred and
    that the bot should exit.

    """

    fatal: bool = field(default=True, init=False)


@dataclass
class HeartbeatFrame(SystemFrame):
    """This frame is used by the pipeline task as a mechanism to know if the
    pipeline is running properly.

    """

    timestamp: int


@dataclass
class EndTaskFrame(SystemFrame):
    """This is used to notify the pipeline task that the pipeline should be
    closed nicely (flushing all the queued frames) by pushing an EndFrame
    downstream.

    """

    pass


@dataclass
class CancelTaskFrame(SystemFrame):
    """This is used to notify the pipeline task that the pipeline should be
    stopped immediately by pushing a CancelFrame downstream.

    """

    pass


@dataclass
class StopTaskFrame(SystemFrame):
    """This is used to notify the pipeline task that it should be stopped as
    soon as possible (flushing all the queued frames) but that the pipeline
    processors should be kept in a running state.

    """

    pass


@dataclass
class StartInterruptionFrame(SystemFrame):
    """Emitted by VAD to indicate that a user has started speaking (i.e. is
    interruption). This is similar to UserStartedSpeakingFrame except that it
    should be pushed concurrently with other frames (so the order is not
    guaranteed).

    """

    pass


@dataclass
class StopInterruptionFrame(SystemFrame):
    """Emitted by VAD to indicate that a user has stopped speaking (i.e. no more
    interruptions). This is similar to UserStoppedSpeakingFrame except that it
    should be pushed concurrently with other frames (so the order is not
    guaranteed).

    """

    pass


@dataclass
class UserStartedSpeakingFrame(SystemFrame):
    """Emitted by VAD to indicate that a user has started speaking. This can be
    used for interruptions or other times when detecting that someone is
    speaking is more important than knowing what they're saying (as you will
    with a TranscriptionFrame)

    """

    pass


@dataclass
class UserStoppedSpeakingFrame(SystemFrame):
    """Emitted by the VAD to indicate that a user stopped speaking."""

    pass


@dataclass
class VADActiveFrame(SystemFrame):
    """Send when the VAD detects the user is speaking"""
    pass


@dataclass
class VADInactiveFrame(SystemFrame):
    """Send when the VAD detects the user stopped speaking"""
    pass


@dataclass
class EmulateUserStartedSpeakingFrame(SystemFrame):
    """Emitted by internal processors upstream to emulate VAD behavior when a
    user starts speaking.
    """

    pass


@dataclass
class EmulateUserStoppedSpeakingFrame(SystemFrame):
    """Emitted by internal processors upstream to emulate VAD behavior when a
    user stops speaking.
    """

    pass


@dataclass
class BotInterruptionFrame(SystemFrame):
    """Emitted by when the bot should be interrupted. This will mainly cause the
    same actions as if the user interrupted except that the
    UserStartedSpeakingFrame and UserStoppedSpeakingFrame won't be generated.

    """

    pass


@dataclass
class BotStartedSpeakingFrame(SystemFrame):
    """Emitted upstream by transport outputs to indicate the bot started speaking."""

    pass


@dataclass
class BotStoppedSpeakingFrame(SystemFrame):
    """Emitted upstream by transport outputs to indicate the bot stopped speaking."""

    pass


@dataclass
class BotSpeakingFrame(SystemFrame):
    """Emitted upstream by transport outputs while the bot is still
    speaking. This can be used, for example, to detect when a user is idle. That
    is, while the bot is speaking we don't want to trigger any user idle timeout
    since the user might be listening.

    """

    pass


@dataclass
class MetricsFrame(SystemFrame):
    """Emitted by processor that can compute metrics like latencies."""

    data: List[MetricsData]


@dataclass
class FunctionCallInProgressFrame(SystemFrame):
    """A frame signaling that a function call is in progress."""

    function_name: str
    tool_call_id: str
    arguments: str


@dataclass
class STTMuteFrame(SystemFrame):
    """System frame to mute/unmute the STT service."""

    mute: bool

@dataclass
class STTRestartFrame(SystemFrame):
    """System frame to restart the STT service."""
    pass



@dataclass
class TransportMessageUrgentFrame(SystemFrame):
    message: Any

    def __str__(self):
        return f"{self.name}(message: {self.message})"


@dataclass
class UserImageRequestFrame(SystemFrame):
    """A frame user to request an image from the given user."""

    user_id: str
    context: Optional[Any] = None

    def __str__(self):
        return f"{self.name}, user: {self.user_id}"


@dataclass
class InputAudioRawFrame(SystemFrame, AudioRawFrame):
    """A chunk of audio usually coming from an input transport."""

    def __post_init__(self):
        super().__post_init__()
        self.num_frames = int(len(self.audio) / (self.num_channels * 2))

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, size: {len(self.audio)}, frames: {self.num_frames}, sample_rate: {self.sample_rate}, channels: {self.num_channels})"


@dataclass
class InputImageRawFrame(SystemFrame, ImageRawFrame):
    """An image usually coming from an input transport."""

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, size: {self.size}, format: {self.format})"


@dataclass
class UserImageRawFrame(InputImageRawFrame):
    """An image associated to a user."""

    user_id: str

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, user: {self.user_id}, size: {self.size}, format: {self.format})"


@dataclass
class VisionImageRawFrame(InputImageRawFrame):
    """An image with an associated text to ask for a description of it."""

    text: Optional[str]

    def __str__(self):
        pts = format_pts(self.pts)
        return f"{self.name}(pts: {pts}, text: [{self.text}], size: {self.size}, format: {self.format})"


#
# Control frames
#


@dataclass
class EndFrame(ControlFrame):
    """Indicates that a pipeline has ended and frame processors and pipelines
    should be shut down. If the transport receives this frame, it will stop
    sending frames to its output channel(s) and close all its threads. Note,
    that this is a control frame, which means it will received in the order it
    was sent (unline system frames).

    """

    pass


@dataclass
class StopFrame(ControlFrame):
    """Indicates that a pipeline should be stopped but that the pipeline
    processors should be kept in a running state. This is normally queued from
    the pipeline task.

    """

    pass


@dataclass
class LLMFullResponseStartFrame(ControlFrame):
    """Used to indicate the beginning of an LLM response. Following by one or
    more TextFrame and a final LLMFullResponseEndFrame.
    """

    pass


@dataclass
class LLMFullResponseEndFrame(ControlFrame):
    """Indicates the end of an LLM response."""

    pass


@dataclass
class TTSStartedFrame(ControlFrame):
    """Used to indicate the beginning of a TTS response. Following
    TTSAudioRawFrames are part of the TTS response until an
    TTSStoppedFrame. These frames can be used for aggregating audio frames in a
    transport to optimize the size of frames sent to the session, without
    needing to control this in the TTS service.

    """

    pass


@dataclass
class TTSStoppedFrame(ControlFrame):
    """Indicates the end of a TTS response."""

    pass


@dataclass
class ServiceUpdateSettingsFrame(ControlFrame):
    """A control frame containing a request to update service settings."""

    settings: Mapping[str, Any]


@dataclass
class LLMUpdateSettingsFrame(ServiceUpdateSettingsFrame):
    pass


@dataclass
class TTSUpdateSettingsFrame(ServiceUpdateSettingsFrame):
    pass


@dataclass
class STTUpdateSettingsFrame(ServiceUpdateSettingsFrame):
    pass


@dataclass
class VADParamsUpdateFrame(ControlFrame):
    """A control frame containing a request to update VAD params. Intended
    to be pushed upstream from RTVI processor.
    """

    params: VADParams


@dataclass
class FilterControlFrame(ControlFrame):
    """Base control frame for other audio filter frames."""

    pass


@dataclass
class FilterUpdateSettingsFrame(FilterControlFrame):
    """Control frame to update filter settings."""

    settings: Mapping[str, Any]


@dataclass
class FilterEnableFrame(FilterControlFrame):
    """Control frame to enable or disable the filter at runtime."""

    enable: bool


@dataclass
class MixerControlFrame(ControlFrame):
    """Base control frame for other audio mixer frames."""

    pass


@dataclass
class MixerUpdateSettingsFrame(MixerControlFrame):
    """Control frame to update mixer settings."""

    settings: Mapping[str, Any]


@dataclass
class MixerEnableFrame(MixerControlFrame):
    """Control frame to enable or disable the mixer at runtime."""

    enable: bool
