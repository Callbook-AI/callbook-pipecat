import asyncio
from typing import Tuple

from pipecat.processors.frame_processor import FrameProcessor, FrameDirection  # Pipecat core :contentReference[oaicite:0]{index=0}
from pipecat.frames.frames import Frame, StartFrame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.audio.utils import (
    create_default_resampler,
    ulaw_to_pcm,
    pcm_to_ulaw,       # helper funcs inside Pipecat utils :contentReference[oaicite:1]{index=1}
)
from pipecat.transports.base_transport import BaseTransport
from bridge import Bridge                      # the in-memory queues you added

# ---------------------------------------------------------------------------#
#                              Low-level legs                                #
# ---------------------------------------------------------------------------#


class SipProcessor(FrameProcessor):
    """
    One direction of media for a SIP call.
    Implements the minimal FrameProcessor interface expected by PipelineRunner.
    """

    def __init__(
        self,
        queues: Tuple[asyncio.Queue, asyncio.Queue],
        sample_rate: int,
        is_rx: bool,
        parent: "SipTransport",
    ):
        super().__init__()
        self._rx_q, self._tx_q = queues
        self._sample_rate = sample_rate
        self._resampler = create_default_resampler()
        self._is_rx = is_rx
        self._parent = parent

    # ---------- FrameProcessor life-cycle hooks -----------------------------

    async def _start(self, frame: StartFrame):
        # Negotiate final sample-rate once the graph starts
        if frame.audio_in_sample_rate:
            self._sample_rate = frame.audio_in_sample_rate

    # ---------- Rx leg ------------------------------------------------------

    async def start(self, downstream: FrameProcessor):
        """
        Producer loop – runs only on the Rx leg.
        Drains μ-law bytes from SIP and injects PCM upstream.
        """
        if not self._is_rx:
            return
        first = True
        while True:
            ulaw = await self._rx_q.get()
            if first:
                # Fire the "someone joined" hook exactly once
                await self._parent._call_event_handler("on_first_participant_joined")
                first = False

            pcm = await ulaw_to_pcm(ulaw, 8000, self._sample_rate, self._resampler)
            await downstream.queue_frame(
                InputAudioRawFrame(pcm, 1, self._sample_rate), FrameDirection.UPSTREAM
            )

    # ---------- Tx leg ------------------------------------------------------

    async def queue_frame(self, frame: OutputAudioRawFrame, direction: FrameDirection):
        """
        Consumer for frames flowing **out** of Pipecat (bot→caller).
        Runs only on the Tx leg.
        """
        if self._is_rx:
            return
        ulaw = await pcm_to_ulaw(frame.audio, frame.sample_rate, 8000, self._resampler)
        await self._tx_q.put(ulaw)

    # ---------- Helpers -----------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # lifecycle + frame routing
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            # apply negotiated rates
            await self._start(frame)
            # let the next processor start
            await self.push_frame(frame, direction)
            if self._is_rx:
                # kick off SIP→core audio pump
                self.create_task(self.start(self._next), f"{self}::rx_loop")

        # bot→caller audio: send directly out over SIP
        elif not self._is_rx and isinstance(frame, OutputAudioRawFrame):
            await self.queue_frame(frame, direction)


# ---------------------------------------------------------------------------#
#                        High-level Pipecat transport                         #
# ---------------------------------------------------------------------------#


class SipTransport(BaseTransport):
    """
    A leaf transport – `input()` returns the Rx leg, `output()` the Tx leg.
    """

    def __init__(
        self,
        call_id: str,
        sample_rate: int = 16000,
        sip_ip: str = "",
        mixer=None,
        serializer=None,
    ):
        super().__init__()
        # remember for teardown
        self._call_id = call_id
        self._sip_ip = sip_ip
        self._mixer = mixer
        self._serializer = serializer
        queues = Bridge.get(call_id)  # (rx_from_sip, tx_to_sip)
        self._register_event_handler("on_first_participant_joined")
        self._register_event_handler("on_participant_disconnected")

        self._in = SipProcessor(queues, sample_rate, is_rx=True, parent=self)
        self._out = SipProcessor(queues, sample_rate, is_rx=False, parent=self)

    # Pipecat asks for these two accessors:
    def input(self) -> FrameProcessor:
        return self._in

    def output(self) -> FrameProcessor:
        return self._out

    # Called by run_bot() → _end_external_connection()
    async def close(self):
        # notify pipeline
        await self._call_event_handler("on_participant_disconnected")
        # tear down Bridge queues
        Bridge.close(self._call_id)