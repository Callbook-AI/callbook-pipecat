#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import itertools
import sys
import time
from typing import AsyncGenerator, List

from loguru import logger
from PIL import Image

from pipecat.audio.utils import create_default_resampler
from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    MixerControlFrame,
    OutputAudioRawFrame,
    OutputImageRawFrame,
    SpriteFrame,
    StartFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    SystemFrame,
    TransportMessageFrame,
    TransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.time import nanoseconds_to_seconds

BOT_VAD_STOP_SECS = 0.3


class BaseOutputTransport(FrameProcessor):
    def __init__(self, params: TransportParams, **kwargs):
        super().__init__(**kwargs)

        self._params = params

        # Task to process incoming frames so we don't block upstream elements.
        self._sink_task = None

        # Task to process incoming frames using a clock.
        self._sink_clock_task = None

        # Task to write/send audio and image frames.
        self._camera_out_task = None

        # These are the images that we should send to the camera at our desired
        # framerate.
        self._camera_images = None

        # Output sample rate. It will be initialized on StartFrame.
        self._sample_rate = 0
        self._resampler = create_default_resampler()

        # Chunk size that will be written. It will be computed on StartFrame
        self._audio_chunk_size = 0
        self._audio_buffer = bytearray()

        self._stopped_event = asyncio.Event()

        # Indicates if the bot is currently speaking.
        self._bot_speaking = False

        self._debug = False
        self._handle_audio_buffer = bytearray()
        self._handle_audio_buffer_resampled = bytearray()
        self._sink_queue_source_buffer = bytearray()
        self._sink_queue_exit_buffer = bytearray()
        self._mixed_buffer = bytearray()
        self._writing_frame_buffer = bytearray()
        self._final_audio_buffer = bytearray()
        
    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def start(self, frame: StartFrame):
        self._sample_rate = self._params.audio_out_sample_rate or frame.audio_out_sample_rate

        # We will write 20ms audio at a time. If we receive long audio frames we
        # will chunk them. This will help with interruption handling.
        audio_bytes_10ms = int(self._sample_rate / 100) * self._params.audio_out_channels * 2
        self._audio_chunk_size = audio_bytes_10ms * 2

        # Start audio mixer.
        if self._params.audio_out_mixer:
            await self._params.audio_out_mixer.start(self._sample_rate)
        self._create_camera_task()
        self._create_sink_tasks()

    async def stop(self, frame: EndFrame):
        # Let the sink tasks process the queue until they reach this EndFrame.
        await self._sink_clock_queue.put((sys.maxsize, frame.id, frame))
        await self._sink_queue.put(frame)

        # At this point we have enqueued an EndFrame and we need to wait for
        # that EndFrame to be processed by the sink tasks. We also need to wait
        # for these tasks before cancelling the camera and audio tasks below
        # because they might be still rendering.
        if self._sink_task:
            await self.wait_for_task(self._sink_task)
        if self._sink_clock_task:
            await self.wait_for_task(self._sink_clock_task)

        # We can now cancel the camera task.
        await self._cancel_camera_task()

    async def cancel(self, frame: CancelFrame):
        # Since we are cancelling everything it doesn't matter if we cancel sink
        # tasks first or not.
        await self._cancel_sink_tasks()
        await self._cancel_camera_task()

    async def send_message(self, frame: TransportMessageFrame | TransportMessageUrgentFrame):
        pass

    async def write_frame_to_camera(self, frame: OutputImageRawFrame):
        pass

    async def write_raw_audio_frames(self, frames: bytes):
        pass

    async def send_audio(self, frame: OutputAudioRawFrame):
        await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

    async def send_image(self, frame: OutputImageRawFrame | SpriteFrame):
        await self.queue_frame(frame, FrameDirection.DOWNSTREAM)

    #
    # Frame processor
    #

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        #
        # System frames (like StartInterruptionFrame) are pushed
        # immediately. Other frames require order so they are put in the sink
        # queue.
        #
        if isinstance(frame, StartFrame):
            # Push StartFrame before start(), because we want StartFrame to be
            # processed by every processor before any other frame is processed.
            await self.push_frame(frame, direction)
            await self.start(frame)
        elif isinstance(frame, CancelFrame):
            await self.cancel(frame)
            await self.push_frame(frame, direction)
        elif isinstance(frame, (StartInterruptionFrame, StopInterruptionFrame)):
            await self.push_frame(frame, direction)
            await self._handle_interruptions(frame)
        elif isinstance(frame, TransportMessageUrgentFrame):
            await self.send_message(frame)
        elif isinstance(frame, SystemFrame):
            await self.push_frame(frame, direction)
        # Control frames.
        elif isinstance(frame, EndFrame):
            await self.stop(frame)
            # Keep pushing EndFrame down so all the pipeline stops nicely.
            await self.push_frame(frame, direction)
        elif isinstance(frame, MixerControlFrame) and self._params.audio_out_mixer:
            await self._params.audio_out_mixer.process_frame(frame)
        # Other frames.
        elif isinstance(frame, OutputAudioRawFrame):
            await self._handle_audio(frame)
        elif isinstance(frame, (OutputImageRawFrame, SpriteFrame)):
            await self._handle_image(frame)
        # TODO(aleix): Images and audio should support presentation timestamps.
        elif frame.pts:
            await self._sink_clock_queue.put((frame.pts, frame.id, frame))
        elif direction == FrameDirection.UPSTREAM:
            await self.push_frame(frame, direction)
        else:
            await self._sink_queue.put(frame)

    async def _handle_interruptions(self, frame: Frame):
        if not self.interruptions_allowed:
            return

        if isinstance(frame, StartInterruptionFrame):
            # Cancel sink and camera tasks.
            await self._cancel_sink_tasks()
            await self._cancel_camera_task()
            # Create sink and camera tasks.
            self._create_camera_task()
            self._create_sink_tasks()
            # Let's send a bot stopped speaking if we have to.
            await self._bot_stopped_speaking()

    async def _handle_audio(self, frame: OutputAudioRawFrame):
        if not self._params.audio_out_enabled:
            return
        
        if self._debug:
            self._handle_audio_buffer.extend(frame.audio)

        # We might need to resample if incoming audio doesn't match the
        # transport sample rate.
        resampled = await self._resampler.resample(
            frame.audio, frame.sample_rate, self._sample_rate
        )

    

        cls = type(frame)
        audio_buffer = bytearray(resampled)

        if self._debug:
            self._handle_audio_buffer_resampled.extend(audio_buffer)

        while len(audio_buffer) >= self._audio_chunk_size:
            chunk = cls(
                bytes(audio_buffer[: self._audio_chunk_size]),
                sample_rate=self._sample_rate,
                num_channels=frame.num_channels,
            )
            self._sink_queue_source_buffer.extend(audio_buffer[:self._audio_chunk_size])
            self._sink_queue.put_nowait(chunk)
            audio_buffer = audio_buffer[self._audio_chunk_size:]

    async def _handle_image(self, frame: OutputImageRawFrame | SpriteFrame):
        if not self._params.camera_out_enabled:
            return

        if self._params.camera_out_is_live:
            await self._camera_out_queue.put(frame)
        else:
            await self._sink_queue.put(frame)

    async def _bot_started_speaking(self):
        if not self._bot_speaking:
            logger.debug("Bot started speaking")
            await self.push_frame(BotStartedSpeakingFrame())
            await self.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._bot_speaking = True

    async def _bot_stopped_speaking(self):
        if self._bot_speaking:
            logger.debug("Bot stopped speaking")
            await self.push_frame(BotStoppedSpeakingFrame())
            await self.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            self._bot_speaking = False
            # Clean audio buffer (there could be tiny left overs if not multiple
            # to our output chunk size).
            self._audio_buffer = bytearray()

    #
    # Sink tasks
    #

    def _create_sink_tasks(self):
        if not self._sink_task:
            self._sink_queue = asyncio.Queue()
            self._sink_task = self.create_monitored_task(self._sink_task_handler)
        if not self._sink_clock_task:
            self._sink_clock_queue = asyncio.PriorityQueue()
            self._sink_clock_task = self.create_monitored_task(self._sink_clock_task_handler)

    async def _cancel_sink_tasks(self):
        # Stop sink tasks.
        if self._sink_task:
            await self.cancel_task(self._sink_task)
            self._sink_task = None
        # Stop sink clock tasks.
        if self._sink_clock_task:
            await self.cancel_task(self._sink_clock_task)
            self._sink_clock_task = None

    async def _sink_frame_handler(self, frame: Frame):
        if isinstance(frame, OutputImageRawFrame):
            await self._set_camera_image(frame)
        elif isinstance(frame, SpriteFrame):
            await self._set_camera_images(frame.images)
        elif isinstance(frame, TransportMessageFrame):
            await self.send_message(frame)

    async def _sink_clock_task_handler(self, task_name):
        running = True
        while running:
            try:

                if not self.is_monitored_task_active(task_name): 
                    return

                timestamp, _, frame = await self._sink_clock_queue.get()

                # If we hit an EndFrame, we can finish right away.
                running = not isinstance(frame, EndFrame)

                # If we have a frame we check it's presentation timestamp. If it
                # has already passed we process it, otherwise we wait until it's
                # time to process it.
                logger.debug(f"Pushing frame: {frame}")

                if running:
                    current_time = self.get_clock().get_time()
                    if timestamp > current_time:
                        wait_time = nanoseconds_to_seconds(timestamp - current_time)
                        await asyncio.sleep(wait_time)

                    # Handle frame.
                    await self._sink_frame_handler(frame)

                    # Also, push frame downstream in case anyone else needs it.
                    await self.push_frame(frame)

                self._sink_clock_queue.task_done()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"{self} error processing sink clock queue: {e}")

    def _next_frame(self) -> AsyncGenerator[Frame, None]:
        async def without_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
            while True:
                try:
                    frame = await asyncio.wait_for(self._sink_queue.get(), timeout=vad_stop_secs)
                    yield frame
                except asyncio.TimeoutError:
                    # Notify the bot stopped speaking upstream if necessary.
                    await self._bot_stopped_speaking()

        async def with_mixer(vad_stop_secs: float) -> AsyncGenerator[Frame, None]:
            last_frame_time = 0
            silence = b"\x00" * self._audio_chunk_size
            while True:
                try:
                    frame = self._sink_queue.get_nowait()

                    last_frame_time = time.time()

                    if isinstance(frame, OutputAudioRawFrame):
                        self._sink_queue_exit_buffer.extend(frame.audio)
                        mixed_audio = await self._params.audio_out_mixer.mix(frame.audio)
                        self._mixed_buffer.extend(mixed_audio)
                        
                        new_frame = frame.__class__(
                            audio=mixed_audio,
                            sample_rate=frame.sample_rate,
                            num_channels=frame.num_channels,
                        )
                        yield new_frame
                    else:
                        yield frame

                except asyncio.QueueEmpty:
                    diff_time = time.time() - last_frame_time
                    if diff_time > vad_stop_secs:
                        await self._bot_stopped_speaking()

                    mixed_silence = await self._params.audio_out_mixer.mix(silence)
                    frame = OutputAudioRawFrame(
                        audio=mixed_silence,
                        sample_rate=self._sample_rate,
                        num_channels=self._params.audio_out_channels,
                    )
                    yield frame

        if self._params.audio_out_mixer:
            return with_mixer(BOT_VAD_STOP_SECS)
        else:
            return without_mixer(BOT_VAD_STOP_SECS)

    async def _sink_task_handler(self, task_name):

        async for frame in self._next_frame():

            if not self.is_monitored_task_active(task_name): 
                return 

            # Notify the bot started speaking upstream if necessary and that
            # it's actually speaking.
            if isinstance(frame, TTSAudioRawFrame):
                await self._bot_started_speaking()
                await self.push_frame(BotSpeakingFrame())
                await self.push_frame(BotSpeakingFrame(), FrameDirection.UPSTREAM)

            # No need to push EndFrame, it's pushed from process_frame().
            if isinstance(frame, EndFrame):
                break

            # Handle frame.
            await self._sink_frame_handler(frame)

            # Also, push frame downstream in case anyone else needs it.
            await self.push_frame(frame)

            # Send audio.
            if isinstance(frame, OutputAudioRawFrame):
                self._writing_frame_buffer.extend(frame.audio)
                await self.write_raw_audio_frames(frame.audio)

    #
    # Camera task
    #

    def _create_camera_task(self):
        # Create camera output queue and task if needed.
        if not self._camera_out_task and self._params.camera_out_enabled:
            self._camera_out_queue = asyncio.Queue()
            self._camera_out_task = self.create_task(self._camera_out_task_handler())

    async def _cancel_camera_task(self):
        # Stop camera output task.
        if self._camera_out_task and self._params.camera_out_enabled:
            await self.cancel_task(self._camera_out_task)
            self._camera_out_task = None

    async def _draw_image(self, frame: OutputImageRawFrame):
        desired_size = (self._params.camera_out_width, self._params.camera_out_height)

        if frame.size != desired_size:
            image = Image.frombytes(frame.format, frame.size, frame.image)
            resized_image = image.resize(desired_size)
            logger.warning(f"{frame} does not have the expected size {desired_size}, resizing")
            frame = OutputImageRawFrame(
                resized_image.tobytes(), resized_image.size, resized_image.format
            )

        await self.write_frame_to_camera(frame)

    async def _set_camera_image(self, image: OutputImageRawFrame):
        self._camera_images = itertools.cycle([image])

    async def _set_camera_images(self, images: List[OutputImageRawFrame]):
        self._camera_images = itertools.cycle(images)

    async def _camera_out_task_handler(self):
        self._camera_out_start_time = None
        self._camera_out_frame_index = 0
        self._camera_out_frame_duration = 1 / self._params.camera_out_framerate
        self._camera_out_frame_reset = self._camera_out_frame_duration * 5
        while True:
            if self._params.camera_out_is_live:
                await self._camera_out_is_live_handler()
            elif self._camera_images:
                image = next(self._camera_images)
                await self._draw_image(image)
                await asyncio.sleep(self._camera_out_frame_duration)
            else:
                await asyncio.sleep(self._camera_out_frame_duration)

    async def _camera_out_is_live_handler(self):
        image = await self._camera_out_queue.get()

        # We get the start time as soon as we get the first image.
        if not self._camera_out_start_time:
            self._camera_out_start_time = time.time()
            self._camera_out_frame_index = 0

        # Calculate how much time we need to wait before rendering next image.
        real_elapsed_time = time.time() - self._camera_out_start_time
        real_render_time = self._camera_out_frame_index * self._camera_out_frame_duration
        delay_time = self._camera_out_frame_duration + real_render_time - real_elapsed_time

        if abs(delay_time) > self._camera_out_frame_reset:
            self._camera_out_start_time = time.time()
            self._camera_out_frame_index = 0
        elif delay_time > 0:
            await asyncio.sleep(delay_time)
            self._camera_out_frame_index += 1

        # Render image
        await self._draw_image(image)

        self._camera_out_queue.task_done()
