#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import inspect
from typing import Awaitable, Callable, Union

from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADActiveFrame,
    VADInactiveFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class UserIdleProcessor(FrameProcessor):
    """Monitors user inactivity and triggers callbacks after timeout periods.

    Starts monitoring only after the first conversation activity (UserStartedSpeaking
    or BotSpeaking).

    Args:
        callback: Function to call when user is idle. Can be either:
            - Basic callback(processor) -> None
            - Retry callback(processor, retry_count) -> bool
              Return True to continue monitoring for idle events,
              Return False to stop the idle monitoring task
        timeout: Seconds to wait before considering user idle
        **kwargs: Additional arguments passed to FrameProcessor

    Example:
        # Retry callback:
        async def handle_idle(processor: "UserIdleProcessor", retry_count: int) -> bool:
            if retry_count < 3:
                await send_reminder("Are you still there?")
                return True
            return False

        # Basic callback:
        async def handle_idle(processor: "UserIdleProcessor") -> None:
            await send_reminder("Are you still there?")

        processor = UserIdleProcessor(
            callback=handle_idle,
            timeout=5.0
        )
    """

    def __init__(
        self,
        *,
        callback: Union[
            Callable[["UserIdleProcessor"], Awaitable[None]],  # Basic
            Callable[["UserIdleProcessor", int], Awaitable[bool]],  # Retry
        ],
        timeout: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._callback = self._wrap_callback(callback)
        self._timeout = timeout
        self._retry_count = 0
        self._interrupted = False
        self._conversation_started = False
        self._idle_task = None
        self._idle_event = asyncio.Event()
        self._bot_is_responding = False  # Track if bot is generating or speaking a response

    def _wrap_callback(
        self,
        callback: Union[
            Callable[["UserIdleProcessor"], Awaitable[None]],
            Callable[["UserIdleProcessor", int], Awaitable[bool]],
        ],
    ) -> Callable[["UserIdleProcessor", int], Awaitable[bool]]:
        """Wraps callback to support both basic and retry signatures.

        Args:
            callback: The callback function to wrap.

        Returns:
            A wrapped callback that returns bool to indicate whether to continue monitoring.
        """
        sig = inspect.signature(callback)
        param_count = len(sig.parameters)

        async def wrapper(processor: "UserIdleProcessor", retry_count: int) -> bool:
            if param_count == 1:
                # Basic callback
                await callback(processor)  # type: ignore
                return True
            else:
                # Retry callback
                return await callback(processor, retry_count)  # type: ignore

        return wrapper

    def _create_idle_task(self) -> None:
        """Creates the idle task if it hasn't been created yet."""
        if not self._idle_task:
            self._idle_task = self.create_task(self._idle_task_handler())

    @property
    def retry_count(self) -> int:
        """Returns the current retry count."""
        return self._retry_count

    async def _stop(self) -> None:
        """Stops and cleans up the idle monitoring task."""
        if self._idle_task:
            await self.cancel_task(self._idle_task)
            self._idle_task = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Processes incoming frames and manages idle monitoring state.

        Args:
            frame: The frame to process
            direction: Direction of the frame flow
        """
        await super().process_frame(frame, direction)

        # Check for end frames before processing
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self.push_frame(frame, direction)  # Push the frame down the pipeline
            if self._idle_task:
                await self._stop()  # Stop the idle task, if it exists
            return

        await self.push_frame(frame, direction)

        # Start monitoring on first conversation activity
        if not self._conversation_started and isinstance(
            frame, (UserStartedSpeakingFrame, BotSpeakingFrame)
        ):
            self._conversation_started = True
            self._create_idle_task()

        # Only process these events if conversation has started
        if self._conversation_started:
            # Track when bot starts generating a response (LLM processing)
            if isinstance(frame, LLMFullResponseStartFrame):
                self._bot_is_responding = True
                self._idle_event.set()
            # Track when bot finishes generating a response
            elif isinstance(frame, LLMFullResponseEndFrame):
                # Keep responding state true - will be cleared when bot stops speaking
                pass
            # We shouldn't call the idle callback if the user or the bot are speaking
            elif isinstance(frame, (UserStartedSpeakingFrame, VADActiveFrame)):
                self._retry_count = 0  # Reset retry count when user speaks
                self._interrupted = True
                self._idle_event.set()
            elif isinstance(frame, (UserStoppedSpeakingFrame, VADInactiveFrame)):
                self._interrupted = False
                self._idle_event.set()
            elif isinstance(frame, BotSpeakingFrame):
                self._bot_is_responding = True
                self._idle_event.set()
            elif isinstance(frame, BotStoppedSpeakingFrame):
                self._bot_is_responding = False
                self._idle_event.set()
            # Ignore idle callback triggering when LLM is processing
            elif isinstance(frame, LLMFullResponseStartFrame):
                self._interrupted = True
                self._idle_event.set()
            elif isinstance(frame, LLMFullResponseEndFrame):
                self._interrupted = False
                self._idle_event.set()

    async def cleanup(self) -> None:
        """Cleans up resources when processor is shutting down."""
        await super().cleanup()
        if self._idle_task:  # Only stop if task exists
            await self._stop()

    async def _idle_task_handler(self) -> None:
        """Monitors for idle timeout and triggers callbacks.

        Runs in a loop until cancelled or callback indicates completion.
        """
        while True:
            try:
                await asyncio.wait_for(self._idle_event.wait(), timeout=self._timeout)
            except asyncio.TimeoutError:
                # Don't trigger idle if user is speaking, bot is speaking, or bot is generating response
                if not self._interrupted and not self._bot_is_responding:
                    self._retry_count += 1
                    should_continue = await self._callback(self, self._retry_count)
                    if not should_continue:
                        await self._stop()
                        break
            finally:
                self._idle_event.clear()
