#
# Fast Text Aggregator for Optimized STT Processing
# Reduces latency for services without interim results like Gladia
#

import asyncio
import time
from typing import List
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EmulateUserStartedSpeakingFrame,
    EmulateUserStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    LLMMessagesAppendAndProcessFrame
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class FastTextAggregator(FrameProcessor):
    """
    Optimized text aggregator for STT services that don't provide interim results.
    Eliminates the 2-second buffer timeout that causes latency in Gladia-only mode.
    """
    
    def __init__(
        self,
        context: OpenAILLMContext,
        role: str = "user",
        fast_response_timeout: float = 0.1,  # Much faster than default 2.0s
        user_speaking_timeout: float = 0.5,   # Reduced from 1.0s 
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._context = context
        self._role = role
        self._aggregation = ""
        
        # Optimized timeouts for fast response
        self._fast_response_timeout = fast_response_timeout
        self._user_speaking_timeout = user_speaking_timeout
        
        # State tracking
        self._user_speaking = False
        self._last_user_speaking_time = 0
        self._aggregation_event = asyncio.Event()
        self._aggregation_task = None
        
        # Force interim mode to bypass buffering logic
        self._force_fast_mode = True
        
        self.reset()

    def reset(self):
        self._aggregation = ""

    @property
    def messages(self) -> List[dict]:
        return self._context.get_messages()

    @property
    def role(self) -> str:
        return self._role

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._start()
        elif isinstance(frame, EndFrame):
            await self.push_frame(frame, direction)
            await self._stop()
        elif isinstance(frame, CancelFrame):
            await self._cancel()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStartedSpeakingFrame):
            await self._handle_user_started_speaking()
            await self.push_frame(frame, direction)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            await self._handle_user_stopped_speaking()
            await self.push_frame(frame, direction)
        elif isinstance(frame, TranscriptionFrame):
            await self._handle_transcription(frame)
        elif isinstance(frame, InterimTranscriptionFrame):
            # In fast mode, we simulate having seen interim results
            # This bypasses the slow buffering logic
            pass  # Just acknowledge we've seen it
        elif isinstance(frame, LLMMessagesAppendFrame):
            self._context.add_messages(frame.messages)
        elif isinstance(frame, LLMMessagesUpdateFrame):
            self._context.set_messages(frame.messages)
        elif isinstance(frame, LLMSetToolsFrame):
            self._context.set_tools(frame.tools)
        else:
            await self.push_frame(frame, direction)

    async def _start(self):
        self._create_aggregation_task()

    async def _stop(self):
        await self._cancel_aggregation_task()

    async def _cancel(self):
        await self._cancel_aggregation_task()

    async def _handle_user_started_speaking(self):
        self._last_user_speaking_time = time.time()
        self._user_speaking = True

    async def _handle_user_stopped_speaking(self):
        self._last_user_speaking_time = time.time()
        self._user_speaking = False
        
        # In fast mode, push aggregation immediately when user stops
        if self._aggregation.strip():
            logger.debug("FastTextAggregator: User stopped speaking, pushing aggregation immediately")
            await self.push_aggregation()
        else:
            # Trigger the fast timeout
            self._aggregation_event.set()

    async def _handle_transcription(self, frame: TranscriptionFrame):
        text = frame.text

        if not text.strip():
            return

        # Add text to aggregation
        self._aggregation += f" {text}" if self._aggregation else text
        
        logger.debug(f"FastTextAggregator: Received transcript: '{text}'")
        
        # If user is speaking, wait for UserStoppedSpeakingFrame
        if self._user_speaking:
            logger.debug("FastTextAggregator: User still speaking, buffering transcript")
            return
        
        # If user not speaking, use fast timeout to push quickly
        self._aggregation_event.set()

    def _create_aggregation_task(self):
        if not self._aggregation_task:
            self._aggregation_task = self.create_task(self._fast_aggregation_handler())

    async def _cancel_aggregation_task(self):
        if self._aggregation_task:
            await self.cancel_task(self._aggregation_task)
            self._aggregation_task = None

    async def _fast_aggregation_handler(self):
        """Optimized aggregation handler with minimal delays."""
        while True:
            try:
                # Use very short timeout for fast response
                await asyncio.wait_for(
                    self._aggregation_event.wait(), 
                    self._fast_response_timeout
                )
                self._aggregation_event.clear()
                
                # Push immediately if we have content
                if self._aggregation.strip():
                    await self.push_aggregation()
                    
            except asyncio.TimeoutError:
                # Check if we should push on timeout
                if self._aggregation.strip() and not self._user_speaking:
                    current_time = time.time()
                    if current_time - self._last_user_speaking_time > self._user_speaking_timeout:
                        await self.push_aggregation()
            except Exception as e:
                logger.exception(f"Error in fast aggregation handler: {e}")
                break

    async def push_aggregation(self):
        if len(self._aggregation) > 0:
            logger.debug(f"FastTextAggregator: Pushing aggregation: '{self._aggregation}'")
            self._context.add_message({"role": self._role, "content": self._aggregation})
            frame = OpenAILLMContextFrame(self._context)
            await self.push_frame(frame)

            # Reset before pushing to avoid race conditions
            aggregated_text = self._aggregation
            self._aggregation = ""

            logger.info(f"FastTextAggregator: Pushed aggregation: '{aggregated_text}'")
            self.reset()


class FastLLMResponseAggregator(FrameProcessor):
    """
    Fast LLM response aggregator that immediately starts TTS generation
    without waiting for complete LLM response when possible.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._aggregation = ""
        self._started = False
        self._word_count = 0
        self._min_words_for_tts = 5  # Start TTS after 5 words

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption()
        elif isinstance(frame, LLMFullResponseStartFrame):
            await self._handle_llm_start()
        elif isinstance(frame, LLMFullResponseEndFrame):
            await self._handle_llm_end()
        elif isinstance(frame, LLMTextFrame):
            await self._handle_llm_text(frame)
        
        await self.push_frame(frame, direction)

    async def _handle_llm_start(self):
        self._started = True
        self._word_count = 0

    async def _handle_llm_end(self):
        self._started = False
        # Push any remaining content
        if self._aggregation:
            await self._push_text_frame()

    async def _handle_llm_text(self, frame: LLMTextFrame):
        if not self._started:
            return
            
        self._aggregation += frame.text
        
        # Count words and push intermediate results for faster TTS
        words = self._aggregation.split()
        if len(words) >= self._min_words_for_tts and len(words) > self._word_count:
            self._word_count = len(words)
            # Push partial text to TTS for immediate processing
            await self._push_text_frame()

    async def _handle_interruption(self):
        self._aggregation = ""
        self._started = False
        self._word_count = 0

    async def _push_text_frame(self):
        if self._aggregation:
            frame = TextFrame(self._aggregation)
            await self.push_frame(frame)
            logger.debug(f"FastLLMResponseAggregator: Pushed partial text: '{self._aggregation[:50]}...'")
            self._aggregation = ""
