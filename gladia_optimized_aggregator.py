#
# Optimized Context Aggregator for Gladia STT Service
# Designed to work efficiently without interim transcription support
#
import asyncio
import time
from typing import List
from loguru import logger

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesAppendFrame,
    LLMMessagesUpdateFrame,
    LLMSetToolsFrame,
    LLMTextFrame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.aggregators.openai_llm_context import (
    OpenAILLMContext,
    OpenAILLMContextFrame,
)
from pipecat.processors.aggregators.llm_response import LLMUserContextAggregator, LLMAssistantContextAggregator
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class GladiaOptimizedUserContextAggregator(LLMUserContextAggregator):
    """
    Optimized user context aggregator specifically designed for Gladia STT service.
    
    Key optimizations:
    1. Immediate processing of final transcripts without waiting for aggregation timeouts
    2. Simulated interim support to trigger early LLM processing
    3. Reduced buffering delays for faster response times
    4. Better coordination with non-interim STT services
    """
    
    def __init__(
        self,
        context: OpenAILLMContext,
        aggregation_timeout: float = 0.05,  # Much faster than default 1.0s
        bot_interruption_timeout: float = 0.1,  # Faster interruption detection
        immediate_mode: bool = True,  # Process transcripts immediately
        **kwargs,
    ):
        super().__init__(
            context=context,
            aggregation_timeout=aggregation_timeout,
            bot_interruption_timeout=bot_interruption_timeout,
            **kwargs,
        )
        
        self._immediate_mode = immediate_mode
        self._last_final_transcript_time = None
        self._pending_transcription = ""
        self._has_seen_interim = False
        
        logger.debug(f"GladiaOptimizedUserContextAggregator initialized with immediate_mode={immediate_mode}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Handle transcription frames with optimized logic for Gladia
        if isinstance(frame, TranscriptionFrame):
            await self._handle_optimized_transcription(frame)
            return
        elif isinstance(frame, InterimTranscriptionFrame):
            await self._handle_optimized_interim(frame)
            # Don't return here, let the parent handle the frame too
        
        # Call parent implementation for other frame types
        await super().process_frame(frame, direction)

    async def _handle_optimized_transcription(self, frame: TranscriptionFrame):
        """Handle final transcription with immediate processing for Gladia."""
        text = frame.text.strip()
        
        if not text:
            return
            
        logger.info(f"🎯 GladiaOptimized: Processing final transcript: '{text}'")
        
        # Store the transcription
        self._aggregation += f" {text}" if self._aggregation else text
        
        # For immediate mode, process right away
        if self._immediate_mode:
            # Simulate the interim experience by immediately pushing context
            logger.info("⚡ GladiaOptimized: Immediate mode - pushing aggregation immediately")
            await self.push_aggregation()
            return
        
        # For non-immediate mode, use very short timeout
        self._last_final_transcript_time = time.time()
        
        # Cancel existing aggregation task and create new one with shorter timeout
        if self._aggregation_task:
            await self.cancel_task(self._aggregation_task)
            
        self._aggregation_task = self.create_task(self._fast_aggregation_handler())

    async def _handle_optimized_interim(self, frame: InterimTranscriptionFrame):
        """Handle interim transcription to prepare for faster final processing."""
        text = frame.text.strip()
        
        if not text:
            return
            
        logger.debug(f"GladiaOptimized: Received interim: '{text}'")
        
        # Mark that we've seen an interim (even if it's simulated)
        self._has_seen_interim = True
        
        # Store interim for potential early processing
        self._pending_transcription = text
        
        # Push frame upstream to trigger early LLM preparation
        await self.push_frame(frame, FrameDirection.UPSTREAM)

    async def _fast_aggregation_handler(self):
        """Fast aggregation handler with minimal delays for Gladia."""
        try:
            # Wait for a very short time to allow any additional transcripts
            await asyncio.sleep(self._aggregation_timeout)
            
            # Push aggregation if we have content
            if self._aggregation.strip():
                logger.debug("GladiaOptimized: Fast timeout - pushing aggregation")
                await self.push_aggregation()
                
        except Exception as e:
            logger.exception(f"Error in fast aggregation handler: {e}")
            
        finally:
            self._aggregation_task = None

    async def push_aggregation(self):
        """Push aggregation with optimized context handling for Gladia."""
        if not self._aggregation.strip():
            return
            
        logger.info(f"🚀 GladiaOptimized: Pushing aggregation to LLM: '{self._aggregation}'")
        
        # Add message to context
        self._context.add_message({"role": self._role, "content": self._aggregation})
        
        # Create and push context frame
        frame = OpenAILLMContextFrame(self._context)
        await self.push_frame(frame)
        
        # Reset state
        self.reset()
        self._pending_transcription = ""
        self._has_seen_interim = False

    def reset(self):
        """Reset aggregator state."""
        super().reset()
        self._pending_transcription = ""
        self._has_seen_interim = False
        self._last_final_transcript_time = None


class GladiaOptimizedAssistantContextAggregator(LLMAssistantContextAggregator):
    """
    Optimized assistant context aggregator for faster TTS generation with Gladia.
    
    Key features:
    1. Streams text to TTS as soon as reasonable chunks are available
    2. Reduces waiting time for complete LLM responses
    3. Better coordination with fast user context aggregator
    """
    
    def __init__(
        self,
        context: OpenAILLMContext,
        min_chars_for_streaming: int = 20,  # Start TTS after 20 characters
        streaming_enabled: bool = True,
        expect_stripped_words: bool = True,  # Add expect_stripped_words parameter
        **kwargs,
    ):
        super().__init__(context=context, expect_stripped_words=expect_stripped_words, **kwargs)
        
        self._min_chars_for_streaming = min_chars_for_streaming
        self._streaming_enabled = streaming_enabled
        self._chars_sent_to_tts = 0
        self._streaming_buffer = ""
        
        logger.debug(f"GladiaOptimizedAssistantContextAggregator initialized with streaming_enabled={streaming_enabled}")

    async def _handle_text(self, frame: TextFrame):
        """Handle LLM text with streaming optimization."""
        # Call parent implementation first
        await super()._handle_text(frame)
        
        # If streaming is enabled, try to send text to TTS early
        if self._streaming_enabled and self._started:
            self._streaming_buffer += frame.text
            
            # Check if we have enough content to start streaming to TTS
            chars_to_send = len(self._streaming_buffer) - self._chars_sent_to_tts
            
            if chars_to_send >= self._min_chars_for_streaming:
                # Find a good breaking point (end of sentence or after comma)
                buffer_to_check = self._streaming_buffer[self._chars_sent_to_tts:]
                
                good_break_idx = -1
                for i, char in enumerate(buffer_to_check):
                    if char in '.!?':
                        good_break_idx = i + 1
                        break
                    elif char in ',;' and i > self._min_chars_for_streaming // 2:
                        good_break_idx = i + 1
                        break
                
                if good_break_idx > 0:
                    text_to_stream = buffer_to_check[:good_break_idx]
                    logger.info(f"🎵 GladiaOptimized: Streaming text to TTS early: '{text_to_stream}'")
                    
                    # Send text frame for TTS
                    await self.push_frame(TextFrame(text_to_stream))
                    self._chars_sent_to_tts += len(text_to_stream)

    async def _handle_llm_start(self, frame: LLMFullResponseStartFrame):
        """Handle LLM start with streaming reset."""
        await super()._handle_llm_start(frame)
        self._streaming_buffer = ""
        self._chars_sent_to_tts = 0

    async def _handle_llm_end(self, frame: LLMFullResponseEndFrame):
        """Handle LLM end with final streaming."""
        # Send any remaining buffered content to TTS
        if self._streaming_enabled and self._streaming_buffer:
            remaining_text = self._streaming_buffer[self._chars_sent_to_tts:]
            if remaining_text.strip():
                logger.debug(f"GladiaOptimized: Sending final text to TTS: '{remaining_text}'")
                await self.push_frame(TextFrame(remaining_text))
        
        # Call parent implementation
        await super()._handle_llm_end(frame)


class GladiaOptimizedAggregatorPair:
    """
    Aggregator pair that mimics the structure expected by the Pipecat pipeline.
    This allows the optimized aggregators to be used as a drop-in replacement
    for the standard OpenAI aggregator pair.
    """
    
    def __init__(self, user_aggregator, assistant_aggregator):
        self._user_aggregator = user_aggregator
        self._assistant_aggregator = assistant_aggregator
    
    def user(self):
        """Return the user context aggregator."""
        return self._user_aggregator
    
    def assistant(self):
        """Return the assistant context aggregator."""
        return self._assistant_aggregator


def create_gladia_optimized_aggregators(context: OpenAILLMContext, **kwargs):
    """
    Factory function to create optimized aggregators for Gladia STT service.
    
    Returns:
        GladiaOptimizedAggregatorPair: An aggregator pair compatible with Pipecat pipeline
    """
    user_aggregator = GladiaOptimizedUserContextAggregator(
        context=context,
        **kwargs.get('user_kwargs', {})
    )
    
    assistant_aggregator = GladiaOptimizedAssistantContextAggregator(
        context=context,
        **kwargs.get('assistant_kwargs', {})
    )
    
    return GladiaOptimizedAggregatorPair(user_aggregator, assistant_aggregator)
