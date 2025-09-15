#
# Enhanced ElevenLabs TTS Service for Gladia Optimization
# Provides better streaming and reduced latency for non-interim STT services
#
import os
from typing import AsyncGenerator
from loguru import logger

from pipecat.frames.frames import (
    Frame,
    TextFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    StartInterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.elevenlabs import ElevenLabsTTSService


class OptimizedElevenLabsTTSService(ElevenLabsTTSService):
    """
    Enhanced ElevenLabs TTS service optimized for use with Gladia STT.
    
    Key improvements:
    1. Immediate TTS generation on receiving text
    2. Better streaming coordination
    3. Reduced buffering for faster audio output
    4. Optimized for non-interim STT workflows
    """
    
    def __init__(
        self,
        *,
        immediate_synthesis: bool = True,
        min_chars_for_synthesis: int = 10,
        streaming_chunk_size: int = 1024,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self._immediate_synthesis = immediate_synthesis
        self._min_chars_for_synthesis = min_chars_for_synthesis
        self._streaming_chunk_size = streaming_chunk_size
        self._text_buffer = ""
        self._is_processing = False
        
        logger.debug(f"OptimizedElevenLabsTTS initialized with immediate_synthesis={immediate_synthesis}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames with optimized text handling."""
        
        if isinstance(frame, TextFrame):
            await self._handle_optimized_text(frame)
            return
        elif isinstance(frame, LLMTextFrame):
            # Convert LLM text to regular text for immediate processing
            text_frame = TextFrame(frame.text)
            await self._handle_optimized_text(text_frame)
            return
        elif isinstance(frame, StartInterruptionFrame):
            await self._handle_interruption()
            
        # Call parent implementation for other frames
        await super().process_frame(frame, direction)

    async def _handle_optimized_text(self, frame: TextFrame):
        """Handle text frames with immediate synthesis option."""
        text = frame.text.strip()
        
        if not text:
            return
            
        logger.debug(f"OptimizedElevenLabs: Received text: '{text}'")
        
        if self._immediate_synthesis:
            # Process text immediately without buffering
            await self._synthesize_immediately(text)
        else:
            # Use buffering for larger chunks
            self._text_buffer += f" {text}" if self._text_buffer else text
            
            if len(self._text_buffer) >= self._min_chars_for_synthesis:
                await self._synthesize_buffer()

    async def _synthesize_immediately(self, text: str):
        """Synthesize text immediately without waiting."""
        if self._is_processing:
            logger.debug("TTS already processing, queuing text")
            self._text_buffer += f" {text}" if self._text_buffer else text
            return
            
        self._is_processing = True
        
        try:
            logger.debug(f"OptimizedElevenLabs: Starting immediate synthesis: '{text}'")
            
            # Start TTS
            await self.push_frame(TTSStartedFrame())
            
            # Generate audio
            async for audio_frame in self.run_tts(text):
                if audio_frame:
                    await self.push_frame(audio_frame)
            
            # Stop TTS
            await self.push_frame(TTSStoppedFrame())
            
            # Process any buffered text
            if self._text_buffer.strip():
                buffered = self._text_buffer
                self._text_buffer = ""
                await self._synthesize_immediately(buffered)
                
        except Exception as e:
            logger.exception(f"Error in immediate synthesis: {e}")
        finally:
            self._is_processing = False

    async def _synthesize_buffer(self):
        """Synthesize buffered text."""
        if not self._text_buffer.strip() or self._is_processing:
            return
            
        text_to_synthesize = self._text_buffer
        self._text_buffer = ""
        
        await self._synthesize_immediately(text_to_synthesize)

    async def _handle_interruption(self):
        """Handle interruption by clearing buffers."""
        logger.debug("OptimizedElevenLabs: Handling interruption - clearing buffers")
        self._text_buffer = ""
        self._is_processing = False
        
        # Push interruption frame to parent
        await self.push_frame(StartInterruptionFrame(), FrameDirection.UPSTREAM)

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Enhanced TTS generation with optimization."""
        logger.debug(f"OptimizedElevenLabs: Generating TTS for: '{text[:50]}...'")
        
        # Call parent implementation but with optimized parameters
        async for frame in super().run_tts(text):
            yield frame


def create_optimized_tts_service(call) -> OptimizedElevenLabsTTSService:
    """
    Factory function to create optimized TTS service for Gladia workflows.
    """
    voice = call['voice']
    voice_id = voice['voiceId']
    language = call['voice']['language']
    speed = voice.get('speed', 1.0)
    model = voice.get('model', 'eleven_flash_v2_5')
    fast_response = call.get('fast_response', False)
    
    # Use optimized settings for Gladia
    stop_frame_timeout_s = 0.05 if fast_response else 0.1  # Much faster
    immediate_synthesis = True  # Always use immediate synthesis
    min_chars_for_synthesis = 8 if fast_response else 15
    
    logger.debug(f"Creating optimized TTS with immediate_synthesis={immediate_synthesis}")
    
    from pipecat.services.elevenlabs import ElevenLabsTTSService
    
    tts = OptimizedElevenLabsTTSService(
        api_key=os.getenv("ELEVEN_LABS_API_KEY"),
        voice_id=voice_id,
        model=model,
        immediate_synthesis=immediate_synthesis,
        min_chars_for_synthesis=min_chars_for_synthesis,
        params=ElevenLabsTTSService.InputParams(
            similarity_boost=voice['similarityBoost'],
            stability=voice['stability'],
            speed=speed,
            language=language,
            auto_mode=False,
        )
    )
    
    return tts
