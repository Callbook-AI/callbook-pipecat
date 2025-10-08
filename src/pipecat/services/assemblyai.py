#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import time
from typing import AsyncGenerator, Optional, Dict
from urllib.parse import urlencode

from loguru import logger

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
    StartInterruptionFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use AssemblyAI, you need to `pip install pipecat-ai[assemblyai]`. Also, install `websockets`."
    )
    raise Exception(f"Missing module: {e}")


class AssemblyAISTTService(STTService):
    """Optimized AssemblyAI STT Service using Universal-1 model with websocket connection.
    
    This implementation follows the Deepgram service pattern for optimized performance,
    using direct websocket communication with AssemblyAI's Universal-1 model which
    supports Spanish and multilingual transcription.
    """
    
    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: Optional[int] = 16000,
        language: str = "multi",  # Spanish by default, supports "multi" for multilingual
        format_turns: bool = True,  # Enable formatted turns for better punctuation
        enable_vad: bool = True,  # Voice Activity Detection
        allow_interruptions: bool = True,  # Allow user to interrupt bot
        word_boost: Optional[list] = None,  # Boost specific words for better recognition
        speech_threshold: float = 0.3,  # Lower = more sensitive to quiet speech (0.0-1.0)
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._websocket = None
        self._receive_task = None
        self._user_speaking = False
        self._language = language
        self._allow_stt_interruptions = allow_interruptions
        
        # Bot speaking state (for interruption handling)
        self._bot_speaking = False
        
        # Optimized settings following Deepgram pattern
        self._settings = {
            "sample_rate": sample_rate or 16000,
            "format_turns": format_turns,
            "language": language,
            "speech_threshold": speech_threshold,  # More sensitive to low volume
        }
        
        # Add word boost if provided (helps with common words like "hola", "bien")
        if word_boost:
            self._settings["word_boost"] = word_boost
        
        # Performance tracking
        self._stt_response_times = []
        self._current_speech_start_time = None
        self._connection_active = False
        
        # Accumulation for sentence handling (like Deepgram)
        self._accum_transcription_frames = []
        self._last_time_accum_transcription = time.time()
        
        # Audio buffering - use 20ms chunks like Deepgram for maximum responsiveness
        # While AssemblyAI docs say 50-1000ms, testing shows it works with smaller chunks
        # and provides much better single-word detection
        self._audio_buffer = bytearray()
        self._min_buffer_size = int((sample_rate or 16000) * 2 * 0.02)  # 20ms of 16-bit audio (640 bytes @ 16kHz)
        self._max_buffer_time = 0.02  # Force send after 20ms (match incoming audio chunks)
        self._last_send_time = time.time()
        
        logger.info(f"AssemblyAI STT Service initialized with language: {language}, sample_rate: {self._settings['sample_rate']}, speech_threshold: {speech_threshold}, allow_interruptions: {allow_interruptions}")

    def can_generate_metrics(self) -> bool:
        return True

    async def set_language(self, language: str):
        """Switch language for transcription."""
        logger.info(f"Switching STT language to: [{language}]")
        self._settings["language"] = language
        self._language = language
        # Need to reconnect with new language
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk for STT transcription.

        Streams audio data to AssemblyAI websocket for real-time transcription.
        
        Uses minimal buffering (20ms chunks) following Deepgram's pattern for
        maximum responsiveness to single words and quiet speech.

        :param audio: Audio data as bytes (PCM 16-bit)
        :yield: None (transcription frames are pushed via callbacks)
        """
        if self._websocket and self._connection_active:
            try:
                # Buffer audio
                self._audio_buffer.extend(audio)
                
                current_time = time.time()
                time_since_last_send = current_time - self._last_send_time
                
                # Send with minimal buffering (20ms chunks) for best single-word detection
                # This matches Deepgram's approach of sending every chunk immediately
                should_send = (
                    len(self._audio_buffer) >= self._min_buffer_size or
                    (len(self._audio_buffer) > 0 and time_since_last_send >= self._max_buffer_time)
                )
                
                if should_send:
                    buffer_size_ms = (len(self._audio_buffer) / (self._settings["sample_rate"] * 2)) * 1000
                    logger.trace(f"{self}: Sending {len(self._audio_buffer)} bytes ({buffer_size_ms:.1f}ms) to AssemblyAI")
                    
                    await self.start_processing_metrics()
                    await self._websocket.send(bytes(self._audio_buffer))
                    await self.stop_processing_metrics()
                    
                    # Clear buffer and update timestamp
                    self._audio_buffer.clear()
                    self._last_send_time = current_time
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"{self}: WebSocket connection closed, attempting to reconnect")
                await self._reconnect()
            except Exception as e:
                logger.error(f"{self}: Error sending audio: {e}")
        yield None

    async def _connect(self):
        """Establish websocket connection to AssemblyAI Universal-1 service.
        
        Uses the v3 API endpoint with optimized settings for Spanish transcription.
        """
        if self._websocket and self._connection_active:
            logger.debug(f"{self}: Already connected to AssemblyAI")
            return

        try:
            # Build connection URL with parameters
            base_url = "wss://streaming.assemblyai.com/v3/ws"
            connection_params = {
                "sample_rate": self._settings["sample_rate"],
                "format_turns": str(self._settings["format_turns"]).lower(),
                "language": self._settings["language"],
                "speech_threshold": self._settings["speech_threshold"],  # More sensitive
            }
            
            # Add word_boost if configured
            if "word_boost" in self._settings:
                connection_params["word_boost"] = json.dumps(self._settings["word_boost"])
            
            url = f"{base_url}?{urlencode(connection_params)}"
            
            logger.info(f"{self}: Connecting to AssemblyAI at {base_url}")
            logger.debug(f"{self}: Connection params: {connection_params}")
            
            # Connect with authentication header
            self._websocket = await websockets.connect(
                url,
                extra_headers={"Authorization": self._api_key},
                ping_interval=5,
                ping_timeout=20
            )
            
            # Small delay to ensure connection is established
            await asyncio.sleep(0.1)
            
            # Receive and log session begins message
            session_begins = await self._websocket.recv()
            logger.info(f"{self}: Connected to AssemblyAI - Session begins: {session_begins}")
            
            self._connection_active = True
            
            # Start receive task to handle incoming transcriptions
            self._receive_task = self.create_task(
                self._receive_task_handler(),
                f"{self}_receive_task"
            )
            
            await self.start_ttfb_metrics()
            
        except Exception as e:
            logger.error(f"{self}: Failed to connect to AssemblyAI: {e}")
            self._connection_active = False
            await self.push_error(ErrorFrame(f"AssemblyAI connection failed: {str(e)}"))

    async def _disconnect(self):
        """Disconnect from AssemblyAI service and clean up resources."""
        self._connection_active = False
        
        # Cancel receive task
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        
        # Close websocket
        if self._websocket:
            try:
                # Flush any remaining buffered audio before closing
                if len(self._audio_buffer) > 0:
                    logger.debug(f"{self}: Flushing {len(self._audio_buffer)} bytes of buffered audio before disconnect")
                    await self._websocket.send(bytes(self._audio_buffer))
                    self._audio_buffer.clear()
                
                # Send terminate message
                await self._websocket.send(json.dumps({"type": "Terminate"}))
                await self._websocket.close()
                logger.info(f"{self}: Disconnected from AssemblyAI")
            except Exception as e:
                logger.debug(f"{self}: Error during disconnect: {e}")
            finally:
                self._websocket = None
                self._audio_buffer.clear()  # Clear buffer on disconnect

    async def _reconnect(self):
        """Attempt to reconnect to AssemblyAI service."""
        logger.info(f"{self}: Attempting to reconnect...")
        await self._disconnect()
        await asyncio.sleep(1)  # Brief delay before reconnecting
        await self._connect()

    async def _receive_task_handler(self):
        """Handle incoming transcription messages from AssemblyAI.
        
        Processes Turn messages with formatted transcripts, similar to how
        Deepgram handles final and interim transcriptions.
        """
        while self._connection_active:
            try:
                if not self._websocket:
                    break
                    
                result_str = await self._websocket.recv()
                data = json.loads(result_str)
                
                await self._on_message(data)
                
            except websockets.exceptions.ConnectionClosed:
                logger.warning(f"{self}: WebSocket connection closed during receive")
                self._connection_active = False
                break
            except json.JSONDecodeError as e:
                logger.error(f"{self}: Failed to parse JSON response: {e}")
            except Exception as e:
                logger.error(f"{self}: Error in receive task: {e}")
                break

    async def _on_message(self, data: Dict):
        """Process incoming transcription message.
        
        Handles Turn messages from AssemblyAI Universal-1 model.
        Follows Deepgram pattern for transcript processing and frame pushing.
        """
        try:
            message_type = data.get('type')
            
            if message_type == 'Turn':
                transcript = data.get('transcript', '')
                
                if not transcript:
                    return
                
                # Check if it's a formatted turn (final) or interim
                is_formatted = data.get('turn_is_formatted', False)
                
                # Check if we should ignore this transcription (bot speaking, etc.)
                if await self._should_ignore_transcription(transcript, is_formatted):
                    return
                
                # Stop TTFB metrics on first transcript
                await self.stop_ttfb_metrics()
                
                # Track response time
                if self._current_speech_start_time is not None:
                    elapsed = time.perf_counter() - self._current_speech_start_time
                    self._stt_response_times.append(elapsed)
                    logger.debug(f"{self}: STT response time: {elapsed:.3f}s")
                
                timestamp = time_now_iso8601()
                
                # Convert language string to Language enum
                language = Language.ES if self._language == "es" else Language.EN
                
                if is_formatted:
                    # This is a final, formatted transcript
                    logger.info(f"{self}: FINAL transcript: '{transcript}'")
                    
                    # Send interruption if bot was speaking
                    if self._bot_speaking and self._allow_stt_interruptions:
                        logger.info(f"{self}: User interrupted bot with: '{transcript}'")
                        await self.push_frame(StartInterruptionFrame(), FrameDirection.UPSTREAM)
                    
                    await self._handle_user_speaking()
                    
                    frame = TranscriptionFrame(
                        transcript, 
                        "", 
                        timestamp, 
                        language
                    )
                    
                    # Accumulate transcription (like Deepgram)
                    self._accum_transcription_frames.append(frame)
                    self._last_time_accum_transcription = time.time()
                    
                    # Send accumulated transcriptions
                    await self._send_accum_transcriptions()
                    
                else:
                    # This is an interim transcript - push immediately for low latency
                    logger.debug(f"{self}: INTERIM transcript: '{transcript}'")
                    
                    await self._handle_user_speaking()
                    
                    # Create and push interim frame immediately (like Deepgram)
                    frame = InterimTranscriptionFrame(
                        transcript,
                        "",
                        timestamp,
                        language
                    )
                    
                    # Always push interim frames for real-time feedback
                    await self.push_frame(frame)
            
            elif message_type == 'Error':
                error_message = data.get('error', 'Unknown error')
                logger.error(f"{self}: AssemblyAI error: {error_message}")
                await self.push_error(ErrorFrame(f"AssemblyAI error: {error_message}"))
                
            elif message_type == 'SessionBegins':
                # Session information, already logged in connect
                pass
                
        except Exception as e:
            logger.exception(f"{self}: Error processing message: {e}")

    async def _handle_user_speaking(self):
        """Handle user started speaking event."""
        if not self._user_speaking:
            self._user_speaking = True
            self._current_speech_start_time = time.perf_counter()
            await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            logger.debug(f"{self}: User started speaking")

    async def _handle_user_silence(self):
        """Handle user stopped speaking event."""
        if self._user_speaking:
            self._user_speaking = False
            self._current_speech_start_time = None
            await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            logger.debug(f"{self}: User stopped speaking")

    async def _send_accum_transcriptions(self):
        """Send accumulated transcriptions and reset buffer.
        
        Following Deepgram's pattern for sentence-based transcript accumulation.
        """
        if not self._accum_transcription_frames:
            return

        logger.debug(f"{self}: Sending accumulated transcriptions")

        for frame in self._accum_transcription_frames:
            await self.push_frame(frame)
        
        self._accum_transcription_frames = []
        await self._handle_user_silence()
        await self.stop_processing_metrics()

    def get_stt_stats(self) -> Dict:
        """Get STT performance statistics."""
        if not self._stt_response_times:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "latest": 0.0
            }
        
        return {
            "count": len(self._stt_response_times),
            "average": round(sum(self._stt_response_times) / len(self._stt_response_times), 3),
            "min": round(min(self._stt_response_times), 3),
            "max": round(max(self._stt_response_times), 3),
            "latest": round(self._stt_response_times[-1], 3) if self._stt_response_times else 0.0,
        }

    def is_connected(self) -> bool:
        """Check if websocket is connected and active."""
        return self._connection_active and self._websocket is not None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for bot speaking state and interruption handling.
        
        Following Deepgram's pattern for handling bot speaking state.
        """
        await super().process_frame(frame, direction)
        
        # Handle bot speaking state for interruption detection
        if isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_speaking()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_silence()

    async def _handle_bot_speaking(self):
        """Handle bot started speaking event."""
        self._bot_speaking = True
        logger.debug(f"{self}: Bot started speaking")

    async def _handle_bot_silence(self):
        """Handle bot stopped speaking event."""
        self._bot_speaking = False
        logger.debug(f"{self}: Bot stopped speaking")

    async def _should_ignore_transcription(self, transcript: str, is_formatted: bool) -> bool:
        """Determine if a transcription should be ignored.
        
        Following Deepgram's pattern for filtering transcriptions during bot speech.
        """
        # If bot is speaking and interruptions are not allowed, block ONLY final transcripts
        # Allow interim transcripts through for real-time feedback
        if self._bot_speaking and not self._allow_stt_interruptions:
            if is_formatted:
                logger.debug(f"{self}: Ignoring FINAL transcription because bot is speaking and interruptions disabled: '{transcript}'")
                return True
            else:
                # Allow interim transcripts through for UI feedback even when bot is speaking
                return False
        
        # If interruptions are allowed, let everything through
        return False
