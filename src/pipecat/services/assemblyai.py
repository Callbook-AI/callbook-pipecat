#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import time
from typing import AsyncGenerator, Optional, Dict, List
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
    VADActiveFrame,
    VADInactiveFrame,
    VoicemailFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import STTService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.string import is_equivalent_basic
from pipecat.utils.text import voicemail

try:
    import websockets
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use AssemblyAI, you need to `pip install pipecat-ai[assemblyai]`. Also, install `websockets`."
    )
    raise Exception(f"Missing module: {e}")

# Constants for optimized behavior (matching Deepgram patterns)
DEFAULT_ON_NO_PUNCTUATION_SECONDS = 2
IGNORE_REPEATED_MSG_AT_START_SECONDS = 4
VOICEMAIL_DETECTION_SECONDS = 10
FALSE_INTERIM_SECONDS = 1.3


class AssemblyAISTTService(STTService):
    """Optimized AssemblyAI STT Service using Universal-1 model with websocket connection.
    
    This implementation follows the Deepgram service pattern for optimized performance,
    using direct websocket communication with AssemblyAI's Universal-1 model which
    supports Spanish and multilingual transcription.
    
    Features:
    - Fast response mode for minimal latency
    - Voicemail detection
    - Advanced transcript filtering and accumulation
    - Bot speaking state tracking
    - Comprehensive performance metrics
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
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS,
        detect_voicemail: bool = True,  # Enable voicemail detection
        fast_response: bool = False,  # Enable fast response mode for minimal latency
        false_interim_seconds: float = FALSE_INTERIM_SECONDS,  # Time threshold for false interim detection
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._websocket = None
        self._receive_task = None
        self._async_handler_task = None
        self._user_speaking = False
        self._language = language
        self._allow_stt_interruptions = allow_interruptions
        self.detect_voicemail = detect_voicemail
        self._fast_response = fast_response
        self._on_no_punctuation_seconds = on_no_punctuation_seconds
        self._false_interim_seconds = false_interim_seconds  # Configurable threshold for user idle detection
        
        # Bot speaking state (for interruption handling)
        self._bot_speaking = True  # Start as True like Deepgram
        self._bot_has_ever_spoken = False  # Track if bot has spoken at least once
        self._bot_started_speaking_time = None  # Track when bot started speaking
        
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
        
        # Performance tracking (enhanced like Deepgram)
        self._stt_response_times = []
        self._current_speech_start_time = None
        self._last_audio_chunk_time = None
        self._audio_chunk_count = 0
        self._connection_active = False
        
        # Accumulation for sentence handling (like Deepgram)
        self._accum_transcription_frames = []
        self._last_time_accum_transcription = time.time()
        self._last_interim_time = None
        
        # First message tracking (for voicemail and repeated message detection)
        self._first_message = None
        self._first_message_time = None
        self._was_first_transcript_receipt = False
        
        # Timing tracking
        self.start_time = time.time()
        self._last_time_transcription = time.time()
        
        # VAD state tracking (like Deepgram)
        self._vad_active = False
        self._vad_inactive_time = None  # Track when VAD became inactive for grace period
        
        # Audio buffering - AssemblyAI strictly requires 50-1000ms chunks (enforced with error 3005)
        # We use 50ms minimum for best responsiveness while meeting their requirements
        self._audio_buffer = bytearray()
        self._min_buffer_size = int((sample_rate or 16000) * 2 * 0.05)  # 50ms of 16-bit audio (1600 bytes @ 16kHz)
        self._max_buffer_time = 0.05  # Force send after 50ms minimum
        self._last_send_time = time.time()
        
        logger.info(f"AssemblyAI STT Service initialized:")
        logger.info(f"  Language: {language}, Sample rate: {self._settings['sample_rate']}")
        logger.info(f"  Speech threshold: {speech_threshold}, Allow interruptions: {allow_interruptions}")
        logger.info(f"  Fast response: {fast_response}, Detect voicemail: {detect_voicemail}")
        logger.info(f"  User idle threshold: {false_interim_seconds}s, No punctuation timeout: {on_no_punctuation_seconds}s")

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
        
        # Start async handler for fast response and timeout management
        if not self._async_handler_task:
            self._async_handler_task = self.create_monitored_task(self._async_handler)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk for STT transcription.

        Streams audio data to AssemblyAI websocket for real-time transcription.
        
        Uses 50ms buffering (AssemblyAI's strict minimum requirement).
        
        AssemblyAI enforces 50-1000ms chunks with error 3005 if violated.

        :param audio: Audio data as bytes (PCM 16-bit)
        :yield: None (transcription frames are pushed via callbacks)
        """
        if self._websocket and self._connection_active:
            try:
                # Enhanced timing tracking (like Deepgram)
                current_time = time.perf_counter()
                self._last_audio_chunk_time = current_time
                self._audio_chunk_count += 1
                
                # Start timing when we receive first audio after speech detection
                if self._current_speech_start_time is None:
                    self._current_speech_start_time = current_time
                    logger.debug(f"🎤 {self}: Starting speech detection timer at chunk #{self._audio_chunk_count}")
                
                # Buffer audio
                self._audio_buffer.extend(audio)
                
                current_time_wall = time.time()
                time_since_last_send = current_time_wall - self._last_send_time
                
                # CRITICAL: AssemblyAI strictly requires 50-1000ms audio chunks
                # We must NEVER send buffers smaller than 50ms, even after timeout
                # Otherwise we get error 3005 (Input Duration Violation)
                should_send = len(self._audio_buffer) >= self._min_buffer_size
                
                if should_send:
                    buffer_size_ms = (len(self._audio_buffer) / (self._settings["sample_rate"] * 2)) * 1000
                    logger.trace(f"⚡ {self}: Sending {len(self._audio_buffer)} bytes ({buffer_size_ms:.1f}ms) to AssemblyAI")
                    
                    await self.start_processing_metrics()
                    await self._websocket.send(bytes(self._audio_buffer))
                    await self.stop_processing_metrics()
                    
                    # Clear buffer and update timestamp
                    self._audio_buffer.clear()
                    self._last_send_time = current_time_wall
                elif len(self._audio_buffer) > 0 and time_since_last_send >= self._max_buffer_time:
                    # Buffer too small but timeout elapsed - keep accumulating
                    buffer_size_ms = (len(self._audio_buffer) / (self._settings["sample_rate"] * 2)) * 1000
                    logger.trace(f"⏳ {self}: Holding {len(self._audio_buffer)} bytes ({buffer_size_ms:.1f}ms) - need {self._min_buffer_size} bytes minimum")
                    
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
        
        # Cancel async handler task
        if self._async_handler_task:
            await self.cancel_task(self._async_handler_task)
            self._async_handler_task = None
        
        # Cancel receive task
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None
        
        # Close websocket
        if self._websocket:
            try:
                # Only flush if buffer meets minimum size requirement (avoid error 3005)
                buffer_size = len(self._audio_buffer)
                if buffer_size >= self._min_buffer_size:
                    buffer_duration_ms = (buffer_size / (self._settings["sample_rate"] * 2)) * 1000
                    logger.debug(f"{self}: Flushing {buffer_size} bytes ({buffer_duration_ms:.1f}ms) before disconnect")
                    await self._websocket.send(bytes(self._audio_buffer))
                    self._audio_buffer.clear()
                elif buffer_size > 0:
                    # Too small to send - would trigger error 3005
                    buffer_duration_ms = (buffer_size / (self._settings["sample_rate"] * 2)) * 1000
                    logger.debug(f"{self}: Discarding {buffer_size} bytes ({buffer_duration_ms:.1f}ms) - too small for AssemblyAI requirements")
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
                
                # Stop TTFB metrics on first transcript
                await self.stop_ttfb_metrics()
                
                # Enhanced logging with full context for debugging
                transcript_type = "FINAL" if is_formatted else "INTERIM"
                word_count = len(transcript.split())
                current_time = time.time()
                
                logger.info(f"📝 AssemblyAI {transcript_type}: '{transcript}'")
                logger.debug(f"   ├─ Words: {word_count} | Bot speaking: {self._bot_speaking} | User speaking: {self._user_speaking}")
                logger.debug(f"   ├─ VAD active: {self._vad_active} | Allow interruptions: {self._allow_stt_interruptions}")
                
                # Log timing information for debugging
                if self._last_interim_time:
                    time_since_interim = current_time - self._last_interim_time
                    logger.debug(f"   ├─ Time since last interim: {time_since_interim:.3f}s")
                
                if self._last_time_transcription:
                    time_since_last_transcript = current_time - self._last_time_transcription
                    logger.debug(f"   ├─ Time since last transcript: {time_since_last_transcript:.3f}s")
                
                if self._bot_started_speaking_time:
                    time_since_bot_started = current_time - self._bot_started_speaking_time
                    logger.debug(f"   └─ Time since bot started speaking: {time_since_bot_started:.3f}s")
                else:
                    logger.debug(f"   └─ Bot has not started speaking yet")
                
                # Check voicemail detection (only for final transcripts)
                if is_formatted and await self._detect_and_handle_voicemail(transcript):
                    logger.info(f"{self}: Voicemail detected and handled")
                    return
                
                # Check if we should ignore this transcription (bot speaking, repeated messages, etc.)
                if await self._should_ignore_transcription(transcript, is_formatted):
                    logger.debug(f"   ⚠️  Transcript was filtered out (see reason above)")
                    return
                
                logger.debug(f"   ✅ Transcript accepted and will be processed")
                
                timestamp = time_now_iso8601()
                language = self._get_language_enum()
                
                if is_formatted:
                    # This is a final, formatted transcript
                    
                    # Enhanced response time measurement
                    if self._current_speech_start_time is not None:
                        elapsed = time.perf_counter() - self._current_speech_start_time
                        elapsed_formatted = round(elapsed, 3)
                        self._stt_response_times.append(elapsed_formatted)
                        
                        logger.info(f"📊 ⚡ AssemblyAI: ⏱️ STT Response Time: {elapsed_formatted}s")
                        logger.info(f"   📝 Final Transcript: '{transcript}'")
                        logger.info(f"   📦 Audio chunks processed: {self._audio_chunk_count}")
                        
                        self._current_speech_start_time = None
                        self._audio_chunk_count = 0
                        logger.debug(f"🔄 {self}: Reset speech timing counters")
                    
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
                    
                    # Track first message
                    self._handle_first_message(frame.text)
                    
                    # Accumulate transcription (like Deepgram)
                    self._append_accum_transcription(frame)
                    self._was_first_transcript_receipt = True
                    self._last_time_transcription = time.time()
                    
                    # Send accumulated transcriptions based on mode
                    if self._fast_response:
                        await self._fast_response_send_accum_transcriptions()
                    else:
                        if not self._is_accum_transcription(frame.text):
                            logger.debug("Sending final transcription frame (end of sentence)")
                            await self._send_accum_transcriptions()
                    
                else:
                    # This is an interim transcript - push immediately for low latency
                    logger.debug(f"{self}: Pushing INTERIM transcript frame")
                    
                    # Update timing trackers
                    self._last_interim_time = time.time()
                    
                    # Mark user as speaking
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
                    logger.debug(f"   ✅ Interim frame pushed to pipeline")
            
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
            logger.info(f"👤 {self}: User started speaking")
            logger.debug(f"   └─ VAD active: {self._vad_active}, Bot speaking: {self._bot_speaking}")

    async def _handle_user_silence(self):
        """Handle user stopped speaking event."""
        if self._user_speaking:
            self._user_speaking = False
            self._current_speech_start_time = None
            await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            logger.info(f"👤 {self}: User stopped speaking")
            logger.debug(f"   └─ VAD active: {self._vad_active}, Bot speaking: {self._bot_speaking}")

    async def _send_accum_transcriptions(self):
        """Send accumulated transcriptions and reset buffer.
        
        Following Deepgram's pattern for sentence-based transcript accumulation.
        """
        if not len(self._accum_transcription_frames):
            return

        logger.debug(f"{self}: Sending {len(self._accum_transcription_frames)} accumulated transcription(s)")

        await self._handle_user_speaking()

        for frame in self._accum_transcription_frames:
            await self.push_frame(frame)
        
        self._accum_transcription_frames = []
        await self._handle_user_silence()
        await self.stop_processing_metrics()

    def get_stt_stats(self) -> Dict:
        """Get comprehensive STT performance statistics."""
        if not self._stt_response_times:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "latest": 0.0,
                "all_times": []
            }
        
        return {
            "count": len(self._stt_response_times),
            "average": round(sum(self._stt_response_times) / len(self._stt_response_times), 3),
            "min": round(min(self._stt_response_times), 3),
            "max": round(max(self._stt_response_times), 3),
            "latest": round(self._stt_response_times[-1], 3) if self._stt_response_times else 0.0,
            "all_times": [round(t, 3) for t in self._stt_response_times]
        }

    def is_connected(self) -> bool:
        """Check if websocket is connected and active."""
        return self._connection_active and self._websocket is not None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for bot speaking state and interruption handling.
        
        Following Deepgram's pattern for handling bot speaking state and VAD tracking.
        """
        await super().process_frame(frame, direction)
        
        # Handle bot speaking state for interruption detection
        if isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_speaking()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_silence()
        
        # Handle VAD state for false interim detection (like Deepgram)
        elif isinstance(frame, VADActiveFrame):
            logger.info("🎤 AssemblyAISTTService: VAD active (user making noise)")
            logger.debug(f"   └─ User speaking: {self._user_speaking}, Bot speaking: {self._bot_speaking}")
            self._vad_active = True
            self._vad_inactive_time = None  # Clear inactive timestamp when VAD becomes active
        elif isinstance(frame, VADInactiveFrame):
            logger.info("🎤 AssemblyAISTTService: VAD inactive (user silent)")
            logger.debug(f"   └─ User speaking: {self._user_speaking}, Bot speaking: {self._bot_speaking}")
            self._vad_active = False
            self._vad_inactive_time = time.time()  # Record when VAD became inactive

    async def _handle_bot_speaking(self):
        """Handle bot started speaking event."""
        self._bot_speaking = True
        self._bot_has_ever_spoken = True  # Mark that bot has spoken at least once
        self._bot_started_speaking_time = time.time()  # Track when bot started speaking
        logger.debug(f"🤖 {self}: Bot started speaking at {self._bot_started_speaking_time}")

    async def _handle_bot_silence(self):
        """Handle bot stopped speaking event."""
        self._bot_speaking = False
        self._bot_started_speaking_time = None  # Reset the timestamp
        logger.debug(f"🤖 {self}: Bot stopped speaking")

    async def _should_ignore_transcription(self, transcript: str, is_formatted: bool) -> bool:
        """Determine if a transcription should be ignored.
        
        Following Deepgram's pattern for filtering transcriptions during bot speech.
        Enhanced with VAD state, word count, and timing checks to reduce false positives
        from external talking, background noise, and echoes.
        
        Args:
            transcript: The transcription text to evaluate
            is_formatted: Whether this is a final/formatted transcript
        
        Returns:
            True if the transcription should be ignored, False otherwise
        """
        # Check if fast greeting should be ignored
        time_start = self._time_since_init()
        if self._should_ignore_fast_greeting(transcript, time_start):
            return True
        
        # Check for repeated first message
        if self._should_ignore_first_repeated_message(transcript):
            logger.debug(f"{self}: Ignoring repeated first message")
            return True
        
        # If VAD is inactive for interim transcripts, check if we're in grace period
        # Grace period allows transcripts that AssemblyAI is still processing from audio
        # received before VAD went inactive (accounts for STT processing lag)
        if not self._vad_active and not is_formatted:
            # Allow transcripts within 2 seconds of VAD becoming inactive
            # This accounts for AssemblyAI's processing delay
            if self._vad_inactive_time:
                time_since_vad_inactive = time.time() - self._vad_inactive_time
                if time_since_vad_inactive < 2.0:
                    logger.debug(f"{self}: Accepting interim transcript despite VAD inactive (within grace period: {time_since_vad_inactive:.3f}s): '{transcript}'")
                else:
                    logger.debug(f"{self}: Ignoring interim transcript because VAD inactive too long ({time_since_vad_inactive:.3f}s, likely external speech): '{transcript}'")
                    return True
            else:
                # VAD was never inactive during this session, filter it
                logger.debug(f"{self}: Ignoring interim transcript because VAD inactive (likely external speech or noise): '{transcript}'")
                return True
        
        # If bot is speaking and interruptions are not allowed
        logger.debug(f"Bot speaking: {self._bot_speaking} | allow_interruptions: {self._allow_stt_interruptions}")
        if self._bot_speaking and not self._allow_stt_interruptions:
            if is_formatted:
                logger.debug(f"{self}: Ignoring FINAL transcription because bot is speaking and interruptions disabled: '{transcript}'")
                return True
            else:
                # Allow interim transcripts through for UI feedback even when bot is speaking
                return False
        
        # Enhanced interruption filtering when bot is speaking
        # This prevents false positives from external talking, echoes, or background noise
        if self._bot_speaking:
            word_count = self._transcript_words_count(transcript)
            
            # Single word = always ignore when bot speaking (high chance of false positive)
            if word_count == 1:
                logger.debug(f"{self}: Ignoring interruption because bot is speaking (single word, likely false positive): '{transcript}'")
                return True
            
            # For 2-word phrases when bot is speaking, be more conservative
            # Require it to be a final transcript to reduce false positives
            if word_count == 2 and not is_formatted:
                logger.debug(f"{self}: Ignoring interruption - bot speaking, short interim phrase (likely false positive): '{transcript}'")
                return True
            
            # Check if interruption is too soon after bot started speaking
            # This catches echo or immediate false detections (e.g., bot's own speech being picked up)
            if self._bot_started_speaking_time:
                time_since_bot_started = time.time() - self._bot_started_speaking_time
                if time_since_bot_started < 1.5:  # Less than 1.5 seconds
                    logger.debug(f"{self}: Ignoring interruption - too soon after bot started speaking ({time_since_bot_started:.2f}s, likely echo): '{transcript}'")
                    return True
            
            # Check if this transcript is very recent to last transcription
            # Helps filter duplicate/echo transcriptions or rapid successive false positives
            if self._last_time_transcription:
                time_since_last = time.time() - self._last_time_transcription
                if time_since_last < 2.0:
                    logger.debug(f"{self}: Ignoring interruption - too soon after last transcript ({time_since_last:.2f}s, possible echo/duplicate): '{transcript}'")
                    return True
        
        # If interruptions are allowed, let everything through
        return False

    def _time_since_init(self):
        """Get time elapsed since service initialization."""
        return time.time() - self.start_time

    def _transcript_words_count(self, transcript: str):
        """Count words in transcript."""
        return len(transcript.split(" "))

    def _should_ignore_fast_greeting(self, transcript: str, time_start: float) -> bool:
        """
        Check if a fast greeting (single word, early timing) should be ignored.

        Rules:
        - If bot is speaking: ignore
        - If bot hasn't spoken yet: ignore
        - If bot has spoken at least once: allow (return False)
        """
        # Only applies to single-word transcripts within first second
        if time_start >= 1 or self._transcript_words_count(transcript) != 1:
            return False

        # Ignore if bot is currently speaking
        if self._bot_speaking:
            logger.debug("Ignoring fast greeting - bot is speaking")
            return True

        # Ignore if bot hasn't spoken yet (prevents false early detections)
        if not self._bot_has_ever_spoken:
            logger.debug("Ignoring fast greeting - bot hasn't spoken yet")
            return True

        # Allow if bot has spoken at least once (user responding)
        logger.debug("Accepting fast greeting - bot has spoken at least once")
        return False

    def _is_accum_transcription(self, text: str):
        """Check if text should be accumulated or sent immediately."""
        END_OF_PHRASE_CHARACTERS = ['.', '?', '!']
        
        text = text.strip()
        
        if not text:
            return True
        
        return not text[-1] in END_OF_PHRASE_CHARACTERS
    
    def _append_accum_transcription(self, frame: TranscriptionFrame):
        """Append transcription frame to accumulation buffer."""
        self._last_time_accum_transcription = time.time()
        self._accum_transcription_frames.append(frame)

    def _handle_first_message(self, text):
        """Track the first message received for voicemail and duplicate detection."""
        if self._first_message:
            return 
        
        self._first_message = text
        self._first_message_time = time.time()

    def _should_ignore_first_repeated_message(self, text):
        """Check if this is a repeated first message that should be ignored."""
        if not self._first_message:
            return False
        
        time_since_first_message = time.time() - self._first_message_time
        if time_since_first_message > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        return is_equivalent_basic(text, self._first_message)

    async def _detect_and_handle_voicemail(self, transcript: str):
        """Detect and handle voicemail messages."""
        if not self.detect_voicemail:
            return False

        logger.debug(f"Checking voicemail: {transcript} at {self._time_since_init():.2f}s")
        
        # Only detect voicemail in the first few seconds and after first transcript
        if self._time_since_init() > VOICEMAIL_DETECTION_SECONDS and self._was_first_transcript_receipt:
            return False
        
        if not voicemail.is_text_voicemail(transcript):
            return False
        
        logger.debug("Voicemail detected")

        await self.push_frame(
            TranscriptionFrame(transcript, "", time_now_iso8601(), self._get_language_enum())
        )

        await self.push_frame(
            VoicemailFrame(transcript)
        )

        logger.debug("Voicemail pushed")
        return True

    def _get_language_enum(self):
        """Convert language string to Language enum."""
        if self._language == "es":
            return Language.ES
        elif self._language == "en":
            return Language.EN
        elif self._language == "fr":
            return Language.FR
        elif self._language == "pt":
            return Language.PT
        elif self._language == "de":
            return Language.DE
        elif self._language == "it":
            return Language.IT
        else:
            return Language.EN  # Default fallback

    async def _async_handle_accum_transcription(self, current_time):
        """Handle accumulated transcription timeout."""
        if not self._last_interim_time:
            self._last_interim_time = 0.0

        reference_time = max(self._last_interim_time, self._last_time_accum_transcription)

        if self._fast_response:
            await self._fast_response_send_accum_transcriptions()
            return 
            
        if current_time - reference_time > self._on_no_punctuation_seconds and len(self._accum_transcription_frames):
            logger.debug("Sending accum transcription because of timeout")
            await self._send_accum_transcriptions()
            return

    async def _handle_false_interim(self, current_time):
        """Detect and handle false interim transcripts with enhanced logging.
        
        This detects when a user has stopped speaking by checking if no interim transcripts
        have been received within the threshold time. This is part of "user idle" detection.
        
        Key filtering to reduce false positives:
        - Only triggers if user was marked as speaking
        - Requires VAD to be inactive (user not making noise)
        - Uses configurable time threshold
        """
        # Early exits with detailed logging
        if not self._user_speaking:
            logger.trace("False interim check: User not speaking, skipping")
            return
            
        if not self._last_interim_time:
            logger.trace("False interim check: No interim time recorded, skipping")
            return
        
        # CRITICAL: Don't detect false interims when VAD is active (user still making noise)
        # This prevents premature "user stopped speaking" signals while user is mid-sentence
        # This also helps filter out external talking - if VAD is active, someone is making noise
        # but if it's not the intended user, the transcripts should already be filtered by VAD check
        if self._vad_active:
            logger.trace("False interim check: VAD active, skipping (user still making noise)")
            return

        last_interim_delay = current_time - self._last_interim_time

        if last_interim_delay < self._false_interim_seconds:
            logger.trace(f"False interim check: Delay {last_interim_delay:.3f}s < threshold {self._false_interim_seconds}s, skipping")
            return

        # User has been idle long enough - mark as stopped speaking
        logger.info(f"⏰ User idle detected: {last_interim_delay:.3f}s since last interim (threshold: {self._false_interim_seconds}s)")
        logger.debug(f"   ├─ VAD active: {self._vad_active}")
        logger.debug(f"   ├─ Bot speaking: {self._bot_speaking}")
        logger.debug(f"   └─ Sending UserStoppedSpeakingFrame")
        await self._handle_user_silence()

    async def _async_handler(self, task_name):
        """Async handler for timeouts and fast response processing."""
        while True:
            if not self.is_monitored_task_active(task_name):
                return

            await asyncio.sleep(0.1)
            
            current_time = time.time()

            await self._async_handle_accum_transcription(current_time)
            await self._handle_false_interim(current_time)

    async def _fast_response_send_accum_transcriptions(self):
        """Send accumulated transcriptions immediately if fast response is enabled."""
        if not self._fast_response:
            return

        if len(self._accum_transcription_frames) == 0:
            return

        current_time = time.time()
        last_message_time = max(self._last_interim_time or 0, self._last_time_accum_transcription)

        is_short_sentence = len(self._accum_transcription_frames) <= 2
        is_sentence_end = not self._is_accum_transcription(self._accum_transcription_frames[-1].text)
        time_since_last_message = current_time - last_message_time

        if is_short_sentence:
            if is_sentence_end:
                logger.debug("Fast response: Sending accum transcriptions because short sentence and end of phrase")
                await self._send_accum_transcriptions()
            
            if not is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds:
                logger.debug("Fast response: Sending accum transcriptions because short sentence and timeout")
                await self._send_accum_transcriptions() 
        else:
            if is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds:
                logger.debug("Fast response: Sending accum transcriptions because long sentence and end of phrase")
                await self._send_accum_transcriptions()
            
            if not is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds * 2:
                logger.debug("Fast response: Sending accum transcriptions because long sentence and timeout")
                await self._send_accum_transcriptions()

    def get_stt_response_times(self) -> List[float]:
        """Get the list of STT response durations."""
        return self._stt_response_times.copy()
    
    def get_average_stt_response_time(self) -> float:
        """Get the average STT response duration."""
        if not self._stt_response_times:
            return 0.0
        return sum(self._stt_response_times) / len(self._stt_response_times)

    def clear_stt_response_times(self):
        """Clear the list of STT response durations."""
        self._stt_response_times.clear()

    def log_stt_performance(self):
        """Log STT performance statistics."""
        stats = self.get_stt_stats()
        if stats["count"] > 0:
            logger.info(f"🎯 AssemblyAI STT Performance Summary:")
            logger.info(f"   📊 Total responses: {stats['count']}")
            logger.info(f"   ⏱️  Average time: {stats['average']}s")
            logger.info(f"   🏃 Fastest: {stats['min']}s")
            logger.info(f"   🐌 Slowest: {stats['max']}s")
            logger.info(f"   🕐 Latest: {stats['latest']}s")
            logger.info(f"   📈 All times: {[round(t, 3) for t in self._stt_response_times]}")
