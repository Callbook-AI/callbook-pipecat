#
# pipecat/services/soniox.py
#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import (
    AudioRawFrame,
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
except ModuleNotFoundError:
    logger.error("In order to use Soniox, you need to `pip install pipecat-ai[soniox]`")
    raise

# Constants for optimized behavior (matching Deepgram patterns)
DEFAULT_ON_NO_PUNCTUATION_SECONDS = 2
IGNORE_REPEATED_MSG_AT_START_SECONDS = 1.0  # Reduced from 4 to 1 second - more conservative for Soniox
VOICEMAIL_DETECTION_SECONDS = 10
FALSE_INTERIM_SECONDS = 1.3

def language_to_soniox_language(language: Language) -> Optional[str]:
    """Maps Pipecat Language enum to Soniox language codes."""
    # Soniox uses standard IETF language tags (e.g., "en", "es", "fr-CA")
    # We can mostly pass the language code through directly.
    return language.value

class SonioxSTTService(STTService):
    """
    Optimized Soniox STT Service following Deepgram's proven patterns.
    
    This implementation uses the best practices from Deepgram for:
    - Fast response mode with minimal latency
    - Advanced transcript filtering and accumulation
    - Bot speaking state tracking
    - Voicemail detection
    - Comprehensive performance metrics
    - False interim detection
    """

    class InputParams(BaseModel):
        language: Language = Field(default=Language.EN_US)
        model: str = "stt-rt-v3"
        language_hints: List[str] = ["es"]
        audio_format: str = "pcm_s16le"  # PCM 16-bit little-endian for best latency
        num_channels: int = 1  # Number of audio channels (1 for mono, 2 for stereo) - MUST match Soniox API
        enable_speaker_diarization: bool = False
        enable_language_identification: bool = False
        allow_interruptions: bool = True
        detect_voicemail: bool = True
        fast_response: bool = False
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS
        audio_passthrough: bool = True  # Pass audio through to downstream processors (e.g., for recording)
        context: Optional[str] = None  # Optional context for better accuracy

    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: int = 16000,
        params: InputParams = None,
        # Legacy parameters for backward compatibility
        language: Language = None,
        enable_partials: bool = None,
        allow_interruptions: bool = None,
        audio_passthrough: bool = None,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._websocket = None
        self._receive_task = None
        self._async_handler_task = None
        self._connection_active = False

        # Handle legacy parameters or use params
        if params is None:
            params = self.InputParams()
            if language is not None:
                params.language = language
            if enable_partials is not None:
                params.fast_response = enable_partials
            if allow_interruptions is not None:
                params.allow_interruptions = allow_interruptions
            if audio_passthrough is not None:
                params.audio_passthrough = audio_passthrough

        self._params = params
        self._language = params.language
        self._allow_stt_interruptions = params.allow_interruptions
        self.detect_voicemail = params.detect_voicemail
        self._fast_response = params.fast_response
        self._on_no_punctuation_seconds = params.on_no_punctuation_seconds
        self._audio_passthrough = params.audio_passthrough

        # State tracking (following Deepgram pattern)
        self._user_speaking = False
        self._bot_speaking = True  # Start as True like Deepgram
        self._bot_has_ever_spoken = False  # Track if bot has spoken at least once
        self._bot_started_speaking_time = None  # Track when bot started speaking
        self._vad_active = False
        self._vad_inactive_time = None

        # Performance tracking
        self._stt_response_times = []
        self._current_speech_start_time = None
        self._last_audio_chunk_time = None
        self._audio_chunk_count = 0

        # Transcript tracking
        self._last_interim_time = None
        self._last_sent_transcript = None
        
        # Token accumulation (like the working example)
        self._final_tokens = []  # Accumulate final tokens until endpoint
        self._last_token_time = None

        # First message tracking
        self._first_message = None
        self._first_message_time = None
        self._was_first_transcript_receipt = False

        self.start_time = time.time()
        self._last_time_transcription = time.time()

        logger.info("Soniox STT Service initialized (Deepgram-optimized):")
        logger.info(f"  Model: {params.model}, Language: {params.language.value}")
        logger.info(f"  Allow interruptions: {self._allow_stt_interruptions}")
        logger.info(f"  Fast response: {self._fast_response}, Detect voicemail: {self.detect_voicemail}")
        logger.info(f"  Audio passthrough: {self._audio_passthrough}")
        logger.info(f"  No punctuation timeout: {self._on_no_punctuation_seconds}s")

    def can_generate_metrics(self) -> bool:
        return True

    async def set_language(self, language: Language):
        """Switch language for transcription."""
        logger.info(f"Switching STT language to: [{language}]")
        self._language = language
        # Need to reconnect with new language
        await self._disconnect()
        await self._connect()

    async def start(self, frame: StartFrame):
        await super().start(frame)
        await self._connect()
        
        # Start async handler for fast response and timeout management
        if not self._async_handler_task:
            self._async_handler_task = self.create_task(self._async_handler("async_handler"))

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        """Process an audio chunk for STT transcription.

        Streams audio data to Soniox websocket for real-time transcription.

        :param audio: Audio data as bytes (PCM 16-bit)
        :yield: AudioRawFrame if audio_passthrough is enabled, otherwise None
        """
        # If audio passthrough is enabled, yield the audio frame for downstream processors
        # This must happen BEFORE sending to Soniox to maintain proper frame ordering
        if self._audio_passthrough:
            yield AudioRawFrame(
                audio=audio,
                sample_rate=self.sample_rate,
                num_channels=1
            )

        if not self._websocket or not self._connection_active:
            logger.debug("⚠️  WebSocket not connected, skipping audio chunk")
            logger.debug(f"   WebSocket exists: {self._websocket is not None}")
            logger.debug(f"   Connection active: {self._connection_active}")
            # Even if not connected, we already passed through the audio if enabled
            if not self._audio_passthrough:
                yield None
            return

        if self._current_speech_start_time is None:
            self._current_speech_start_time = time.perf_counter()
            self._audio_chunk_count = 0
            logger.info("=" * 70)
            logger.info("🎤 SPEECH DETECTION STARTED")
            logger.info("=" * 70)

        self._audio_chunk_count += 1
        self._last_audio_chunk_time = time.time()

        # Log every 50 chunks to avoid spam
        if self._audio_chunk_count % 50 == 0:
            elapsed = time.perf_counter() - self._current_speech_start_time
            logger.debug(f"🎤 Audio streaming: {self._audio_chunk_count} chunks sent ({elapsed:.2f}s elapsed)")

        try:
            # logger.debug(f"📤 Sending audio chunk #{self._audio_chunk_count} ({len(audio)} bytes)")
            # Soniox accepts raw binary PCM audio after configuration
            await self._websocket.send(audio)
            # logger.debug(f"✓ Audio chunk sent successfully")
        except websockets.exceptions.ConnectionClosed as e:
            logger.error("=" * 70)
            logger.error("❌ WEBSOCKET CONNECTION CLOSED WHILE SENDING AUDIO")
            logger.error("=" * 70)
            logger.error(f"Close Code: {e.code if hasattr(e, 'code') else 'N/A'}")
            logger.error(f"Close Reason: {e.reason if hasattr(e, 'reason') else 'N/A'}")
            logger.error(f"Chunks sent before disconnect: {self._audio_chunk_count}")
            logger.error("=" * 70)
            logger.warning("Attempting to reconnect...")
            await self._reconnect()
        except Exception as e:
            logger.error("=" * 70)
            logger.error("❌ ERROR SENDING AUDIO TO SONIOX")
            logger.error("=" * 70)
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {e}")
            logger.error(f"Audio chunk size: {len(audio)} bytes")
            logger.error(f"Chunks sent: {self._audio_chunk_count}")
            logger.exception("Full traceback:")
            logger.error("=" * 70)

        # Only yield None if audio passthrough is disabled
        if not self._audio_passthrough:
            yield None

    async def _connect(self):
        """Establish websocket connection to Soniox service."""
        if self._websocket and self._connection_active:
            logger.debug("🔗 Already connected to Soniox, skipping reconnection")
            return

        try:
            logger.info("=" * 70)
            logger.info("🔗 SONIOX CONNECTION ATTEMPT")
            logger.info("=" * 70)
            
            # Validate API key before attempting connection
            if not self._api_key or self._api_key.strip() == "":
                raise ValueError("Soniox API key is empty or invalid")
            
            logger.info(f"✓ API Key validated: {self._api_key[:10]}...{self._api_key[-4:]}")
            logger.info(f"✓ Model: {self._params.model}")
            logger.info(f"✓ Language: {self._language} -> {language_to_soniox_language(self._language)}")
            logger.info(f"✓ Sample Rate: {self.sample_rate} Hz")
            logger.info(f"✓ Audio Format: {self._params.audio_format}")
            logger.info(f"✓ Audio Channels: {self._params.num_channels}")
            logger.info(f"✓ Speaker Diarization: {self._params.enable_speaker_diarization}")
            logger.info(f"✓ Language ID: {self._params.enable_language_identification}")

            # Use the correct Soniox WebSocket endpoint
            uri = "wss://stt-rt.soniox.com/transcribe-websocket"
            
            logger.info(f"📡 WebSocket URI: {uri}")
            logger.info("⏳ Attempting WebSocket connection...")

            # Connect to WebSocket (no auth headers needed, will send config message)
            self._websocket = await websockets.connect(uri)
            self._connection_active = True
            
            logger.info("✅ WebSocket connection established")
            logger.info("📤 Sending configuration message...")
            
            # Build language hints from the language parameter
            language_code = language_to_soniox_language(self._language)
            # Convert "en-US" to "en", "es-ES" to "es", etc.
            language_hint = language_code.split('-')[0] if language_code else "en"
            
            # Send configuration message as per Soniox documentation
            config = {
                "api_key": self._api_key,
                "model": self._params.model,
                "audio_format": self._params.audio_format,
                "sample_rate": self.sample_rate,
                "num_channels": self._params.num_channels,  # MUST be "num_channels" not "num_audio_channels"
                "language_hints": [language_hint],
                "enable_speaker_diarization": self._params.enable_speaker_diarization,
                "enable_language_identification": self._params.enable_language_identification,
                # CRITICAL: Enable endpoint detection like the working example
                "enable_endpoint_detection": True,
            }
            
            # Add context if provided
            if self._params.context:
                config["context"] = self._params.context
            
            logger.info(f"📤 Configuration: {json.dumps({k: v if k != 'api_key' else '***' for k, v in config.items()}, indent=2)}")
            
            # Send configuration as JSON text message
            await self._websocket.send(json.dumps(config))
            
            logger.info("⏳ Waiting for configuration confirmation from Soniox...")
            
            # Wait for confirmation response before proceeding
            try:
                response = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
                response_data = json.loads(response)
                logger.info(f"📨 Received configuration response: {json.dumps(response_data, indent=2)}")
                
                # Check for error in confirmation
                if "error_code" in response_data:
                    raise Exception(f"Soniox config error: {response_data.get('error_message')}")
                    
            except asyncio.TimeoutError:
                raise Exception("Timeout waiting for Soniox configuration confirmation")
            
            logger.info("=" * 70)
            logger.info("✅ SUCCESSFULLY CONFIGURED SONIOX SESSION")
            logger.info("=" * 70)

            if not self._receive_task:
                logger.debug("🎧 Starting receive task handler...")
                self._receive_task = self.create_task(self._receive_task_handler())
            
            logger.debug("📊 Starting TTFB metrics...")
            await self.start_ttfb_metrics()
            logger.debug("📊 Starting processing metrics...")
            await self.start_processing_metrics()
            
            logger.info("✅ Soniox service fully initialized and ready")
            
        except websockets.exceptions.InvalidStatusCode as e:
            logger.error("=" * 70)
            logger.error("❌ SONIOX CONNECTION FAILED - HTTP STATUS ERROR")
            logger.error("=" * 70)
            logger.error(f"Status Code: {e.status_code}")
            logger.error(f"Error Message: {e}")
            logger.error(f"Headers: {e.headers if hasattr(e, 'headers') else 'N/A'}")
            logger.error(f"API Key (masked): {self._api_key[:10]}...{self._api_key[-4:]}")
            logger.error(f"Endpoint: wss://api.soniox.com/v1/speech-to-text-rt")
            logger.error(f"Model: {self._params.model}")
            logger.error(f"Language: {language_to_soniox_language(self._language)}")
            logger.error("Possible Issues:")
            logger.error("  1. Invalid API key")
            logger.error("  2. Invalid model name")
            logger.error("  3. Unsupported language code")
            logger.error("  4. API endpoint URL incorrect")
            logger.error("  5. Account not active or insufficient credits")
            logger.error("=" * 70)
            await self.push_error(ErrorFrame(f"Soniox connection failed: {e}"))
            
        except websockets.exceptions.InvalidURI as e:
            logger.error("=" * 70)
            logger.error("❌ SONIOX CONNECTION FAILED - INVALID URI")
            logger.error("=" * 70)
            logger.error(f"URI Error: {e}")
            logger.error(f"Attempted URI: {uri if 'uri' in locals() else 'Not constructed'}")
            logger.error("=" * 70)
            await self.push_error(ErrorFrame(f"Soniox invalid URI: {e}"))
            
        except Exception as e:
            logger.error("=" * 70)
            logger.error("❌ SONIOX CONNECTION FAILED - UNEXPECTED ERROR")
            logger.error("=" * 70)
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {e}")
            logger.error(f"API Key (masked): {self._api_key[:10]}...{self._api_key[-4:] if len(self._api_key) > 14 else '****'}")
            logger.exception("Full traceback:")
            logger.error("=" * 70)
            await self.push_error(ErrorFrame(f"Soniox connection failed: {e}"))

    async def _disconnect(self):
        """Disconnect from Soniox service and clean up resources."""
        logger.info("=" * 70)
        logger.info("🔌 DISCONNECTING FROM SONIOX")
        logger.info("=" * 70)
        
        self._connection_active = False
        
        # Cancel async handler task
        if self._async_handler_task:
            logger.debug("⏹️  Cancelling async handler task...")
            await self.cancel_task(self._async_handler_task)
            self._async_handler_task = None
            logger.debug("✓ Async handler task cancelled")
        
        # Cancel receive task
        if self._receive_task:
            logger.debug("⏹️  Cancelling receive task...")
            await self.cancel_task(self._receive_task)
            self._receive_task = None
            logger.debug("✓ Receive task cancelled")
        
        # Close websocket
        if self._websocket:
            logger.debug("🔌 Closing WebSocket connection...")
            await self._websocket.close()
            self._websocket = None
            logger.info("✅ Disconnected from Soniox")
        
        logger.info("=" * 70)

    async def _reconnect(self):
        """Attempt to reconnect to Soniox service."""
        logger.warning("=" * 70)
        logger.warning("🔄 ATTEMPTING TO RECONNECT TO SONIOX")
        logger.warning("=" * 70)
        await self._disconnect()
        logger.info("⏳ Waiting 1 second before reconnection attempt...")
        await asyncio.sleep(1)
        await self._connect()

    async def _receive_task_handler(self):
        """Handle incoming transcription messages from Soniox."""
        logger.info("🎧 Receive task handler started and listening for messages...")
        message_count = 0
        
        while self._connection_active:
            try:
                logger.debug("⏳ Waiting for message from Soniox WebSocket...")
                message = await self._websocket.recv()
                message_count += 1
                logger.debug(f"📨 Received message #{message_count} from Soniox")
                logger.debug(f"📨 Raw message length: {len(message)} bytes")
                logger.debug(f"📨 Raw message preview: {message[:200]}..." if len(message) > 200 else f"📨 Raw message: {message}")
                
                await self._on_message(json.loads(message))
                
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning("=" * 70)
                logger.warning("⚠️  SONIOX CONNECTION CLOSED DURING RECEIVE")
                logger.warning("=" * 70)
                logger.warning(f"Close Code: {e.code if hasattr(e, 'code') else 'N/A'}")
                logger.warning(f"Close Reason: {e.reason if hasattr(e, 'reason') else 'N/A'}")
                logger.warning(f"Messages received before close: {message_count}")
                logger.warning("=" * 70)
                break
                
            except json.JSONDecodeError as e:
                logger.error("=" * 70)
                logger.error("❌ JSON DECODE ERROR")
                logger.error("=" * 70)
                logger.error(f"Error: {e}")
                logger.error(f"Message that failed to parse: {message[:500] if 'message' in locals() else 'N/A'}")
                logger.error("=" * 70)
                continue
                
            except Exception as e:
                logger.error("=" * 70)
                logger.error("❌ ERROR IN SONIOX RECEIVE TASK")
                logger.error("=" * 70)
                logger.error(f"Error Type: {type(e).__name__}")
                logger.error(f"Error Message: {e}")
                logger.error(f"Messages received before error: {message_count}")
                logger.exception("Full traceback:")
                logger.error("=" * 70)
                break
        
        logger.info(f"🎧 Receive task handler stopped. Total messages received: {message_count}")

    async def _on_message(self, data: Dict):
        """Process incoming transcription message.
        
        Following the working example pattern:
        - Accumulate final tokens until endpoint is detected
        - Send non-final tokens as interim immediately  
        - Only send TranscriptionFrame when we detect endpoint (gap in tokens or explicit signal)
        """
        try:
            logger.debug("=" * 70)
            logger.debug("📨 PROCESSING SONIOX MESSAGE")
            logger.debug("=" * 70)
            logger.debug(f"Message keys: {list(data.keys())}")
            logger.debug(f"Full message data: {json.dumps(data, indent=2)}")
            
            # Check for error response
            if "error_code" in data:
                logger.error(f"❌ Soniox error: {data.get('error_code')} - {data.get('error_message')}")
                await self.push_error(ErrorFrame(f"Soniox error: {data.get('error_message')}"))
                return
            
            # Check for finished response
            if data.get("finished"):
                logger.info("✅ Soniox session finished")
                # Send any remaining final tokens before finishing
                await self._send_accumulated_transcription()
                return
            
            await self.stop_ttfb_metrics()

            # Parse tokens from current response (like working example)
            tokens = data.get("tokens", [])
            logger.debug(f"📝 Tokens count: {len(tokens)}")
            
            if not tokens:
                # No tokens in this message - might indicate endpoint
                # Check if we have accumulated finals that haven't been sent
                if self._final_tokens:
                    elapsed_since_last_token = time.time() - (self._last_token_time or 0)
                    if elapsed_since_last_token > 0.5:  # 500ms gap indicates endpoint
                        logger.info("🔚 Detected endpoint (token gap) - sending accumulated transcription")
                        await self._send_accumulated_transcription()
                logger.debug("⚠️  No tokens in message, skipping")
                logger.debug("=" * 70)
                return

            non_final_tokens = []
            has_final = False
            
            # Separate final vs non-final tokens (like working example)
            for token in tokens:
                text = token.get("text", "")
                if not text or text == "<end>":  # Skip empty and <end> tokens
                    continue
                    
                if token.get("is_final"):
                    # Final token - add to accumulation buffer
                    self._final_tokens.append(token)
                    has_final = True
                    self._last_token_time = time.time()
                else:
                    # Non-final token - will be sent as interim
                    non_final_tokens.append(token)
            
            # Build texts for interim display (already filtered <end> tokens above)
            final_text = "".join(t["text"] for t in self._final_tokens)
            non_final_text = "".join(t["text"] for t in non_final_tokens)
            combined_text = final_text + non_final_text
            
            logger.debug(f"📝 Final tokens accumulated: {len(self._final_tokens)}")
            logger.debug(f"📝 Non-final tokens: {len(non_final_tokens)}")
            logger.debug(f"📝 Combined text: '{combined_text}'")
            
            # Always send interim transcription to show progress
            if combined_text.strip():
                # Check if we should ignore this transcription
                should_ignore = await self._should_ignore_transcription(combined_text)
                if not should_ignore:
                    # Send as interim (even finals are interim until endpoint detected)
                    await self._on_interim_transcript_message(combined_text, self._language)
            
            # Check if we should send the final transcription now
            # This happens when: no more non-finals AND we have finals
            if has_final and len(non_final_tokens) == 0 and len(self._final_tokens) > 0:
                # All tokens are final and no more non-finals coming - this is likely an endpoint
                logger.info("🔚 Detected endpoint (all final, no non-finals) - sending accumulated transcription")
                await self._send_accumulated_transcription()
            
            logger.debug("=" * 70)
                
        except Exception as e:
            logger.error("=" * 70)
            logger.error("❌ ERROR PROCESSING SONIOX MESSAGE")
            logger.error("=" * 70)
            logger.error(f"Error Type: {type(e).__name__}")
            logger.error(f"Error Message: {e}")
            logger.error(f"Message data: {data if 'data' in locals() else 'N/A'}")
            logger.exception("Full traceback:")
            logger.error("=" * 70)

    async def _send_accumulated_transcription(self):
        """Send accumulated final tokens as a complete TranscriptionFrame.
        
        This is called when we detect an endpoint (no more tokens coming).
        """
        if not self._final_tokens:
            logger.debug("No accumulated tokens to send")
            return
        
        # Build final transcript from accumulated tokens, filtering out <end> tokens
        transcript = "".join(t["text"] for t in self._final_tokens if t["text"] != "<end>")
        
        if not transcript.strip():
            logger.debug("Empty accumulated transcript, skipping")
            self._final_tokens = []
            return
        
        # CRITICAL: Check if we should ignore this transcription (e.g., bot speaking with interruptions disabled)
        should_ignore = await self._should_ignore_transcription(transcript)
        if should_ignore:
            logger.info("=" * 70)
            logger.info("🚫 IGNORING ACCUMULATED FINAL TRANSCRIPT")
            logger.info("=" * 70)
            logger.info(f"📝 '{transcript}'")
            logger.info(f"🔤 Tokens: {len(self._final_tokens)}, Chars: {len(transcript)}")
            logger.info(f"❌ Reason: Bot speaking and interruptions disabled")
            logger.info("=" * 70)
            # Clear accumulated tokens for next utterance
            self._final_tokens = []
            self._last_token_time = None
            return
        
        logger.info("=" * 70)
        logger.info("✅ SENDING ACCUMULATED FINAL TRANSCRIPT")
        logger.info("=" * 70)
        logger.info(f"📝 '{transcript}'")
        logger.info(f"🔤 Tokens: {len(self._final_tokens)}, Chars: {len(transcript)}")
        logger.info("=" * 70)
        
        # Record performance
        self._record_stt_performance(transcript, self._final_tokens)
        
        # Send the final transcription
        await self._on_final_transcript_message(transcript, self._language)
        
        # Clear accumulated tokens for next utterance
        self._final_tokens = []
        self._last_token_time = None
        self._last_final_transcript_time = time.time()
        self._last_time_transcription = time.time()

    async def _on_final_transcript_message(self, transcript: str, language: Language):
        """Handle final transcript - user has FINISHED speaking.
        
        CRITICAL: We send UserStoppedSpeakingFrame BEFORE the TranscriptionFrame.
        This ensures the aggregator receives them in the correct order:
        1. UserStoppedSpeakingFrame clears aggregator's user_speaking state
        2. TranscriptionFrame is received and processed immediately (not buffered)
        
        This ordering is essential because the aggregator checks user_speaking state
        when it receives the transcript. If user_speaking=True, it buffers. If False,
        it processes immediately.
        """
        logger.debug("🔵 Processing final transcript...")
        
        # Check for voicemail detection
        if self.detect_voicemail:
            logger.debug("🔍 Checking for voicemail...")
            await self._detect_and_handle_voicemail(transcript)
        
        # Check for repeated first message BEFORE setting first message
        if self._should_ignore_first_repeated_message(transcript):
            logger.debug(f"⏭️  Ignoring repeated first message: '{transcript}'")
            return
        
        # Handle first message tracking AFTER duplicate check
        self._handle_first_message(transcript)
        
        # CRITICAL: Send UserStoppedSpeakingFrame BEFORE the transcript
        # This ensures the aggregator's user_speaking state is cleared
        # before it receives the TranscriptionFrame
        if self._user_speaking:
            logger.info("⏸️  Sending UserStoppedSpeakingFrame BEFORE transcript")
            # Set state first
            self._user_speaking = False
            self._current_speech_start_time = None
            
            # Send UserStoppedSpeakingFrame in BOTH directions to ensure aggregator gets it
            logger.debug("⬆️  Pushing UserStoppedSpeakingFrame UPSTREAM")
            await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            logger.debug("⬇️  Pushing UserStoppedSpeakingFrame DOWNSTREAM")
            await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.DOWNSTREAM)
            
            logger.info("✅ UserStoppedSpeakingFrame sent in both directions")
            # CRITICAL: Yield control to event loop to ensure the UserStoppedSpeakingFrame
            # is fully processed by the aggregator before we send the TranscriptionFrame.
            # Without this, both frames may be queued simultaneously and processed in
            # the wrong order due to asyncio task scheduling.
            # Increased delay from 0 to 0.05 to ensure proper processing order
            await asyncio.sleep(0.05)
            logger.info("✅ Yielded to event loop - aggregator state should be updated")
        
        # Create transcription frame
        frame = TranscriptionFrame(
            transcript,
            "",
            time_now_iso8601(),
            language
        )
        logger.debug(f"📦 Created TranscriptionFrame: '{transcript}'")
        
        # Send the TranscriptionFrame - aggregator will process it immediately
        logger.info("=" * 70)
        logger.info("⬇️  SENDING FINAL TRANSCRIPTION TO AGGREGATOR")
        logger.info(f"   Text: '{transcript}'")
        logger.info(f"   User speaking: {self._user_speaking} (should be False)")
        logger.info("=" * 70)
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)
        logger.info("✅ TranscriptionFrame sent DOWNSTREAM")
        
        await self.stop_processing_metrics()

    async def _on_interim_transcript_message(self, transcript: str, language: Language):
        """Handle interim transcript.
        
        Shows real-time progress. Triggers user_speaking state ONLY on first call.
        Both non-final tokens AND accumulated finals (before endpoint) are sent as interim.
        """
        logger.debug(f"🟡 Interim: '{transcript[:50]}...' ({len(transcript)} chars)")
        
        # Update interim time for false interim detection
        # This ensures we don't trigger false interim while receiving transcripts
        self._last_interim_time = time.time()
        
        # Only trigger user speaking state if not already active AND VAD hasn't gone inactive
        # This prevents repeated interim transcripts from re-triggering after VAD detected silence
        if not self._user_speaking and not (self._vad_inactive_time and time.time() - self._vad_inactive_time < 2.0):
            logger.info("👤 First interim - triggering user speaking state...")
            await self._handle_user_speaking()
        elif self._vad_inactive_time and time.time() - self._vad_inactive_time < 2.0:
            logger.debug(f"⚠️  Skipping user speaking trigger - VAD went inactive {time.time() - self._vad_inactive_time:.2f}s ago")
        
        frame = InterimTranscriptionFrame(
            transcript,
            "",
            time_now_iso8601(),
            language
        )
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def _record_stt_performance(self, transcript, tokens):
        """Record STT performance metrics."""
        if self._current_speech_start_time:
            elapsed = time.perf_counter() - self._current_speech_start_time
            self._stt_response_times.append(elapsed)
            # Calculate average confidence from tokens
            confidence = sum(t.get("confidence", 0) for t in tokens) / len(tokens) if tokens else 0
            logger.info(f"📊 ⚡ Soniox: ⏱️ STT Response Time: {elapsed:.3f}s")
            logger.info(f"   📝 Final Transcript: '{transcript}'")
            logger.info(f"   🎯 Avg. Confidence: {confidence:.2f}")
            logger.info(f"   📦 Audio chunks processed: {self._audio_chunk_count}")
            self._current_speech_start_time = None

    async def _handle_user_speaking(self):
        """Handle user started speaking event."""
        if not self._user_speaking:
            logger.info("=" * 70)
            logger.info("👤 USER STARTED SPEAKING")
            logger.info("=" * 70)
            logger.debug("⬆️  Pushing StartInterruptionFrame")
            await self.push_frame(StartInterruptionFrame())
            self._user_speaking = True
            logger.debug("⬆️  Pushing UserStartedSpeakingFrame")
            await self.push_frame(UserStartedSpeakingFrame())
            logger.info("✓ User speaking state activated")
            logger.info("=" * 70)
        else:
            logger.debug("👤 User already marked as speaking, skipping")

    async def _handle_user_silence(self):
        """Handle user stopped speaking event."""
        if self._user_speaking:
            logger.info("=" * 70)
            logger.info("👤 USER STOPPED SPEAKING")
            logger.info("=" * 70)
            self._user_speaking = False
            self._current_speech_start_time = None
            logger.debug("⬆️  Pushing UserStoppedSpeakingFrame UPSTREAM")
            await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            logger.info("✓ User silence state activated")
            logger.info("=" * 70)
        else:
            logger.debug("👤 User already marked as not speaking, skipping")
            
    async def _handle_bot_speaking(self):
        """Handle bot started speaking event."""
        self._bot_speaking = True
        self._bot_has_ever_spoken = True
        self._bot_started_speaking_time = time.time()
        logger.debug(f"🤖 {self}: Bot started speaking at {self._bot_started_speaking_time}")

    async def _handle_bot_silence(self):
        """Handle bot stopped speaking event."""
        self._bot_speaking = False
        self._bot_started_speaking_time = None
        logger.debug(f"🤖 {self}: Bot stopped speaking")

    async def _should_ignore_transcription(self, transcript: str) -> bool:
        """Check if transcription should be ignored based on various conditions."""
        
        # Ignore if bot is speaking and interruptions are not allowed
        if self._bot_speaking and not self._allow_stt_interruptions:
            logger.debug(f"Ignoring transcript: bot speaking and interruptions disabled: '{transcript}'")
            return True
        
        # Ignore fast greetings at start
        time_since_init = self._time_since_init()
        if self._should_ignore_fast_greeting(transcript, time_since_init):
            logger.debug(f"Ignoring fast greeting at start: '{transcript}'")
            return True
        
        return False

    def _time_since_init(self):
        """Get time since service initialization."""
        return time.time() - self.start_time

    def _transcript_words_count(self, transcript: str):
        """Count words in transcript."""
        return len(transcript.split())

    def _should_ignore_fast_greeting(self, transcript: str, time_start: float) -> bool:
        """Ignore very fast greetings that might be false positives."""
        if time_start > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        # Common greeting words that might be false positives
        greeting_words = ["hello", "hi", "hey", "hola", "yes", "no"]
        transcript_lower = transcript.lower().strip()
        
        # If it's a single word greeting in the first few seconds, might be false
        words = transcript_lower.split()
        if len(words) == 1 and words[0] in greeting_words and time_start < 1.0:
            return True
        
        return False



    def _handle_first_message(self, text):
        """Track first message for duplicate detection."""
        if self._first_message:
            return
        
        self._first_message = text
        self._first_message_time = time.time()

    def _should_ignore_first_repeated_message(self, text):
        """Check if this is a repeated first message to ignore.
        
        Only ignores EXACT duplicates within 1 second - Soniox is less prone
        to repetition than Deepgram, so we're very conservative here.
        """
        if not self._first_message:
            return False
        
        time_since_first_message = time.time() - self._first_message_time
        if time_since_first_message > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        # EXACT match only (not fuzzy match) to avoid false positives
        return text.strip() == self._first_message.strip()

    async def _detect_and_handle_voicemail(self, transcript: str):
        """Detect and handle voicemail messages."""
        time_since_init = self._time_since_init()
        
        # Only detect voicemail in the first N seconds
        if time_since_init > VOICEMAIL_DETECTION_SECONDS:
            return
        
        # Check if transcript matches voicemail patterns
        if voicemail(transcript):
            logger.info(f"🔊 Voicemail detected: '{transcript}'")
            await self.push_frame(VoicemailFrame())



    async def _async_handle_accum_transcription(self, current_time):
        """Handle accumulated transcriptions with timeout.
        
        Note: With Soniox, we send transcripts immediately when is_final=true,
        so this method is mostly for compatibility. The timeout logic is not needed.
        """
        # No longer accumulating - transcripts are sent immediately
        pass

    async def _handle_false_interim(self, current_time):
        """Handle false interim detection.
        
        DISABLED: Soniox provides explicit endpoint detection via <end> tokens,
        so we don't need false interim detection. This was causing premature
        UserStoppedSpeakingFrame to be sent before all tokens arrived.
        """
        return
        
        # Legacy code kept for reference:
        # if not self._user_speaking:
        #     return
        # if not self._last_interim_time:
        #     return
        # if self._vad_active:
        #     return
        # # Don't trigger false interim if we have accumulated final tokens waiting to be sent
        # if self._final_tokens:
        #     return
        #
        # last_interim_delay = current_time - self._last_interim_time
        #
        # if last_interim_delay > FALSE_INTERIM_SECONDS:
        #     return
        #
        # logger.debug("False interim detected")
        # await self._handle_user_silence()

    async def _async_handler(self, task_name):
        """Async handler for timeout management and false interim detection."""
        while True:
            await asyncio.sleep(0.1)
            
            current_time = time.time()

            # Check if we should send accumulated transcription due to timeout
            if self._final_tokens and self._last_token_time:
                elapsed = current_time - self._last_token_time
                # If no new tokens for 800ms, consider it an endpoint
                if elapsed > 0.8:
                    logger.debug(f"🔚 Endpoint detected by timeout ({elapsed:.2f}s since last token)")
                    await self._send_accumulated_transcription()

            await self._async_handle_accum_transcription(current_time)
            await self._handle_false_interim(current_time)

    async def _fast_response_send_accum_transcriptions(self):
        """Send accumulated transcriptions immediately if fast response is enabled.
        
        Note: No longer used with Soniox as transcripts are sent immediately.
        Kept for compatibility.
        """
        # No longer accumulating - transcripts are sent immediately
        pass

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

    def get_stt_response_times(self) -> List[float]:
        """Get list of STT response times."""
        return self._stt_response_times
    
    def get_average_stt_response_time(self) -> float:
        """Get average STT response time."""
        if not self._stt_response_times:
            return 0.0
        return sum(self._stt_response_times) / len(self._stt_response_times)

    def clear_stt_response_times(self):
        """Clear STT response times."""
        self._stt_response_times = []

    def log_stt_performance(self):
        """Log comprehensive STT performance statistics."""
        stats = self.get_stt_stats()
        logger.info("=" * 50)
        logger.info("📊 Soniox STT Performance Summary")
        logger.info("=" * 50)
        logger.info(f"Total transcriptions: {stats['count']}")
        logger.info(f"Average response time: {stats['average']:.3f}s")
        logger.info(f"Min response time: {stats['min']:.3f}s")
        logger.info(f"Max response time: {stats['max']:.3f}s")
        logger.info(f"Latest response time: {stats['latest']:.3f}s")
        logger.info("=" * 50)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames for bot speaking state and interruption handling."""
        # Log all frames for debugging
        # frame_name = type(frame).__name__
        # logger.debug(f"🎯 Processing frame: {frame_name} (direction: {direction.name})")
        
        await super().process_frame(frame, direction)
        
        # Handle bot speaking state for interruption detection
        if isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("🤖 Received BotStartedSpeakingFrame")
            await self._handle_bot_speaking()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("🤖 Received BotStoppedSpeakingFrame")
            await self._handle_bot_silence()
        elif isinstance(frame, VADActiveFrame):
            self._vad_active = True
            self._vad_inactive_time = None  # Reset when VAD becomes active
            logger.info(f"🎤 {self}: VAD ACTIVE (voice detected)")
        elif isinstance(frame, VADInactiveFrame):
            self._vad_active = False
            self._vad_inactive_time = time.time()
            logger.info(f"🎤 {self}: VAD INACTIVE (silence detected at {self._vad_inactive_time})")