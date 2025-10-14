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
        model: str = "stt-rt-preview"  # Soniox's real-time model (use stt-rt-preview or stt-rt-preview-v2)
        audio_format: str = "pcm_s16le"  # PCM 16-bit little-endian for best latency
        num_channels: int = 1  # Number of audio channels (1 for mono, 2 for stereo) - MUST match Soniox API
        enable_speaker_diarization: bool = False
        enable_language_identification: bool = False
        allow_interruptions: bool = True
        detect_voicemail: bool = True
        fast_response: bool = False
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS
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

        self._params = params
        self._language = params.language
        self._allow_stt_interruptions = params.allow_interruptions
        self.detect_voicemail = params.detect_voicemail
        self._fast_response = params.fast_response
        self._on_no_punctuation_seconds = params.on_no_punctuation_seconds

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

        # Transcript accumulation (following Deepgram pattern)
        self._accum_transcription_frames = []
        self._last_time_accum_transcription = time.time()
        self._last_interim_time = None
        self._last_sent_transcript = None

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
        :yield: None (transcription frames are pushed via callbacks)
        """
        if not self._websocket or not self._connection_active:
            logger.debug("⚠️  WebSocket not connected, skipping audio chunk")
            logger.debug(f"   WebSocket exists: {self._websocket is not None}")
            logger.debug(f"   Connection active: {self._connection_active}")
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
        
        Handles transcript messages from Soniox with Deepgram-style processing.
        Soniox returns 'tokens' array with 'is_final' flag per token.
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
                return
            
            await self.stop_ttfb_metrics()

            # Soniox uses 'tokens' not 'words'
            tokens = data.get("tokens", [])
            logger.debug(f"📝 Tokens count: {len(tokens)}")
            
            if not tokens:
                logger.debug("⚠️  No tokens in message, skipping")
                logger.debug("=" * 70)
                return

            # Build transcript from tokens
            transcript = " ".join(t["text"] for t in tokens)
            
            # Check if all tokens are final (Soniox marks each token with is_final)
            is_final = all(t.get("is_final", False) for t in tokens)
            
            logger.debug(f"📝 Transcript: '{transcript}'")
            logger.debug(f"✓ Is Final: {is_final}")
            logger.debug(f"✓ Length: {len(transcript)} chars, {len(tokens)} tokens")

            if not transcript.strip():
                logger.debug("⚠️  Empty transcript after stripping, skipping")
                logger.debug("=" * 70)
                return

            # Check if we should ignore this transcription
            should_ignore = await self._should_ignore_transcription(transcript)
            logger.debug(f"✓ Should ignore: {should_ignore}")
            
            if should_ignore:
                logger.debug("⏭️  Transcript ignored based on filtering rules")
                logger.debug("=" * 70)
                return

            # Update interim time for false interim detection
            if not is_final:
                self._last_interim_time = time.time()
                logger.debug(f"⏰ Updated interim time: {self._last_interim_time}")

            timestamp = time_now_iso8601()
            language_enum = self._language

            if is_final:
                logger.info("=" * 70)
                logger.info("✅ FINAL TRANSCRIPT RECEIVED")
                logger.info("=" * 70)
                logger.info(f"📝 '{transcript}'")
                logger.info(f"🔤 Tokens: {len(tokens)}, Chars: {len(transcript)}")
                logger.info("=" * 70)
                
                self._record_stt_performance(transcript, tokens)
                await self._on_final_transcript_message(transcript, language_enum)
                self._last_final_transcript_time = time.time()
                self._last_time_transcription = time.time()
            else:
                logger.debug("⏳ INTERIM TRANSCRIPT")
                logger.debug(f"📝 '{transcript}'")
                await self._on_interim_transcript_message(transcript, language_enum)
            
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

    async def _on_final_transcript_message(self, transcript: str, language: Language):
        """Handle final transcript following Deepgram pattern."""
        logger.debug("🔵 Processing final transcript...")
        
        # Check for voicemail detection
        if self.detect_voicemail:
            logger.debug("🔍 Checking for voicemail...")
            await self._detect_and_handle_voicemail(transcript)
        
        # Check for repeated first message BEFORE setting first message
        # This prevents interim transcripts from polluting the first message tracking
        if self._should_ignore_first_repeated_message(transcript):
            logger.debug(f"⏭️  Ignoring repeated first message: '{transcript}'")
            return
        
        # Handle first message tracking AFTER duplicate check
        # This ensures we only track FINAL transcripts as first message
        self._handle_first_message(transcript)
        
        logger.debug("👤 Triggering user speaking state...")
        await self._handle_user_speaking()
        
        # Create transcription frame
        frame = TranscriptionFrame(
            transcript,
            "",
            time_now_iso8601(),
            language
        )
        logger.debug(f"📦 Created TranscriptionFrame: '{transcript}'")
        
        # Handle accumulation or immediate sending based on fast response mode
        is_accum = self._is_accum_transcription(transcript)
        logger.debug(f"✓ Should accumulate: {is_accum} (ends with punctuation: {not is_accum})")
        
        if is_accum:
            logger.debug("📥 Appending to accumulation buffer...")
            self._append_accum_transcription(frame)
        else:
            logger.debug("📥 Appending to buffer and sending immediately...")
            self._append_accum_transcription(frame)
            await self._send_accum_transcriptions()

    async def _on_interim_transcript_message(self, transcript: str, language: Language):
        """Handle interim transcript."""
        logger.debug("🟡 Processing interim transcript...")
        logger.debug("👤 Triggering user speaking state...")
        await self._handle_user_speaking()
        
        frame = InterimTranscriptionFrame(
            transcript,
            "",
            time_now_iso8601(),
            language
        )
        logger.debug(f"📦 Created InterimTranscriptionFrame: '{transcript}'")
        logger.debug(f"⬇️  Pushing InterimTranscriptionFrame DOWNSTREAM")
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)
        logger.debug("✓ InterimTranscriptionFrame pushed")

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

    def _is_accum_transcription(self, text: str):
        """Check if transcription should be accumulated."""
        END_OF_PHRASE_CHARACTERS = ['.', '?', '!']
        
        text = text.strip()
        if not text:
            return True
        
        return not text[-1] in END_OF_PHRASE_CHARACTERS
    
    def _append_accum_transcription(self, frame: TranscriptionFrame):
        """Append transcription frame to accumulation buffer."""
        self._last_time_accum_transcription = time.time()
        self._accum_transcription_frames.append(frame)
        logger.debug(f"Accumulated transcript: '{frame.text}' (total: {len(self._accum_transcription_frames)})")

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

    async def _send_accum_transcriptions(self):
        """Send accumulated transcriptions following Deepgram/AssemblyAI pattern."""
        logger.info("=" * 70)
        logger.info("📤 SENDING ACCUMULATED TRANSCRIPTIONS")
        logger.info("=" * 70)
        
        if not len(self._accum_transcription_frames):
            logger.debug("⚠️  No accumulated frames to send")
            logger.info("=" * 70)
            return

        # Combine all transcripts into one message
        full_text = " ".join([frame.text for frame in self._accum_transcription_frames])
        logger.info(f"📝 Combined transcript: '{full_text}'")
        logger.info(f"📊 Total frames: {len(self._accum_transcription_frames)}")
        
        # Check if this is a DUPLICATE transcript
        if full_text.strip() == self._last_sent_transcript:
            logger.warning(f"⚠️  DUPLICATE transcript detected, skipping: '{full_text}'")
            logger.warning(f"   Last sent: '{self._last_sent_transcript}'")
            self._accum_transcription_frames = []
            logger.info("=" * 70)
            return
        
        logger.info(f"✓ New transcript (not duplicate)")
        
        # Store what we're sending for deduplication
        self._last_sent_transcript = full_text.strip()
        
        # Ensure transport processes UserStoppedSpeakingFrame BEFORE TranscriptionFrame
        # This prevents race condition in transport's frame emulation logic
        was_user_speaking = self._user_speaking
        logger.info(f"👤 User was speaking: {was_user_speaking}")
        
        if was_user_speaking:
            logger.info("⏸️  Stopping user speaking state before sending transcript...")
            self._user_speaking = False
            self._current_speech_start_time = None
            logger.debug("⬆️  Pushing UserStoppedSpeakingFrame UPSTREAM")
            await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            logger.info("✓ UserStoppedSpeakingFrame sent UPSTREAM")
            
            # Give transport time to process UserStoppedSpeakingFrame
            logger.debug("⏳ Waiting 1ms for transport to process UserStoppedSpeakingFrame...")
            await asyncio.sleep(0.001)  # 1ms for async frame processing
            logger.debug("✓ Wait complete")
        
        # Now send TranscriptionFrame
        logger.info("⬇️  Pushing TranscriptionFrame DOWNSTREAM")
        logger.info(f"   Text: '{full_text}'")
        logger.info(f"   Language: {self._language}")
        
        await self.push_frame(
            TranscriptionFrame(
                full_text,
                "",
                time_now_iso8601(),
                self._language
            ),
            FrameDirection.DOWNSTREAM
        )
        logger.info("✅ TranscriptionFrame sent DOWNSTREAM")
        
        self._accum_transcription_frames = []
        logger.debug("📊 Stopping processing metrics...")
        await self.stop_processing_metrics()
        
        logger.info("=" * 70)

    async def _async_handle_accum_transcription(self, current_time):
        """Handle accumulated transcriptions with timeout."""
        if not self._last_interim_time:
            self._last_interim_time = 0.0

        reference_time = max(self._last_interim_time, self._last_time_accum_transcription)

        if self._fast_response:
            await self._fast_response_send_accum_transcriptions()
            return
            
        if current_time - reference_time > self._on_no_punctuation_seconds and len(self._accum_transcription_frames):
            logger.debug("Sending accum transcription because of timeout")
            await self._send_accum_transcriptions()

    async def _handle_false_interim(self, current_time):
        """Handle false interim detection."""
        if not self._user_speaking:
            return
        if not self._last_interim_time:
            return
        if self._vad_active:
            return

        last_interim_delay = current_time - self._last_interim_time

        if last_interim_delay > FALSE_INTERIM_SECONDS:
            return

        logger.debug("False interim detected")
        await self._handle_user_silence()

    async def _async_handler(self, task_name):
        """Async handler for timeout management and false interim detection."""
        while True:
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

        if self._vad_active:
            # Do not send if VAD is active and it's been less than 10 seconds
            if self._first_message_time and (current_time - self._first_message_time) > 10.0:
                return

        last_message_time = max(self._last_interim_time, self._last_time_accum_transcription)

        is_short_sentence = len(self._accum_transcription_frames) <= 2
        is_sentence_end = not self._is_accum_transcription(self._accum_transcription_frames[-1].text)
        time_since_last_message = current_time - last_message_time

        if is_short_sentence:
            if is_sentence_end:
                logger.debug("Fast response: Sending accum transcriptions - short sentence & end of phrase")
                await self._send_accum_transcriptions()
            elif time_since_last_message > self._on_no_punctuation_seconds:
                logger.debug("Fast response: Sending accum transcriptions - short sentence & timeout")
                await self._send_accum_transcriptions()
        else:
            if is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds:
                logger.debug("Fast response: Sending accum transcriptions - long sentence & end of phrase")
                await self._send_accum_transcriptions()
            elif not is_sentence_end and time_since_last_message > self._on_no_punctuation_seconds * 2:
                logger.debug("Fast response: Sending accum transcriptions - long sentence & timeout")
                await self._send_accum_transcriptions()

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
            logger.info(f"🎤 {self}: VAD ACTIVE (voice detected)")
        elif isinstance(frame, VADInactiveFrame):
            self._vad_active = False
            self._vad_inactive_time = time.time()
            logger.info(f"🎤 {self}: VAD INACTIVE (silence detected)")