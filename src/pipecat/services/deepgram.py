#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import re
import time
import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Callable

import aiohttp
import json
import websockets

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
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADActiveFrame,
    VADInactiveFrame,
    VoicemailFrame,
    STTRestartFrame
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import STTService, TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.time import time_now_iso8601
from pipecat.utils.rex import regex_list_matches
from pipecat.utils.string import is_equivalent_basic
from pipecat.utils.text import voicemail
from pipecat.services.gladia import GladiaSTTService


# See .env.example for Deepgram configuration needed
try:
    from deepgram import (
        AsyncListenWebSocketClient,
        DeepgramClient,
        DeepgramClientOptions,
        ErrorResponse,
        LiveOptions,
        LiveResultResponse,
        LiveTranscriptionEvents,
        SpeakOptions,
    )
except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error(
        "In order to use Deepgram, you need to `pip install pipecat-ai[deepgram]`. Also, set `DEEPGRAM_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


DEFAULT_ON_NO_PUNCTUATION_SECONDS = 3
IGNORE_REPEATED_MSG_AT_START_SECONDS = 4
VOICEMAIL_DETECTION_SECONDS = 10
FALSE_INTERIM_SECONDS = 1.3


class DeepgramTTSService(TTSService):
    def __init__(
        self,
        *,
        api_key: str,
        voice: str = "aura-helios-en",
        sample_rate: Optional[int] = None,
        encoding: str = "linear16",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._settings = {
            "encoding": encoding,
        }
        self.set_voice(voice)
        self._deepgram_client = DeepgramClient(api_key=api_key)

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.debug(f"{self}: Generating TTS [{text}]")

        options = SpeakOptions(
            model=self._voice_id,
            encoding=self._settings["encoding"],
            sample_rate=self.sample_rate,
            container="none",
        )

        try:
            await self.start_ttfb_metrics()

            response = await asyncio.to_thread(
                self._deepgram_client.speak.v("1").stream, {"text": text}, options
            )

            await self.start_tts_usage_metrics(text)
            yield TTSStartedFrame()

            # The response.stream_memory is already a BytesIO object
            audio_buffer = response.stream_memory

            if audio_buffer is None:
                raise ValueError("No audio data received from Deepgram")

            # Read and yield the audio data in chunks
            audio_buffer.seek(0)  # Ensure we're at the start of the buffer
            chunk_size = 8192  # Use a fixed buffer size
            while True:
                await self.stop_ttfb_metrics()
                chunk = audio_buffer.read(chunk_size)
                if not chunk:
                    break
                frame = TTSAudioRawFrame(audio=chunk, sample_rate=self.sample_rate, num_channels=1)
                yield frame

                yield TTSStoppedFrame()

        except Exception as e:
            logger.exception(f"{self} exception: {e}")
            yield ErrorFrame(f"Error getting audio: {str(e)}")


class DeepgramSiDetector:
    """
    Connects to Deepgram in Spanish, streams raw audio,
    and calls `callback(transcript)` whenever a final
    transcript contains "si", "sí", "si!", "sí!", etc.
    """
    def __init__(
        self,
        api_key: str,
        callback: Callable[[str], None],
        url: str = "wss://api.deepgram.com/v1/listen",
        sample_rate: int = 16_000,
    ):
        # callback to invoke on detection
        self._callback = callback
        # match standalone si or sí (case-insensitive), optional !/¡
        self._pattern = re.compile(r'\b(?:si|sí)\b[!¡]?', re.IGNORECASE)

        # build a Deepgram client
        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(
                url=url,
                options={"keepalive": "true"}
            ),
        )

        # settings for Spanish transcription
        self._settings = {
            "encoding":       "linear16",
            "language":       "es",         
            "model":          "general",
            "sample_rate":    sample_rate,
            "interim_results": False,      
            "punctuate":      True,
        }

        self._conn = None  
        self.start_times = set()

    async def start(self):
        """
        Open the Deepgram websocket. Must be called before sending audio.
        """
        try:
            self._conn = self._client.listen.asyncwebsocket.v("1")
            self._conn.on(
                LiveTranscriptionEvents.Transcript,
                self._on_transcript_event
            )
            await self._conn.start(options=self._settings)
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector.start: {e}")

    async def send_audio(self, chunk: bytes):
        """
        Send a chunk of raw audio (bytes) to Deepgram.
        """
        try:
            if not self._conn or not self._conn.is_connected:
                raise RuntimeError("Connection not started. Call start() first.")
            await self._conn.send(chunk)
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector.send_audio: {e}")

    async def stop(self):
        """
        Gracefully close the Deepgram websocket when you’re done sending.
        """
        try:
            if self._conn and self._conn.is_connected:
                await self._conn.finish()
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector.stop: {e}")

    async def _on_transcript_event(self, *args, **kwargs):
        """
        Internal handler for every transcription event.
        Filters down to final transcripts and applies the "si" regex.
        """
        try:
            result = kwargs.get("result")
            if not result:
                return

            alts = result.channel.alternatives
            if not alts:
                return

            transcript = alts[0].transcript.strip()
            if self._pattern.search(transcript):

                if result.start in self.start_times: return
                logger.debug("Si detected")
                
                self.start_times.add(result.start)
                
                logger.debug("Si detected")

                await self._callback(result)
        except Exception as e:
            logger.exception(f"{self} exception in DeepgramSiDetector._on_transcript_event: {e}")


class DeepgramGladiaDetector:
    """
    Connects to Gladia for high-accuracy Spanish transcription,
    streams raw audio, and calls `callback(transcript)` whenever a final
    transcript is received. This provides enhanced accuracy for Spanish
    while maintaining Deepgram's real-time capabilities.
    """
    def __init__(
        self,
        api_key: str,
        callback: Callable[[str, str, float], None],  # transcript, language, confidence
        language: Language = Language.ES,
        url: str = "https://api.gladia.io/v2/live",
        sample_rate: int = 16_000,
        timeout_seconds: float = 2.0,  # Timeout for Gladia response
    ):
        self._api_key = api_key
        self._callback = callback
        self._language = language
        self._url = url
        self._sample_rate = sample_rate
        self._timeout_seconds = timeout_seconds
        
        # WebSocket connection
        self._websocket = None
        self._receive_task = None
        self._connection_id = None
        
        # Timing and tracking
        self.start_times = set()
        self._last_transcript_time = None
        self._pending_finals = {}  # Track pending final transcripts
        
        # Gladia configuration optimized for Spanish
        self._settings = {
            "encoding": "wav/pcm",
            "bit_depth": 16,
            "sample_rate": sample_rate,
            "channels": 1,
            "model": "solaria-1",  # Best Spanish model
            "endpointing": 0.3,  # Faster endpointing for Spanish
            "maximum_duration_without_endpointing": 4,
            "language_config": {
                "languages": [self._language_to_gladia_code(language)],
                "code_switching": False,
            },
            "pre_processing": {
                "audio_enhancer": True,  # Enhanced for better Spanish accuracy
                "speech_threshold": 0.5,
            },
            "realtime_processing": {
                "words_accurate_timestamps": True,
            },
            "messages_config": {
                "receive_final_transcripts": True,
                "receive_speech_events": True,
                "receive_pre_processing_events": False,
                "receive_realtime_processing_events": False,
                "receive_post_processing_events": False,
                "receive_acknowledgments": False,
                "receive_errors": True,
                "receive_lifecycle_events": False
            }
        }

    def _language_to_gladia_code(self, language: Language) -> str:
        """Convert Pipecat Language to Gladia language code."""
        lang_map = {
            Language.ES: "es",
            Language.EN: "en",
            Language.FR: "fr",
            Language.DE: "de",
            Language.IT: "it",
            Language.PT: "pt",
        }
        return lang_map.get(language, "es")  # Default to Spanish

    async def start(self):
        """
        Initialize Gladia connection and start receiving transcripts.
        """
        try:
            logger.info("🎯 DeepgramGladiaDetector: Starting Gladia complementary service")
            await self._setup_gladia()
            await self._connect()
            logger.info("✅ DeepgramGladiaDetector: Gladia service ready")
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector exception in start: {e}")
            raise

    async def send_audio(self, chunk: bytes):
        """
        Send a chunk of raw audio (bytes) to Gladia.
        """
        try:
            if not self._websocket:
                logger.warning("⚠️ DeepgramGladiaDetector: WebSocket not connected, skipping audio chunk")
                return
                
            # Send audio data as base64
            audio_data = {
                "type": "audio_chunk",
                "data": {
                    "chunk": chunk.hex()  # Send as hex for better compatibility
                }
            }
            await self._websocket.send(json.dumps(audio_data))
            
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector exception in send_audio: {e}")

    async def stop(self):
        """
        Gracefully close the Gladia websocket when done.
        """
        try:
            logger.info("🛑 DeepgramGladiaDetector: Stopping Gladia service")
            
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
            
            if self._websocket:
                # Send stop recording message
                stop_message = {"type": "stop_recording"}
                await self._websocket.send(json.dumps(stop_message))
                await self._websocket.close()
                
            logger.info("✅ DeepgramGladiaDetector: Gladia service stopped")
            
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector exception in stop: {e}")

    async def _setup_gladia(self):
        """
        Setup Gladia session and get connection URL.
        """
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "X-Gladia-Key": self._api_key,
                    "Content-Type": "application/json"
                }
                
                payload = {
                    "x_gladia_key": self._api_key,
                    "reinitialize_session": True,
                    "live_config": self._settings
                }
                
                logger.debug("🔧 DeepgramGladiaDetector: Setting up Gladia session")
                async with session.post(self._url, json=payload, headers=headers) as response:
                    if response.status != 201:
                        error_text = await response.text()
                        raise Exception(f"Failed to create Gladia session: {response.status} - {error_text}")
                    
                    data = await response.json()
                    self._connection_id = data.get("id")
                    logger.info(f"✅ DeepgramGladiaDetector: Gladia session created - ID: {self._connection_id}")
                    
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector exception in _setup_gladia: {e}")
            raise

    async def _connect(self):
        """
        Connect to Gladia WebSocket.
        """
        try:
            if not self._connection_id:
                raise Exception("No connection ID available")
            
            ws_url = f"wss://api.gladia.io/v2/live/{self._connection_id}"
            logger.debug(f"🔌 DeepgramGladiaDetector: Connecting to Gladia WebSocket: {ws_url}")
            
            self._websocket = await websockets.connect(ws_url)
            self._receive_task = asyncio.create_task(self._receive_task_handler())
            
            logger.info("✅ DeepgramGladiaDetector: Connected to Gladia WebSocket")
            
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector exception in _connect: {e}")
            raise

    async def _receive_task_handler(self):
        """
        Handle incoming messages from Gladia WebSocket.
        """
        try:
            logger.debug("📡 DeepgramGladiaDetector: Starting message receiver")
            
            async for message in self._websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ DeepgramGladiaDetector: Invalid JSON received: {e}")
                except Exception as e:
                    logger.exception(f"❌ DeepgramGladiaDetector: Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 DeepgramGladiaDetector: WebSocket connection closed")
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector exception in _receive_task_handler: {e}")

    async def _handle_message(self, data: dict):
        """
        Handle different types of messages from Gladia.
        """
        try:
            msg_type = data.get("type")
            
            if msg_type == "transcript":
                await self._handle_transcript_message(data)
            elif msg_type == "speech_event":
                await self._handle_speech_event(data)
            elif msg_type == "error":
                logger.error(f"❌ DeepgramGladiaDetector: Gladia error: {data}")
            elif msg_type == "acknowledgment":
                logger.debug(f"✅ DeepgramGladiaDetector: Gladia acknowledgment: {data}")
            else:
                logger.debug(f"📨 DeepgramGladiaDetector: Unknown message type: {msg_type}")
                
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector exception in _handle_message: {e}")

    async def _handle_transcript_message(self, data: dict):
        """
        Handle transcript messages from Gladia.
        """
        try:
            transcript_data = data.get("data", {})
            transcript = transcript_data.get("transcript", "").strip()
            confidence = transcript_data.get("confidence", 0.0)
            language = transcript_data.get("language", "es")
            is_final = transcript_data.get("is_final", False)
            start_time = transcript_data.get("start_time", 0)
            
            if not transcript:
                return
            
            logger.debug(f"📝 DeepgramGladiaDetector: Transcript ({'final' if is_final else 'interim'}): '{transcript}' (confidence: {confidence:.2f})")
            
            # Only process final transcripts for accuracy
            if is_final and confidence > 0.6:  # Higher confidence threshold for Spanish
                # Avoid duplicates
                if start_time in self.start_times:
                    logger.debug(f"🔄 DeepgramGladiaDetector: Duplicate transcript ignored (start_time: {start_time})")
                    return
                    
                self.start_times.add(start_time)
                self._last_transcript_time = time.time()
                
                logger.info(f"🎯 DeepgramGladiaDetector: High-quality final transcript: '{transcript}' (confidence: {confidence:.2f})")
                
                # Call the callback with enhanced transcript
                await self._callback(transcript, language, confidence)
                
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector exception in _handle_transcript_message: {e}")

    async def _handle_speech_event(self, data: dict):
        """
        Handle speech events from Gladia.
        """
        try:
            event_data = data.get("data", {})
            event_type = event_data.get("type")
            
            if event_type in ["speech_started", "speech_stopped"]:
                logger.debug(f"🎤 DeepgramGladiaDetector: Speech event - {event_type}")
                
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector exception in _handle_speech_event: {e}")


class DeepgramSTTService(STTService):
    def __init__(
        self,
        *,
        api_key: str,
        url: str = "",
        sample_rate: Optional[int] = None,
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS,
        live_options: Optional[LiveOptions] = None,
        addons: Optional[Dict] = None,
        detect_voicemail: bool = True,  
        allow_interruptions: bool = True,
        gladia_api_key: Optional[str] = None,  # Gladia API key for enhanced accuracy
        gladia_timeout_seconds: float = 2.0,   # Timeout for Gladia response
        use_gladia_for_finals: bool = False,    # Enable Gladia for final transcripts
        **kwargs,
    ):
        sample_rate = sample_rate or (live_options.sample_rate if live_options else None)
        super().__init__(sample_rate=sample_rate, **kwargs)

        default_options = LiveOptions(
            encoding="linear16",
            language=Language.EN,
            model="nova-3-general",
            channels=1,
            interim_results=True,
            smart_format=True,
            punctuate=True,
            profanity_filter=True,
            vad_events=False,
        )

        merged_options = default_options
        if live_options:
            merged_options = LiveOptions(**{**default_options.to_dict(), **live_options.to_dict()})

        # deepgram connection requires language to be a string
        if isinstance(merged_options.language, Language) and hasattr(
            merged_options.language, "value"
        ):
            merged_options.language = merged_options.language.value
        
        self.language = merged_options.language  # Store language for Gladia setup
        self.api_key = api_key
        self.detect_voicemail = detect_voicemail  
        self._allow_stt_interruptions = allow_interruptions
        self._gladia_api_key = gladia_api_key
        self._gladia_timeout_seconds = gladia_timeout_seconds
        self._use_gladia_for_finals = use_gladia_for_finals and gladia_api_key is not None
        
        logger.debug(f"Allow ** interruptions: {self._allow_stt_interruptions}")
        if self._use_gladia_for_finals:
            logger.info(f"🎯 Gladia enhanced transcription enabled for language: {self.language}")

        self._settings = merged_options.to_dict()
        self._addons = addons
        self._user_speaking = False
        self._bot_speaking = True
        self._on_no_punctuation_seconds = on_no_punctuation_seconds
        self._vad_active = False

        self._first_message = None
        self._first_message_time = None
        self._last_interim_time = None
        self._restarted = False

        # Gladia integration flags
        self._ignore_deepgram_finals = False  # Flag to ignore Deepgram finals when Gladia provides them
        self._pending_deepgram_finals = {}    # Store Deepgram finals with timeout
        self._gladia_response_times = []      # Track Gladia response performance

        self._setup_sibling_deepgram()
        self._setup_complementary_gladia()


        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(
                url=url,
                options={"keepalive": "true"}, 
            ),
        )

        if self.vad_enabled:
            logger.debug(f"Deepgram VAD Enabled: {self.vad_enabled}")
            self._register_event_handler("on_speech_started")
            self._register_event_handler("on_utterance_end")

        self._async_handler_task = None
        self._accum_transcription_frames = []
        self._last_time_accum_transcription = time.time()
        self._last_time_transcription = time.time()
        self._was_first_transcript_receipt = False

        self.start_time = time.time()
        
        self._stt_response_times = [] 
        self._current_speech_start_time = None 
        self._last_audio_chunk_time = None  
        self._audio_chunk_count = 0 


    @property
    def vad_enabled(self):
        return self._settings["vad_events"]
    
    def _setup_sibling_deepgram(self):
        self._sibling_deepgram = None

        if self.language != 'it': 
            return 

        self._sibling_deepgram = DeepgramSiDetector(self.api_key, self.sibling_transcript_handler)

    def _setup_complementary_gladia(self):
        """
        Set up complementary Gladia service for enhanced Spanish transcription accuracy.
        """
        self._complementary_gladia = None

        # Only enable for Spanish or when explicitly requested
        if not self._use_gladia_for_finals:
            logger.debug("🎯 Gladia complementary service disabled")
            return
        
        if not self._gladia_api_key:
            logger.warning("⚠️ Gladia API key not provided, complementary service disabled")
            return

        try:
            # Determine language for Gladia
            gladia_language = Language.ES  # Default to Spanish
            if isinstance(self.language, str):
                if self.language.startswith('es'):
                    gladia_language = Language.ES
                elif self.language.startswith('en'):
                    gladia_language = Language.EN
                elif self.language.startswith('fr'):
                    gladia_language = Language.FR
                elif self.language.startswith('de'):
                    gladia_language = Language.DE
                elif self.language.startswith('it'):
                    gladia_language = Language.IT
                elif self.language.startswith('pt'):
                    gladia_language = Language.PT
            
            self._complementary_gladia = DeepgramGladiaDetector(
                api_key=self._gladia_api_key,
                callback=self.gladia_transcript_handler,
                language=gladia_language,
                sample_rate=self.sample_rate or 16000,
                timeout_seconds=self._gladia_timeout_seconds
            )
            
            logger.info(f"✅ Gladia complementary service initialized for {gladia_language}")
            
        except Exception as e:
            logger.exception(f"❌ Failed to initialize Gladia complementary service: {e}")
            self._complementary_gladia = None

    async def sibling_transcript_handler(self, result: LiveResultResponse):
        result_time = result.start

        if abs(self._last_time_transcription - result_time) < 0.5:
            logger.debug("Ignoring 'Si' because recent transcript")
            return
        
        await self._on_message(result=result)

    async def gladia_transcript_handler(self, transcript: str, language: str, confidence: float):
        """
        Handle high-quality final transcripts from Gladia.
        Create a synthetic LiveResultResponse to integrate with existing Deepgram logic.
        """
        try:
            current_time = time.time()
            start_time = current_time
            
            # Track Gladia response performance
            if self._current_speech_start_time is not None:
                elapsed = current_time - self._current_speech_start_time
                self._gladia_response_times.append(elapsed)
                logger.debug(f"🎯 Gladia Response Time: {round(elapsed, 3)}s")
            
            logger.info(f"🚀 Gladia Enhanced Final: '{transcript}' (confidence: {confidence:.2f}, lang: {language})")
            
            # Set flag to ignore upcoming Deepgram finals for a short period
            self._ignore_deepgram_finals = True
            asyncio.create_task(self._reset_ignore_flag(delay=0.5))  # Reset after 500ms
            
            # Create synthetic LiveResultResponse compatible with existing Deepgram logic
            from deepgram import LiveResultResponse
            
            # Create a mock result that looks like Deepgram but with Gladia's data
            mock_alternative = type('Alternative', (), {
                'transcript': transcript,
                'confidence': confidence,
                'words': [],
                'languages': [language] if language else []
            })()
            
            mock_channel = type('Channel', (), {
                'alternatives': [mock_alternative]
            })()
            
            mock_result = type('LiveResultResponse', (), {
                'channel': mock_channel,
                'is_final': True,
                'speech_final': True,
                'start': start_time,
                'duration': 0.0
            })()
            
            # Process through existing Deepgram pipeline with Gladia enhancement marker
            logger.debug(f"🔄 Processing Gladia transcript through Deepgram pipeline")
            await self._on_message(result=mock_result, is_gladia_enhanced=True)
            
        except Exception as e:
            logger.exception(f"❌ Error in gladia_transcript_handler: {e}")

    async def _reset_ignore_flag(self, delay: float = 0.5):
        """Reset the ignore Deepgram finals flag after a delay."""
        await asyncio.sleep(delay)
        self._ignore_deepgram_finals = False
        logger.debug("🔄 Reset ignore Deepgram finals flag")

    async def _handle_deepgram_final_with_timeout(self, transcript: str, language: Language, speech_final: bool, start_time: float) -> bool:
        """
        Handle Deepgram final transcripts with timeout for Gladia response.
        Returns True if we should wait for Gladia, False if we should process Deepgram final.
        """
        try:
            if self._ignore_deepgram_finals:
                logger.debug(f"🚫 Ignoring Deepgram final (Gladia enhanced mode): '{transcript}'")
                return True
            
            # Store the Deepgram final with timeout
            timeout_id = f"deepgram_{start_time}_{transcript[:20]}"
            self._pending_deepgram_finals[timeout_id] = {
                'transcript': transcript,
                'language': language,
                'speech_final': speech_final,
                'timestamp': time.time()
            }
            
            logger.debug(f"⏰ Storing Deepgram final with timeout: '{transcript}' (ID: {timeout_id})")
            
            # Set timeout for Gladia response
            asyncio.create_task(self._timeout_deepgram_final(timeout_id, self._gladia_timeout_seconds))
            
            return True  # Wait for Gladia
            
        except Exception as e:
            logger.exception(f"❌ Error in _handle_deepgram_final_with_timeout: {e}")
            return False  # Fallback to processing Deepgram final

    async def _timeout_deepgram_final(self, timeout_id: str, delay: float):
        """
        Process pending Deepgram final if Gladia doesn't respond within timeout.
        """
        try:
            await asyncio.sleep(delay)
            
            # Check if the final is still pending (not processed by Gladia)
            if timeout_id in self._pending_deepgram_finals:
                pending_final = self._pending_deepgram_finals.pop(timeout_id)
                logger.warning(f"⏰ Gladia timeout! Processing Deepgram final: '{pending_final['transcript']}'")
                
                # Process the Deepgram final as fallback
                await self._on_final_transcript_message(
                    pending_final['transcript'],
                    pending_final['language'], 
                    pending_final['speech_final']
                )
                
        except Exception as e:
            logger.exception(f"❌ Error in _timeout_deepgram_final: {e}")

    def get_gladia_stats(self) -> Dict:
        """Get comprehensive Gladia performance statistics."""
        if not self._gladia_response_times:
            return {
                "count": 0,
                "average": 0.0,
                "min": 0.0,
                "max": 0.0,
                "latest": 0.0,
                "enabled": self._use_gladia_for_finals
            }
        
        return {
            "count": len(self._gladia_response_times),
            "average": round(sum(self._gladia_response_times) / len(self._gladia_response_times), 3),
            "min": round(min(self._gladia_response_times), 3),
            "max": round(max(self._gladia_response_times), 3),
            "latest": round(self._gladia_response_times[-1], 3) if self._gladia_response_times else 0.0,
            "all_times": [round(t, 3) for t in self._gladia_response_times],
            "enabled": self._use_gladia_for_finals
        }

    def log_gladia_performance(self):
        """Log Gladia performance statistics."""
        stats = self.get_gladia_stats()
        if stats["enabled"]:
            logger.info(f"🎯 Gladia Enhanced Transcription Performance:")
            if stats["count"] > 0:
                logger.info(f"   📊 Total enhanced responses: {stats['count']}")
                logger.info(f"   ⏱️  Average time: {stats['average']}s")
                logger.info(f"   🏃 Fastest: {stats['min']}s") 
                logger.info(f"   🐌 Slowest: {stats['max']}s")
                logger.info(f"   🕐 Latest: {stats['latest']}s")
            else:
                logger.info(f"   📊 No enhanced responses received yet")
        else:
            logger.info(f"🎯 Gladia Enhanced Transcription: Disabled")

    def log_stt_performance(self):
        """Log STT performance statistics."""
        stats = self.get_stt_stats()
        if stats["count"] > 0:
            logger.info(f"🎯 Deepgram STT Performance Summary:")
            logger.info(f"   📊 Total responses: {stats['count']}")
            logger.info(f"   ⏱️  Average time: {stats['average']}s")
            logger.info(f"   🏃 Fastest: {stats['min']}s")
            logger.info(f"   🐌 Slowest: {stats['max']}s")
            logger.info(f"   🕐 Latest: {stats['latest']}s")
            logger.info(f"   📈 All times: {stats['all_times']}")
        
        # Also log Gladia performance if enabled
        if self._use_gladia_for_finals:
            self.log_gladia_performance()

    def log_combined_performance(self):
        """Log combined performance statistics for both Deepgram and Gladia."""
        logger.info("=" * 60)
        logger.info("🎭 COMBINED STT PERFORMANCE REPORT")
        logger.info("=" * 60)
        
        # Deepgram stats
        deepgram_stats = self.get_stt_stats()
        logger.info(f"🎤 DEEPGRAM PERFORMANCE:")
        if deepgram_stats["count"] > 0:
            logger.info(f"   📊 Responses: {deepgram_stats['count']}")
            logger.info(f"   ⏱️  Avg Time: {deepgram_stats['average']}s")
            logger.info(f"   🏃 Best: {deepgram_stats['min']}s | 🐌 Worst: {deepgram_stats['max']}s")
        else:
            logger.info(f"   📊 No responses recorded")
        
        # Gladia stats
        if self._use_gladia_for_finals:
            gladia_stats = self.get_gladia_stats()
            logger.info(f"🎯 GLADIA ENHANCED PERFORMANCE:")
            if gladia_stats["count"] > 0:
                logger.info(f"   📊 Enhanced Responses: {gladia_stats['count']}")
                logger.info(f"   ⏱️  Avg Time: {gladia_stats['average']}s")
                logger.info(f"   🏃 Best: {gladia_stats['min']}s | 🐌 Worst: {gladia_stats['max']}s")
                
                # Calculate accuracy improvement metrics
                if deepgram_stats["count"] > 0:
                    time_diff = gladia_stats["average"] - deepgram_stats["average"]
                    improvement = "faster" if time_diff < 0 else "slower"
                    logger.info(f"   📈 vs Deepgram: {abs(time_diff):.3f}s {improvement}")
            else:
                logger.info(f"   📊 No enhanced responses recorded")
        else:
            logger.info(f"🎯 GLADIA ENHANCED: Disabled")
        
        logger.info("=" * 60)

    async def set_model(self, model: str):
        try:
            await super().set_model(model)
            logger.info(f"Switching STT model to: [{model}]")
            self._settings["model"] = model
            await self._disconnect()
            await self._connect()
        except Exception as e:
            logger.exception(f"{self} exception in set_model: {e}")
            raise

    async def set_language(self, language: Language):
        try:
            logger.info(f"Switching STT language to: [{language}]")
            self._settings["language"] = language
            await self._disconnect()
            await self._connect()
        except Exception as e:
            logger.exception(f"{self} exception in set_language: {e}")
            raise

    async def start(self, frame: StartFrame):
        try:
            await super().start(frame)
            self._settings["sample_rate"] = self.sample_rate
            await self._connect()

            if self._sibling_deepgram:
                await self._sibling_deepgram.start()
                
            if self._complementary_gladia:
                logger.info("🚀 Starting Gladia complementary service")
                await self._complementary_gladia.start()
        except Exception as e:
            logger.exception(f"{self} exception in start: {e}")
            raise

    async def stop(self, frame: EndFrame):
        try:
            await super().stop(frame)
            await self._disconnect()

            if self._sibling_deepgram:
                await self._sibling_deepgram.stop()
                
            if self._complementary_gladia:
                logger.info("🛑 Stopping Gladia complementary service")
                await self._complementary_gladia.stop()
        except Exception as e:
            logger.exception(f"{self} exception in stop: {e}")
            raise

    async def cancel(self, frame: CancelFrame):
        try:
            await super().cancel(frame)
            await self._disconnect()
            
            if self._complementary_gladia:
                logger.info("🛑 Cancelling Gladia complementary service")
                await self._complementary_gladia.stop()
        except Exception as e:
            logger.exception(f"{self} exception in cancel: {e}")
            raise

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        try:
            # Enhanced timing tracking
            current_time = time.perf_counter()
            self._last_audio_chunk_time = current_time
            self._audio_chunk_count += 1
            
            # Start timing when we receive first audio after speech detection
            if self._current_speech_start_time is None:
                self._current_speech_start_time = current_time
                logger.debug(f"🎤 Deepgram: Starting speech detection timer at chunk #{self._audio_chunk_count}")
            
            # Send audio to Deepgram
            await self._connection.send(audio)
            
            # Send audio to sibling Deepgram (Si detector)
            if self._sibling_deepgram:
                await self._sibling_deepgram.send_audio(audio)
                
            # Send audio to complementary Gladia for enhanced accuracy
            if self._complementary_gladia:
                await self._complementary_gladia.send_audio(audio)
                
            yield None
        except Exception as e:
            logger.exception(f"{self} exception in run_stt: {e}")
            yield ErrorFrame(f"run_stt error: {e}")

    async def _connect(self):
        try:
            logger.debug("Connecting to Deepgram")

            self._connection: AsyncListenWebSocketClient = self._client.listen.asyncwebsocket.v("1")

            self._connection.on(
                LiveTranscriptionEvents(LiveTranscriptionEvents.Transcript), self._on_message
            )
            self._connection.on(LiveTranscriptionEvents(LiveTranscriptionEvents.Error), self._on_error)

            if not self._async_handler_task:
                self._async_handler_task = self.create_monitored_task(self._async_handler)

            if self.vad_enabled:
                self._connection.on(
                    LiveTranscriptionEvents(LiveTranscriptionEvents.SpeechStarted),
                    self._on_speech_started,
                )
                self._connection.on(
                    LiveTranscriptionEvents(LiveTranscriptionEvents.UtteranceEnd),
                    self._on_utterance_end,
                )

            if not await self._connection.start(options=self._settings, addons=self._addons):
                logger.error(f"{self}: unable to connect to Deepgram")
            else:
                logger.debug(f"Connected to Deepgram")
        except Exception as e:
            logger.exception(f"{self} exception in _connect: {e}")
            raise

    async def _disconnect(self):
        try:
            if self._async_handler_task:
                await self.cancel_task(self._async_handler_task)

            if self._connection.is_connected:
                logger.debug("Disconnecting from Deepgram")
                try:
                    await asyncio.wait_for(self._connection.finish(), timeout=0.1)
                    logger.debug("Safe disconnect from Deepgram")
                except asyncio.TimeoutError:
                    logger.warning("Timeout while disconnecting from Deepgram.")
        except Exception as e:
            logger.exception(f"{self} exception in _disconnect: {e}")

    async def start_metrics(self):
        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        # Start timing for STT response
        self._current_request_start_time = time.perf_counter()

    async def _on_error(self, *args, **kwargs):
        error: ErrorResponse = kwargs["error"]
        logger.warning(f"{self} connection error, will retry: {error}")
        await self.stop_all_metrics()
        # NOTE(aleix): we don't disconnect (i.e. call finish on the connection)
        # because this triggers more errors internally in the Deepgram SDK. So,
        # we just forget about the previous connection and create a new one.
        await self._connect()

    async def _on_speech_started(self, *args, **kwargs):
        await self.start_metrics()
        await self._call_event_handler("on_speech_started", *args, **kwargs)

    async def _on_utterance_end(self, *args, **kwargs):
        await self._call_event_handler("on_utterance_end", *args, **kwargs)


    async def _handle_user_speaking(self):

        await self.push_frame(StartInterruptionFrame())
        if self._user_speaking == True: return

        self._user_speaking = True

        await self.push_frame(UserStartedSpeakingFrame())

    async def _handle_user_silence(self):

        if self._user_speaking == False: return

        self._user_speaking = False
        await self.push_frame(UserStoppedSpeakingFrame())

    async def _handle_bot_speaking(self):
        self._bot_speaking = True


    async def _handle_bot_silence(self):
        self._bot_speaking = False


    def _transcript_words_count(self, transcript: str):
        return len(transcript.split(" "))

    async def _async_handle_accum_transcription(self, current_time):

        if current_time - self._last_time_accum_transcription > self._on_no_punctuation_seconds and len(self._accum_transcription_frames):
            logger.debug("Sending accum transcription because of timeout")
            await self._send_accum_transcriptions()

    async def _handle_false_interim(self, current_time):

        if not self._user_speaking: return
        if not self._last_interim_time: return
        if self._vad_active: return

        last_interim_delay = current_time - self._last_interim_time

        if last_interim_delay > FALSE_INTERIM_SECONDS: return

        logger.debug("False interim detected")

        await self._handle_user_silence()


    
    async def _async_handler(self, task_name):

        while True:
            if not self.is_monitored_task_active(task_name): return

            await asyncio.sleep(0.1)
            
            current_time = time.time()

            await self._async_handle_accum_transcription(current_time)
            await self._handle_false_interim(current_time)



            
    
    async def _send_accum_transcriptions(self):

        if not len(self._accum_transcription_frames): return

        await self._handle_user_speaking()

        for frame in self._accum_transcription_frames:
            await self.push_frame(
                frame
            )
        self._accum_transcription_frames = []

        await self._handle_user_silence()
        await self.stop_processing_metrics()

    def _is_accum_transcription(self, text: str):

        END_OF_PHRASE_CHARACTERS = ['.', '?']

        text =  text.strip()

        if not text: return True

        return not text[-1] in END_OF_PHRASE_CHARACTERS
    
    def _append_accum_transcription(self, frame: TranscriptionFrame):
        self._last_time_accum_transcription = time.time()
        self._accum_transcription_frames.append(frame)


    def _handle_first_message(self, text):

        if self._first_message: return 

        self._first_message = text
        self._first_message_time = time.time()

    def _should_ignore_first_repeated_message(self, text):

        if not self._first_message: return
        
        time_since_first_message = time.time() - self._first_message_time
        if time_since_first_message > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        return is_equivalent_basic(text, self._first_message)



    async def _on_final_transcript_message(self, transcript, language, speech_final: bool):

        await self._handle_user_speaking()
        frame = TranscriptionFrame(transcript, "", time_now_iso8601(), language)

        self._handle_first_message(frame.text)
        self._append_accum_transcription(frame)
        self._was_first_transcript_receipt = True
        if not self._is_accum_transcription(frame.text) or speech_final:
            await self._send_accum_transcriptions()
    
    async def _on_interim_transcript_message(self, transcript, language, start_time):
        
        self._last_interim_time = time.time()
        await self._handle_user_speaking()
        await self.push_frame(
            InterimTranscriptionFrame(transcript, "", time_now_iso8601(), language)
        )

    async def _should_ignore_transcription(self, result: LiveResultResponse, is_gladia_enhanced: bool = False):
        """
        Determine if a transcription should be ignored.
        Enhanced logic for Gladia vs Deepgram transcripts.
        """
        is_final = result.is_final
        confidence = result.channel.alternatives[0].confidence
        transcript = result.channel.alternatives[0].transcript
        time_start = result.channel.alternatives[0].words[0].start if result.channel.alternatives[0].words else 0
        
        # Gladia enhanced transcripts have higher priority and different thresholds
        if is_gladia_enhanced:
            logger.debug(f"🎯 Processing Gladia enhanced transcript: '{transcript}' (confidence: {confidence:.2f})")
            
            # Lower confidence threshold for Gladia (it's generally more accurate)
            if confidence < 0.5:
                logger.debug("🚫 Ignoring Gladia transcript - confidence too low")
                return True
                
            # Gladia finals are always processed unless clearly wrong
            if is_final:
                return False
        
        # Standard Deepgram logic
        if not is_final and confidence < 0.7:
            logger.debug("Ignoring interim because low confidence")
            return True

        if time_start < 1 and self._transcript_words_count(transcript) == 1:
            logger.debug("Ignoring first message, fast greeting")
            return True
        
        if self._should_ignore_first_repeated_message(transcript):
            logger.debug("Ignoring repeated first message")
            return True
        
        if not self._vad_active and not is_final:
            logger.debug("Ignoring Deepgram interruption because VAD inactive")
            return True
        
        logger.debug("Bot speaking: " + str(self._bot_speaking ) + " ** allow_interruptions: " + str(self._allow_stt_interruptions))
        if self._bot_speaking and not self._allow_stt_interruptions:
            # Gladia enhanced transcripts can interrupt more easily (higher accuracy)
            if not is_gladia_enhanced:
                logger.debug("Ignoring Deepgram interruption because allow_interruptions is False")
                return True
        
        if self._bot_speaking and self._transcript_words_count(transcript) == 1 and not is_gladia_enhanced: 
            logger.debug(f"Ignoring Deepgram interruption because bot is speaking: {transcript}")
            return True

        return False
    

    async def _detect_and_handle_voicemail(self, transcript: str):

        if not self.detect_voicemail: return False

        logger.debug(transcript)
        logger.debug(self._time_since_init())
        
        if self._time_since_init() > VOICEMAIL_DETECTION_SECONDS and self._was_first_transcript_receipt: return False
        
        if not voicemail.is_text_voicemail(transcript): return False
        
        logger.debug("Voicemail detected")

        await self.push_frame(
            VoicemailFrame(transcript)
        )

        logger.debug("Voicemail pushed")
        return True



    async def _on_message(self, *args, **kwargs):
        if not self._restarted: return
        try:
            result: LiveResultResponse = kwargs["result"]
            is_gladia_enhanced = kwargs.get("is_gladia_enhanced", False)
            
            logger.debug(result)
            
            if len(result.channel.alternatives) == 0:
                return
            
            is_final = result.is_final
            speech_final = result.speech_final
            transcript = result.channel.alternatives[0].transcript
            confidence = result.channel.alternatives[0].confidence
            start_time = result.start
            
            language = None
            if result.channel.alternatives[0].languages:
                language = result.channel.alternatives[0].languages[0]
                language = Language(language)
            
            if len(transcript) > 0:
                await self.stop_ttfb_metrics()

                if await self._detect_and_handle_voicemail(transcript):
                    return 
                
                # Enhanced logging for Gladia vs Deepgram
                source = "🎯 Gladia Enhanced" if is_gladia_enhanced else "🎤 Deepgram"
                logger.debug(f"{source} Transcription{'' if is_final else ' interim'}: {transcript}")
                logger.debug(f"{source} Confidence: {confidence}")
                
                if await self._should_ignore_transcription(result, is_gladia_enhanced):
                    return
                
                if is_final:
                    # Handle timeout logic for Gladia vs Deepgram finals
                    if self._use_gladia_for_finals and not is_gladia_enhanced:
                        # This is a Deepgram final, check if we should wait for Gladia
                        if await self._handle_deepgram_final_with_timeout(transcript, language, speech_final, start_time):
                            return  # Waiting for Gladia, don't process Deepgram final yet
                    
                    if self._current_speech_start_time is not None:
                        elapsed = time.perf_counter() - self._current_speech_start_time
                        elapsed_formatted = round(elapsed, 3)
                        
                        if is_gladia_enhanced:
                            self._gladia_response_times.append(elapsed_formatted)
                            logger.info(f"🎯 Gladia Enhanced Response Time: {elapsed_formatted}s")
                        else:
                            self._stt_response_times.append(elapsed_formatted)
                            logger.debug(f"📊 Deepgram STT Response Time: {elapsed_formatted}s")
                        
                        logger.debug(f"   📝 Transcript: '{transcript}'")
                        logger.debug(f"   🎯 Confidence: {confidence:.2f}")
                        logger.debug(f"   📦 Audio chunks processed: {self._audio_chunk_count}")
                        
                        if not is_gladia_enhanced:  # Only reset for Deepgram, Gladia might come later
                            self._current_speech_start_time = None
                            self._audio_chunk_count = 0
                    
                    await self._on_final_transcript_message(transcript, language, speech_final)
                    self._last_time_transcription = start_time
                else:
                    await self._on_interim_transcript_message(transcript, language, start_time)

        except Exception as e:
            logger.exception(f"{self} unexpected error in _on_message: {e}")


    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("Received bot started speaking on deepgram")
            await self._handle_bot_speaking()

        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("Received bot stopped speaking on deepgram")
            await self._handle_bot_silence() 

        if isinstance(frame, STTRestartFrame):
            logger.debug("Received STT Restart Frame")
            self._restarted = True
            await self._disconnect()
            await self._connect()
            return

        if isinstance(frame, UserStartedSpeakingFrame) and not self.vad_enabled:
            await self.start_metrics()
            self._current_speech_start_time = time.perf_counter()
            self._audio_chunk_count = 0
            logger.debug(f"🎤 Deepgram: User started speaking - resetting timer")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            # https://developers.deepgram.com/docs/finalize
            await self._connection.finalize()
            logger.trace(f"Triggered finalize event on: {frame.name}, {direction}")
        
        if isinstance(frame, VADInactiveFrame):
            self._vad_active = False
            if self._connection and self._connection.is_connected:
                await self._connection.finalize()  
        elif isinstance(frame, VADActiveFrame):
            self._vad_active = True