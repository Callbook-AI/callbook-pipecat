#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import base64
import json
import re
import time
import asyncio
from typing import AsyncGenerator, Dict, List, Optional, Callable


import aiohttp
from loguru import logger
import websockets

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
            "language":       "es",          # Spanish
            "model":          "general",
            "sample_rate":    sample_rate,
            "interim_results": False,        # only finals
            "punctuate":      True,
        }

        self._conn = None  # will hold the websocket client
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

def language_to_gladia_language(language: Language) -> str:
    """Convert Pipecat Language to Gladia language code."""
    lang_map = {
        Language.ES: "es",
        Language.EN: "en", 
        Language.FR: "fr",
        Language.DE: "de",
        Language.IT: "it",
        Language.PT: "pt",
        Language.CA: "ca",
    }
    if isinstance(language, Language):
        return lang_map.get(language, "es")
    elif isinstance(language, str):
        return language if language in lang_map.values() else "es"
    return "es"

class DeepgramGladiaDetector:
    """
    High-quality STT enhancement using Gladia's Solaria-1 model.
    Runs parallel to Deepgram, providing superior final transcriptions
    while maintaining Deepgram's speed for interim results.
    """
    def __init__(
        self,
        api_key: str,
        callback: Callable[[str, float, float], None],  # transcript, confidence, timestamp
        language: Language = Language.ES,
        url: str = "https://api.gladia.io/v2/live",
        sample_rate: int = 16_000,
        confidence: float = 0.6,
        endpointing: float = 0.4,
        speech_threshold: float = 0.6,
        timeout_seconds: float = 1.5,  # Timeout to fallback to Deepgram
    ):
        self._api_key = api_key
        self._callback = callback
        self._language = language
        self._url = url
        self._sample_rate = sample_rate
        self._confidence = confidence
        self._timeout_seconds = timeout_seconds
        
        # WebSocket connection
        self._websocket = None
        self._receive_task = None
        
        # Timing and deduplication
        self.processed_transcripts = set()
        self._last_transcript_time = 0
        self._start_time = time.time()
        
        # Gladia configuration optimized for Spanish accuracy
        self._settings = {
            "encoding": "wav/pcm",
            "bit_depth": 16,
            "sample_rate": sample_rate,
            "channels": 1,
            "model": "solaria-1",  # Best model for Spanish
            "endpointing": endpointing,
            "maximum_duration_without_endpointing": 6,
            "language_config": {
                "languages": [language_to_gladia_language(language)],
                "code_switching": False,
            },
            "pre_processing": {
                "audio_enhancer": False,  # Faster processing
                "speech_threshold": speech_threshold,
            },
            "realtime_processing": {
                "words_accurate_timestamps": False,
            },
            "messages_config": {
                "receive_final_transcripts": True,
                "receive_speech_events": False,
                "receive_pre_processing_events": False,
                "receive_realtime_processing_events": False,
                "receive_post_processing_events": False,
                "receive_acknowledgments": False,
                "receive_errors": True,
                "receive_lifecycle_events": False
            }
        }

    async def start(self):
        """Initialize Gladia connection for high-quality transcription."""
        try:
            logger.info("🎯 DeepgramGladiaDetector: Starting enhanced transcription service")
            response = await self._setup_gladia()
            self._websocket = await websockets.connect(response["url"])
            self._receive_task = asyncio.create_task(self._receive_task_handler())
            logger.info("✅ DeepgramGladiaDetector: Enhanced transcription ready")
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector failed to start: {e}")
            raise

    async def send_audio(self, chunk: bytes):
        """Send audio chunk to Gladia for enhanced transcription."""
        try:
            if not self._websocket:
                logger.warning("🎯 DeepgramGladiaDetector: WebSocket not available, skipping audio chunk")
                return
                
            chunk_size = len(chunk)
            logger.trace(f"🎯 DeepgramGladiaDetector: Sending audio chunk ({chunk_size} bytes)")
            
            data = base64.b64encode(chunk).decode("utf-8")
            message = {"type": "audio_chunk", "data": {"chunk": data}}
            await self._websocket.send(json.dumps(message))
            
        except Exception as e:
            logger.debug(f"🔧 DeepgramGladiaDetector audio send error: {e}")

    async def stop(self):
        """Gracefully stop the enhanced transcription service."""
        try:
            logger.info("🛑 DeepgramGladiaDetector: Stopping enhanced transcription")
            
            if self._receive_task:
                self._receive_task.cancel()
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    pass
            
            if self._websocket:
                await self._websocket.send(json.dumps({"type": "stop_recording"}))
                await self._websocket.close()
                
            logger.info("✅ DeepgramGladiaDetector: Enhanced transcription stopped")
            
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector stop error: {e}")

    async def _setup_gladia(self):
        """Setup Gladia session for enhanced transcription."""
        async with aiohttp.ClientSession() as session:
            logger.debug("🔧 DeepgramGladiaDetector: Configuring enhanced transcription")
            logger.debug(f"🎯 Gladia settings: {self._settings}")
            
            async with session.post(
                self._url,
                headers={"X-Gladia-Key": self._api_key, "Content-Type": "application/json"},
                json=self._settings,
            ) as response:
                if response.ok:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise Exception(f"Gladia session failed: {response.status} - {error_text}")

    async def _receive_task_handler(self):
        """Handle enhanced transcription results from Gladia."""
        try:
            logger.debug("📡 DeepgramGladiaDetector: Listening for enhanced transcriptions")
            
            async for message in self._websocket:
                try:
                    content = json.loads(message)
                    
                    if content["type"] != "transcript":
                        logger.trace(f"🎯 DeepgramGladiaDetector: Ignoring non-transcript message: {content['type']}")
                        continue
                        
                    utterance = content["data"]["utterance"]
                    confidence = utterance.get("confidence", 0)
                    transcript = utterance["text"].strip()
                    is_final = content["data"]["is_final"]
                    
                    logger.debug(f"🎯 DeepgramGladiaDetector: Raw message - Final: {is_final}, Text: '{transcript}', Confidence: {confidence:.2f}")
                    
                    # Only process high-quality final transcripts
                    if not is_final:
                        logger.trace(f"🎯 DeepgramGladiaDetector: Skipping interim result: '{transcript}'")
                        continue
                    
                    if not transcript:
                        logger.trace(f"🎯 DeepgramGladiaDetector: Skipping empty transcript")
                        continue
                        
                    if confidence < self._confidence:
                        logger.debug(f"🎯 DeepgramGladiaDetector: Skipping low confidence transcript: '{transcript}' (conf: {confidence:.2f} < {self._confidence})")
                        continue
                    
                    # Enhanced deduplication
                    current_time = time.time()
                    transcript_key = f"{transcript.lower()}_{confidence:.2f}"
                    
                    if transcript_key in self.processed_transcripts:
                        logger.debug(f"🔄 DeepgramGladiaDetector: Duplicate ignored: '{transcript}' (key: {transcript_key})")
                        continue
                        
                    self.processed_transcripts.add(transcript_key)
                    logger.debug(f"🎯 DeepgramGladiaDetector: Added to processed set: {transcript_key}")
                    
                    # Clean old entries to prevent memory growth
                    if len(self.processed_transcripts) > 100:
                        logger.debug(f"🎯 DeepgramGladiaDetector: Cleaning processed transcripts cache (size: {len(self.processed_transcripts)})")
                        self.processed_transcripts.clear()
                    
                    self._last_transcript_time = current_time
                    
                    logger.info(f"🎯 DeepgramGladiaDetector: ✅ Enhanced transcript ACCEPTED: '{transcript}' (confidence: {confidence:.2f})")
                    
                    # Call the callback with enhanced transcript
                    logger.debug(f"🎯 DeepgramGladiaDetector: Calling callback for: '{transcript}'")
                    await self._callback(transcript, confidence, current_time)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ DeepgramGladiaDetector: Invalid JSON: {e}")
                except Exception as e:
                    logger.exception(f"❌ DeepgramGladiaDetector message error: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("🔌 DeepgramGladiaDetector: Connection closed")
        except Exception as e:
            logger.exception(f"❌ DeepgramGladiaDetector receive error: {e}")


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
        gladia_api_key: Optional[str] = None,  # NEW: Gladia API key for enhanced accuracy
        gladia_timeout: float = 1.5,
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
        
        self.language = merged_options.language
        self.api_key = api_key
        self.detect_voicemail = detect_voicemail  
        self._allow_stt_interruptions = allow_interruptions
        logger.debug(f"Allow ** interruptions: {self._allow_stt_interruptions}")

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


        self._setup_sibling_deepgram()


        self._client = DeepgramClient(
            api_key,
            config=DeepgramClientOptions(
                url=url,
                options={"keepalive": "true"},  # verbose=logging.DEBUG
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
        
        # Enhanced response time tracking
        self._stt_response_times = []  # List to store STT response durations
        self._current_speech_start_time = None  # Track when speech detection starts
        self._last_audio_chunk_time = None  # Track last audio chunk received
        self._audio_chunk_count = 0  # Count audio chunks for debugging


        self._gladia_api_key = gladia_api_key
        self._gladia_timeout = gladia_timeout
        self._pending_deepgram_finals = {}  # Store Deepgram finals waiting for Gladia
        self._ignore_deepgram_finals = False  # Flag to ignore Deepgram when Gladia is used
        
        self._setup_complementary_gladia()

    def _setup_complementary_gladia(self):
        """Setup Gladia service for enhanced Spanish transcription accuracy."""
        self._complementary_gladia = None

        if not self._gladia_api_key:
            logger.info("🔧 DeepgramSTTService: No Gladia API key provided, using Deepgram only")
            return

        # Only activate for languages that benefit from Gladia's enhanced accuracy
        enhanced_languages = ['es', 'en', 'fr', 'pt', 'ca']
        current_lang = self.language.lower() if isinstance(self.language, str) else str(self.language).lower()
        
        logger.debug(f"🔧 DeepgramSTTService: Checking language '{current_lang}' for Gladia enhancement")
        
        if not any(lang in current_lang for lang in enhanced_languages):
            logger.info(f"🔧 DeepgramSTTService: Language '{current_lang}' doesn't benefit from Gladia enhancement")
            return

        try:
            # Convert language for Gladia
            if isinstance(self.language, str):
                if self.language.lower() == 'es':
                    gladia_language = Language.ES
                elif self.language.lower() == 'en':
                    gladia_language = Language.EN
                else:
                    gladia_language = Language.ES  # Default fallback
            else:
                gladia_language = self.language

            logger.info(f"🔧 DeepgramSTTService: Setting up Gladia enhancement for {gladia_language}")

            self._complementary_gladia = DeepgramGladiaDetector(
                api_key=self._gladia_api_key,
                callback=self.gladia_transcript_handler,
                language=gladia_language,
                sample_rate=self.sample_rate or 16000,
                confidence=0.6,
                endpointing=0.4,
                speech_threshold=0.6,
                timeout_seconds=self._gladia_timeout
            )
            
            logger.info(f"🎯 DeepgramSTTService: ✅ Enhanced transcription enabled for {gladia_language} (timeout: {self._gladia_timeout}s)")
            
        except Exception as e:
            logger.exception(f"❌ DeepgramSTTService: Gladia setup failed: {e}")
            self._complementary_gladia = None

    async def gladia_transcript_handler(self, transcript: str, confidence: float, timestamp: float):
        """Handle enhanced transcription from Gladia."""
        try:
            logger.info(f"🎯 DeepgramSTTService: ⬇️ Gladia callback received: '{transcript}' (confidence: {confidence:.2f}, timestamp: {timestamp})")
            
            # Create a mock Deepgram result for compatibility
            mock_result = type('MockResult', (), {
                'is_final': True,
                'speech_final': True,
                'start': timestamp,
                'channel': type('Channel', (), {
                    'alternatives': [type('Alternative', (), {
                        'transcript': transcript,
                        'confidence': confidence,
                        'words': [],
                        'languages': [str(self.language)] if hasattr(self, 'language') else ['es']
                    })()]
                })()
            })()
            
            logger.debug(f"🎯 DeepgramSTTService: Created mock Deepgram result for Gladia transcript")
            
            # Process through existing Deepgram logic but mark as enhanced
            logger.debug(f"🎯 DeepgramSTTService: Setting ignore_deepgram_finals=True")
            self._ignore_deepgram_finals = True
            
            logger.debug(f"🎯 DeepgramSTTService: Calling _on_message with enhanced=True")
            await self._on_message(result=mock_result, enhanced=True)
            
            # Reset flag after processing
            await asyncio.sleep(0.1)
            logger.debug(f"🎯 DeepgramSTTService: Resetting ignore_deepgram_finals=False")
            self._ignore_deepgram_finals = False
            
        except Exception as e:
            logger.exception(f"❌ DeepgramSTTService: Gladia handler error: {e}")

    @property
    def vad_enabled(self):
        return self._settings["vad_events"]
    
    def _setup_sibling_deepgram(self):

        self._sibling_deepgram = None

        if self.language != 'it': return 

        self._sibling_deepgram = DeepgramSiDetector(self.api_key, self.sibling_transcript_handler)


    async def sibling_transcript_handler(self, result: LiveResultResponse):

        result_time = result.start

        if abs(self._last_time_transcription - result_time) < 0.5:
            logger.debug("Ignoring 'Si' because recent transcript")
            return
        
        await self._on_message(result=result)


    def _time_since_init(self):
        return time.time() - self.start_time

    def can_generate_metrics(self) -> bool:
        return True

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
    
    def get_stt_stats(self) -> Dict:
        """Get comprehensive STT performance statistics."""
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
            "all_times": [round(t, 3) for t in self._stt_response_times]
        }
    
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
                logger.debug("🔧 DeepgramSTTService: Sibling Deepgram started")
                
            if self._complementary_gladia:
                logger.info("🎯 DeepgramSTTService: Starting enhanced Gladia transcription...")
                await self._complementary_gladia.start()
                logger.info("🎯 DeepgramSTTService: ✅ Enhanced transcription started and ready")
        except Exception as e:
            logger.exception(f"{self} exception in start: {e}")
            raise       

    async def stop(self, frame: EndFrame):
        try:
            await super().stop(frame)
            await self._disconnect()

            if self._sibling_deepgram:
                await self._sibling_deepgram.stop()
                logger.debug("🔧 DeepgramSTTService: Sibling Deepgram stopped")
                
            if self._complementary_gladia:
                logger.info("🎯 DeepgramSTTService: Stopping enhanced Gladia transcription...")
                await self._complementary_gladia.stop()
                logger.info("🎯 DeepgramSTTService: ✅ Enhanced transcription stopped")
                
        except Exception as e:
            logger.exception(f"{self} exception in stop: {e}")
            raise

    async def cancel(self, frame: CancelFrame):

        try:
            await super().cancel(frame)
            await self._disconnect()
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
                logger.debug(f"🎤 DeepgramSTTService: ⏱️ Starting speech detection timer at chunk #{self._audio_chunk_count}")
            
            # Send audio to primary Deepgram service
            logger.trace(f"⚡ DeepgramSTTService: Sending audio chunk #{self._audio_chunk_count} to Deepgram ({len(audio)} bytes)")
            await self._connection.send(audio)
            
            # Send to sibling services if available
            if self._sibling_deepgram:
                logger.trace(f"🔧 DeepgramSTTService: Sending audio to sibling Deepgram")
                await self._sibling_deepgram.send_audio(audio)
                
            if self._complementary_gladia:
                logger.trace(f"🎯 DeepgramSTTService: Sending audio to complementary Gladia")
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

    async def _should_ignore_transcription(self, result: LiveResultResponse):

        is_final = result.is_final
        confidence = result.channel.alternatives[0].confidence
        transcript = result.channel.alternatives[0].transcript
        time_start = result.channel.alternatives[0].words[0].start if result.channel.alternatives[0].words else 0
        
        if not is_final and confidence < 0.7:
            logger.debug("Ignoring iterim because low confidence")
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
            logger.debug("Ignoring Deepgram interruption because allow_interruptions is False")
            return True
        
        if self._bot_speaking and self._transcript_words_count(transcript) == 1: 
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
        if not self._restarted: 
            return
            
        enhanced = kwargs.pop('enhanced', False)
        
        try:
            result: LiveResultResponse = kwargs["result"]
            
            if len(result.channel.alternatives) == 0:
                return
            
            is_final = result.is_final
            speech_final = result.speech_final
            transcript = result.channel.alternatives[0].transcript
            confidence = result.channel.alternatives[0].confidence
            start_time = result.start
            
            # Enhanced logging for debugging
            source = "🎯 Gladia Enhanced" if enhanced else "⚡ Deepgram Standard"
            transcript_type = "FINAL" if is_final else "INTERIM"
            logger.info(f"{source}: 📋 {transcript_type} transcript received")
            logger.info(f"   📝 Text: '{transcript}'")
            logger.info(f"   🎯 Confidence: {confidence:.2f}")
            logger.info(f"   ⏰ Start time: {start_time}")
            logger.info(f"   🗣️ Speech final: {speech_final}")
            
            # Skip Deepgram finals if we're expecting enhanced results
            if is_final and not enhanced and self._complementary_gladia and not self._ignore_deepgram_finals:
                logger.debug(f"⚡ DeepgramSTTService: Storing Deepgram final and waiting for Gladia enhancement...")
                # Store Deepgram final and wait briefly for Gladia
                transcript_key = f"{transcript.lower()}_{start_time}"
                self._pending_deepgram_finals[transcript_key] = {
                    'result': result,
                    'timestamp': time.time()
                }
                
                logger.debug(f"⚡ DeepgramSTTService: Stored pending final with key: {transcript_key}")
                logger.debug(f"⏰ DeepgramSTTService: Setting {self._gladia_timeout}s timeout for Gladia response")
                
                # Set timeout to process Deepgram if Gladia doesn't respond
                async def timeout_handler():
                    await asyncio.sleep(self._gladia_timeout)
                    if transcript_key in self._pending_deepgram_finals:
                        logger.warning(f"⏰ DeepgramSTTService: TIMEOUT - No Gladia response, using Deepgram final: '{transcript}'")
                        pending = self._pending_deepgram_finals.pop(transcript_key)
                        await self._process_final_transcript(pending['result'], enhanced=False)
                
                asyncio.create_task(timeout_handler())
                return
            
            # Skip Deepgram finals when enhanced version is being processed
            if is_final and not enhanced and self._ignore_deepgram_finals:
                logger.info(f"🚫 DeepgramSTTService: Skipping Deepgram final (Gladia processing): '{transcript}'")
                return
            
            logger.debug(f"✅ DeepgramSTTService: Processing transcript from {source}")
            await self._process_transcript_message(result, enhanced)
                
        except Exception as e:
            logger.exception(f"{self} unexpected error in _on_message: {e}")

    async def _process_transcript_message(self, result, enhanced=False):
        """Process transcript message with enhanced source tracking."""
        is_final = result.is_final
        transcript = result.channel.alternatives[0].transcript
        confidence = result.channel.alternatives[0].confidence
        start_time = result.start
        
        language = None
        if result.channel.alternatives[0].languages:
            language = result.channel.alternatives[0].languages[0]
            language = Language(language)
        
        if len(transcript) > 0:
            source_detailed = "🎯 Enhanced Gladia" if enhanced else "⚡ Standard Deepgram"
            logger.debug(f"📏 DeepgramSTTService: Non-empty transcript received from {source_detailed}")
            await self.stop_ttfb_metrics()

            if await self._detect_and_handle_voicemail(transcript):
                logger.info(f"📞 DeepgramSTTService: Voicemail detected and handled")
                return 
            
            transcript_status = "FINAL" if is_final else "INTERIM"
            logger.debug(f"{source_detailed}: Processing {transcript_status} - '{transcript}'")
            logger.debug(f"   🎯 Confidence: {confidence:.2f}")
            logger.debug(f"   ⏰ Start time: {start_time}")
            
            if await self._should_ignore_transcription(result):
                logger.debug(f"🚫 DeepgramSTTService: Transcript ignored by filter")
                return
            
            if is_final:
                logger.info(f"🎯 DeepgramSTTService: Processing FINAL transcript from {source_detailed}")
                await self._process_final_transcript(result, enhanced)
            else:
                logger.debug(f"📢 DeepgramSTTService: Processing INTERIM transcript from {source_detailed}")
                await self._on_interim_transcript_message(transcript, language, start_time)

    async def _process_final_transcript(self, result, enhanced=False):
        """Process final transcript with enhanced source tracking."""
        transcript = result.channel.alternatives[0].transcript
        confidence = result.channel.alternatives[0].confidence
        start_time = result.start
        speech_final = getattr(result, 'speech_final', True)
        
        language = None
        if result.channel.alternatives[0].languages:
            language = result.channel.alternatives[0].languages[0]
            language = Language(language)
        
        # Enhanced response time measurement
        if self._current_speech_start_time is not None:
            elapsed = time.perf_counter() - self._current_speech_start_time
            elapsed_formatted = round(elapsed, 3)
            self._stt_response_times.append(elapsed_formatted)
            
            source_name = "🎯 Enhanced Gladia" if enhanced else "⚡ Standard Deepgram"
            logger.info(f"📊 {source_name}: ⏱️ STT Response Time: {elapsed_formatted}s")
            logger.info(f"   📝 Final Transcript: '{transcript}'")
            logger.info(f"   🎯 Confidence: {confidence:.2f}")
            logger.info(f"   📦 Audio chunks processed: {self._audio_chunk_count}")
            logger.info(f"   🗣️ Speech final: {speech_final}")
            
            self._current_speech_start_time = None
            self._audio_chunk_count = 0
            logger.debug(f"🔄 DeepgramSTTService: Reset speech timing counters")
        
        logger.debug(f"🎯 DeepgramSTTService: Calling _on_final_transcript_message for: '{transcript}'")
        await self._on_final_transcript_message(transcript, language, speech_final)
        self._last_time_transcription = start_time
        logger.debug(f"⏰ DeepgramSTTService: Updated last transcription time to {start_time}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("🤖 DeepgramSTTService: Received bot started speaking")
            await self._handle_bot_speaking()

        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("🤖 DeepgramSTTService: Received bot stopped speaking")
            await self._handle_bot_silence() 

        if isinstance(frame, STTRestartFrame):
            logger.info("🔄 DeepgramSTTService: Received STT Restart Frame - restarting services")
            self._restarted = True
            await self._disconnect()
            await self._connect()
            return

        if isinstance(frame, UserStartedSpeakingFrame) and not self.vad_enabled:
            logger.info("🎤 DeepgramSTTService: User started speaking (VAD disabled)")
            # Start metrics if Deepgram VAD is disabled & pipeline VAD has detected speech
            await self.start_metrics()
            # Reset timing when user starts speaking
            self._current_speech_start_time = time.perf_counter()
            self._audio_chunk_count = 0
            logger.debug(f"⏱️ DeepgramSTTService: Speech timer reset - waiting for transcriptions")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.debug("🎤 DeepgramSTTService: User stopped speaking - finalizing connection")
            # https://developers.deepgram.com/docs/finalize
            await self._connection.finalize()
            logger.trace(f"Triggered finalize event on: {frame.name}, {direction}")
        
        if isinstance(frame, VADInactiveFrame):
            logger.debug("🎤 DeepgramSTTService: VAD inactive")
            self._vad_active = False
            if self._connection and self._connection.is_connected:
                await self._connection.finalize()  
        elif isinstance(frame, VADActiveFrame):
            logger.debug("🎤 DeepgramSTTService: VAD active")
            self._vad_active = True