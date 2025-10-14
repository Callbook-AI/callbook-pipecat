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
IGNORE_REPEATED_MSG_AT_START_SECONDS = 4
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
        model: str = "stt-rt-v2"  # Soniox's real-time model
        enable_speaker_diarization: bool = False
        enable_language_identification: bool = False
        allow_interruptions: bool = True
        detect_voicemail: bool = True
        fast_response: bool = False
        on_no_punctuation_seconds: float = DEFAULT_ON_NO_PUNCTUATION_SECONDS

    def __init__(
        self,
        *,
        api_key: str,
        sample_rate: int = 16000,
        params: InputParams = InputParams(),
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._websocket = None
        self._receive_task = None
        self._async_handler_task = None
        self._connection_active = False

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
            yield None
            return

        if self._current_speech_start_time is None:
            self._current_speech_start_time = time.perf_counter()
            self._audio_chunk_count = 0
            logger.debug("🎤 Soniox: Starting speech detection timer.")

        self._audio_chunk_count += 1
        self._last_audio_chunk_time = time.time()
        
        try:
            await self._websocket.send(audio)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Soniox WebSocket connection closed, attempting to reconnect.")
            await self._reconnect()
        yield None

    async def _connect(self):
        """Establish websocket connection to Soniox service."""
        if self._websocket and self._connection_active:
            return

        try:
            logger.info("Connecting to Soniox...")
            
            # Validate API key before attempting connection
            if not self._api_key or self._api_key.strip() == "":
                raise ValueError("Soniox API key is empty or invalid")
            
            logger.debug(f"Using Soniox API key: {self._api_key[:10]}...")

            uri = (
                f"wss://api.soniox.com/v1/speech-to-text-rt"
                f"?model={self._params.model}"
                f"&language={language_to_soniox_language(self._language)}"
                f"&sample_rate={self.sample_rate}"
                f"&enable_speaker_diarization={str(self._params.enable_speaker_diarization).lower()}"
                f"&enable_language_identification={str(self._params.enable_language_identification).lower()}"
            )

            self._websocket = await websockets.connect(
                uri, 
                extra_headers={"Authorization": f"Bearer {self._api_key}"}
            )
            self._connection_active = True
            logger.info("Connected to Soniox.")

            if not self._receive_task:
                self._receive_task = self.create_task(self._receive_task_handler())
            
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            
        except Exception as e:
            logger.error(f"Failed to connect to Soniox: {e}")
            await self.push_error(ErrorFrame(f"Soniox connection failed: {e}"))

    async def _disconnect(self):
        """Disconnect from Soniox service and clean up resources."""
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
            await self._websocket.close()
            self._websocket = None
            logger.info("Disconnected from Soniox.")

    async def _reconnect(self):
        await self._disconnect()
        await asyncio.sleep(1)
        await self._connect()

    async def _receive_task_handler(self):
        """Handle incoming transcription messages from Soniox."""
        while self._connection_active:
            try:
                message = await self._websocket.recv()
                await self._on_message(json.loads(message))
            except websockets.exceptions.ConnectionClosed:
                logger.warning("Soniox connection closed during receive.")
                break
            except Exception as e:
                logger.error(f"Error in Soniox receive task: {e}")
                break

    async def _on_message(self, data: Dict):
        """Process incoming transcription message.
        
        Handles transcript messages from Soniox with Deepgram-style processing.
        """
        try:
            await self.stop_ttfb_metrics()

            words = data.get("words", [])
            if not words:
                return

            transcript = " ".join(w["text"] for w in words)
            is_final = data.get("final", False)

            if not transcript.strip():
                return

            # Check if we should ignore this transcription
            if await self._should_ignore_transcription(transcript):
                return

            # Update interim time for false interim detection
            if not is_final:
                self._last_interim_time = time.time()

            timestamp = time_now_iso8601()
            language_enum = self._language

            if is_final:
                self._record_stt_performance(transcript, words)
                await self._on_final_transcript_message(transcript, language_enum)
                self._last_final_transcript_time = time.time()
                self._last_time_transcription = time.time()
            else:
                await self._on_interim_transcript_message(transcript, language_enum)
                
        except Exception as e:
            logger.error(f"Error processing Soniox message: {e}")

    async def _on_final_transcript_message(self, transcript: str, language: Language):
        """Handle final transcript following Deepgram pattern."""
        
        # Check for voicemail detection
        if self.detect_voicemail:
            await self._detect_and_handle_voicemail(transcript)
        
        # Handle first message tracking
        self._handle_first_message(transcript)
        
        # Check for repeated first message
        if self._should_ignore_first_repeated_message(transcript):
            logger.debug(f"Ignoring repeated first message: '{transcript}'")
            return
        
        await self._handle_user_speaking()
        
        # Create transcription frame
        frame = TranscriptionFrame(
            transcript,
            "",
            time_now_iso8601(),
            language
        )
        
        # Handle accumulation or immediate sending based on fast response mode
        if self._is_accum_transcription(transcript):
            self._append_accum_transcription(frame)
        else:
            self._append_accum_transcription(frame)
            await self._send_accum_transcriptions()

    async def _on_interim_transcript_message(self, transcript: str, language: Language):
        """Handle interim transcript."""
        await self._handle_user_speaking()
        
        frame = InterimTranscriptionFrame(
            transcript,
            "",
            time_now_iso8601(),
            language
        )
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)

    def _record_stt_performance(self, transcript, words):
        """Record STT performance metrics."""
        if self._current_speech_start_time:
            elapsed = time.perf_counter() - self._current_speech_start_time
            self._stt_response_times.append(elapsed)
            confidence = sum(w.get("confidence", 0) for w in words) / len(words) if words else 0
            logger.info(f"📊 ⚡ Soniox: ⏱️ STT Response Time: {elapsed:.3f}s")
            logger.info(f"   📝 Final Transcript: '{transcript}'")
            logger.info(f"   🎯 Avg. Confidence: {confidence:.2f}")
            logger.info(f"   📦 Audio chunks processed: {self._audio_chunk_count}")
            self._current_speech_start_time = None

    async def _handle_user_speaking(self):
        """Handle user started speaking event."""
        if not self._user_speaking:
            await self.push_frame(StartInterruptionFrame())
            self._user_speaking = True
            await self.push_frame(UserStartedSpeakingFrame())
            logger.debug(f"👤 {self}: User started speaking")

    async def _handle_user_silence(self):
        """Handle user stopped speaking event."""
        if self._user_speaking:
            self._user_speaking = False
            self._current_speech_start_time = None
            await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            logger.debug(f"👤 {self}: User stopped speaking")
            
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
        """Check if this is a repeated first message to ignore."""
        if not self._first_message:
            return False
        
        time_since_first_message = time.time() - self._first_message_time
        if time_since_first_message > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        return is_equivalent_basic(text, self._first_message)

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
        if not len(self._accum_transcription_frames):
            return

        # Combine all transcripts into one message
        full_text = " ".join([frame.text for frame in self._accum_transcription_frames])
        
        # Check if this is a DUPLICATE transcript
        if full_text.strip() == self._last_sent_transcript:
            logger.debug(f"⚠️  Skipping DUPLICATE transcript: '{full_text}'")
            self._accum_transcription_frames = []
            return
        
        logger.debug(f"{self}: Sending {len(self._accum_transcription_frames)} accumulated transcription(s)")
        logger.debug(f"📝 Sending transcript as TranscriptionFrame: '{full_text}'")
        
        # Store what we're sending for deduplication
        self._last_sent_transcript = full_text.strip()
        
        # Ensure transport processes UserStoppedSpeakingFrame BEFORE TranscriptionFrame
        # This prevents race condition in transport's frame emulation logic
        was_user_speaking = self._user_speaking
        if was_user_speaking:
            self._user_speaking = False
            self._current_speech_start_time = None
            await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)
            logger.debug(f"⚡ Sent UserStoppedSpeakingFrame UPSTREAM")
            
            # Give transport time to process UserStoppedSpeakingFrame
            await asyncio.sleep(0.001)  # 1ms for async frame processing
            logger.debug(f"⚡ Waited for transport to process UserStoppedSpeakingFrame")
        
        # Now send TranscriptionFrame
        await self.push_frame(
            TranscriptionFrame(
                full_text,
                "",
                time_now_iso8601(),
                self._language
            ),
            FrameDirection.DOWNSTREAM
        )
        logger.debug(f"⚡ Sent TranscriptionFrame DOWNSTREAM")
        
        self._accum_transcription_frames = []
        await self.stop_processing_metrics()

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
        await super().process_frame(frame, direction)
        
        # Handle bot speaking state for interruption detection
        if isinstance(frame, BotStartedSpeakingFrame):
            await self._handle_bot_speaking()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self._handle_bot_silence()
        elif isinstance(frame, VADActiveFrame):
            self._vad_active = True
            logger.debug(f"🎤 {self}: VAD active")
        elif isinstance(frame, VADInactiveFrame):
            self._vad_active = False
            self._vad_inactive_time = time.time()
            logger.debug(f"🎤 {self}: VAD inactive")