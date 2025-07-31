
import asyncio
import base64
import json
import time
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp
from loguru import logger
from pydantic import BaseModel

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
    STTRestartFrame,
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
        "In order to use Gladia, you need to `pip install pipecat-ai[gladia]`. Also, set `GLADIA_API_KEY` environment variable."
    )
    raise Exception(f"Missing module: {e}")


DEFAULT_ON_NO_PUNCTUATION_SECONDS = 3 
IGNORE_REPEATED_MSG_AT_START_SECONDS = 4 
VOICEMAIL_DETECTION_SECONDS = 10        
FALSE_INTERIM_SECONDS = 1.3            


def language_to_gladia_language(language: Language) -> Optional[str]:
    BASE_LANGUAGES = {
        Language.AF: "af",
        Language.AM: "am",
        Language.AR: "ar",
        Language.AS: "as",
        Language.AZ: "az",
        Language.BG: "bg",
        Language.BN: "bn",
        Language.BS: "bs",
        Language.CA: "ca",
        Language.CS: "cs",
        Language.CY: "cy",
        Language.DA: "da",
        Language.DE: "de",
        Language.EL: "el",
        Language.EN: "en",
        Language.ES: "es",
        Language.ET: "et",
        Language.EU: "eu",
        Language.FA: "fa",
        Language.FI: "fi",
        Language.FR: "fr",
        Language.GA: "ga",
        Language.GL: "gl",
        Language.GU: "gu",
        Language.HE: "he",
        Language.HI: "hi",
        Language.HR: "hr",
        Language.HU: "hu",
        Language.HY: "hy",
        Language.ID: "id",
        Language.IS: "is",
        Language.IT: "it",
        Language.JA: "ja",
        Language.JV: "jv",
        Language.KA: "ka",
        Language.KK: "kk",
        Language.KM: "km",
        Language.KN: "kn",
        Language.KO: "ko",
        Language.LO: "lo",
        Language.LT: "lt",
        Language.LV: "lv",
        Language.MK: "mk",
        Language.ML: "ml",
        Language.MN: "mn",
        Language.MR: "mr",
        Language.MS: "ms",
        Language.MT: "mt",
        Language.MY: "my",
        Language.NE: "ne",
        Language.NL: "nl",
        Language.NO: "no",
        Language.OR: "or",
        Language.PA: "pa",
        Language.PL: "pl",
        Language.PS: "ps",
        Language.PT: "pt",
        Language.RO: "ro",
        Language.RU: "ru",
        Language.SI: "si",
        Language.SK: "sk",
        Language.SL: "sl",
        Language.SO: "so",
        Language.SQ: "sq",
        Language.SR: "sr",
        Language.SU: "su",
        Language.SV: "sv",
        Language.SW: "sw",
        Language.TA: "ta",
        Language.TE: "te",
        Language.TH: "th",
        Language.TR: "tr",
        Language.UK: "uk",
        Language.UR: "ur",
        Language.UZ: "uz",
        Language.VI: "vi",
        Language.ZH: "zh",
        Language.ZU: "zu",
    }

    result = BASE_LANGUAGES.get(language)

    if not result:
        lang_str = str(language.value)
        base_code = lang_str.split("-")[0].lower()
        result = base_code if base_code in BASE_LANGUAGES.values() else None

    return result


class GladiaSTTService(STTService):
    class InputParams(BaseModel):
        language: Optional[Language] = Language.EN
        endpointing: Optional[float] = 0.4
        maximum_duration_without_endpointing: Optional[int] = 6
        audio_enhancer: Optional[bool] = False
        words_accurate_timestamps: Optional[bool] = None
        speech_threshold: Optional[float] = 0.6
        # Additional parameters to match Deepgram
        model: Optional[str] = "solaria-1"
        allow_interruptions: Optional[bool] = True
        detect_voicemail: Optional[bool] = True
        region: Optional[str] = "us-east"

    def __init__(
        self,
        *,
        api_key: str,
        url: str = "https://api.gladia.io/v2/live",
        confidence: float = 0.6,  # Lowered for faster response
        sample_rate: Optional[int] = None,
        params: InputParams = InputParams(),
        on_no_punctuation_seconds: float = 0.6,  # Reduced for faster aggregation
        detect_voicemail: bool = True,
        allow_interruptions: bool = True,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        
        self._api_key = api_key
        self._url = url
        self.language = params.language
        self.detect_voicemail = params.detect_voicemail if params.detect_voicemail is not None else detect_voicemail
        self._allow_stt_interruptions = params.allow_interruptions if params.allow_interruptions is not None else allow_interruptions
        self._model = params.model
        
        logger.debug(f"Allow interruptions: {self._allow_stt_interruptions}")
        logger.debug(f"Detect voicemail: {self.detect_voicemail}")
        logger.debug(f"Model: {self._model}")
        
        # Optimized settings for low latency
        if params.model == "solaria-1":
            if params.speech_threshold is None:
                params.speech_threshold = 0.65  # Lowered for faster detection
            if params.audio_enhancer is None:
                params.audio_enhancer = False  
            if params.words_accurate_timestamps is None:
                params.words_accurate_timestamps = False 
            if params.endpointing is None:
                params.endpointing = 0.05  # Much more aggressive endpointing
            if params.maximum_duration_without_endpointing is None:
                params.maximum_duration_without_endpointing = 4  # Shorter timeout
        
        self._settings = {
            "encoding": "wav/pcm",
            "bit_depth": 16,
            "sample_rate": 0,  # Will be set in _connect()
            "channels": 1,
            "model": params.model,  # Required top-level field
            "endpointing": params.endpointing,
            "maximum_duration_without_endpointing": params.maximum_duration_without_endpointing,
            "language_config": {
                "languages": [self.language_to_service_language(params.language)]
                if params.language
                else [],
                "code_switching": False,
            },
            "pre_processing": {
                "audio_enhancer": params.audio_enhancer,
                "speech_threshold": params.speech_threshold,
            },
            "realtime_processing": {
                "words_accurate_timestamps": params.words_accurate_timestamps,
            },
            # Required messages_config for WebSocket events
            "messages_config": {
                "receive_final_transcripts": True,
                "receive_speech_events": True,
                "receive_pre_processing_events": True,
                "receive_realtime_processing_events": True,
                "receive_post_processing_events": True,
                "receive_acknowledgments": True,
                "receive_errors": True,
                "receive_lifecycle_events": False
            }
        }
        
        self._confidence = confidence
        self._websocket = None
        self._receive_task = None
        self._async_handler_task = None
        
        self._user_speaking = False
        self._bot_speaking = True
        self._on_no_punctuation_seconds = on_no_punctuation_seconds
        self._vad_active = False
        
        self._first_message = None
        self._first_message_time = None
        self._last_interim_time = None
        self._restarted = False  
        
        self._accum_transcription_frames = []
        self._last_time_accum_transcription = time.time()
        self._last_time_transcription = time.time()
        self._was_first_transcript_receipt = False
        
        # Performance tracking
        self.start_time = time.time()
        self._stt_response_times = [] 
        self._current_speech_start_time = None  
        self._last_audio_chunk_time = None 
        self._transcript_start_times = {}  
        self._audio_chunk_count = 0  
        
        # New streaming optimization variables (limited without interim support)
        self._last_transcript_time = None
        self._fast_response_mode = True  # Focus on fastest final results  

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
            logger.info(f"🎯 Gladia STT Performance Summary:")
            logger.info(f"   📊 Total responses: {stats['count']}")
            logger.info(f"   ⏱️  Average time: {stats['average']}s")
            logger.info(f"   🏃 Fastest: {stats['min']}s")
            logger.info(f"   🐌 Slowest: {stats['max']}s")
            logger.info(f"   🕐 Latest: {stats['latest']}s")
            logger.info(f"   📈 All times: {stats['all_times']}")

    def language_to_service_language(self, language: Language) -> Optional[str]:
        return language_to_gladia_language(language)

    def _transcript_words_count(self, transcript: str):
        return len(transcript.split(" "))
    
    async def set_model(self, model: str):
        try:
            await super().set_model(model)
            logger.info(f"Switching STT model to: [{model}]")
            self._model = model
            await self._disconnect()
            await self._connect()
        except Exception as e:
            logger.exception(f"{self} exception in set_model: {e}")
            raise

    async def set_language(self, language: Language):
        try:
            logger.info(f"Switching STT language to: [{language}]")
            self.language = language
            self._settings["language_config"]["languages"] = [self.language_to_service_language(language)]
            await self._disconnect()
            await self._connect()
        except Exception as e:
            logger.exception(f"{self} exception in set_language: {e}")
            raise

    async def start(self, frame: StartFrame):
        await super().start(frame)
        self._restarted = True 
        await self._connect()

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._disconnect()

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        current_time = time.perf_counter()
        self._last_audio_chunk_time = current_time
        self._audio_chunk_count += 1
        
        if self._current_speech_start_time is None:
            self._current_speech_start_time = current_time
            logger.debug(f"🎤 Gladia: Starting speech detection timer at chunk #{self._audio_chunk_count}")
        
        await self.start_processing_metrics()
        await self._send_audio(audio)
        yield None

    async def _connect(self):
        try:
            if self._websocket:
                return
                
            logger.debug("Connecting to Gladia")
            self._settings["sample_rate"] = self.sample_rate
            response = await self._setup_gladia()
            self._websocket = await websockets.connect(response["url"])
            
            if not self._receive_task:
                self._receive_task = self.create_task(self._receive_task_handler())
                
            if not self._async_handler_task:
                self._async_handler_task = self.create_task(self._async_handler())
                
            logger.debug("Connected to Gladia")
        except Exception as e:
            logger.exception(f"{self} exception in _connect: {e}")
            raise

    async def _disconnect(self):
        try:
            if self._async_handler_task:
                await self.cancel_task(self._async_handler_task)
                self._async_handler_task = None
                
            if self._receive_task:
                await self.cancel_task(self._receive_task)
                self._receive_task = None
                
            if self._websocket:
                await self._send_stop_recording()
                await self._websocket.close()
                self._websocket = None
                logger.debug("Disconnected from Gladia")
        except Exception as e:
            logger.exception(f"{self} exception in _disconnect: {e}")

    async def _setup_gladia(self):
        async with aiohttp.ClientSession() as session:
            logger.debug("Setting up Gladia session with parameters")
            logger.debug(f"Gladia settings: {self._settings}")
            async with session.post(
                self._url,
                headers={"X-Gladia-Key": self._api_key, "Content-Type": "application/json"},
                json=self._settings,
            ) as response:
                if response.ok:
                    return await response.json()
                else:
                    logger.error(
                        f"Gladia error: {response.status}: {response.text or response.reason}"
                    )
                    raise Exception(f"Failed to initialize Gladia session: {response.status}")

    async def _send_audio(self, audio: bytes):
        if self._websocket:
            data = base64.b64encode(audio).decode("utf-8")
            message = {"type": "audio_chunk", "data": {"chunk": data}}
            await self._websocket.send(json.dumps(message))

    async def _send_stop_recording(self):
        if self._websocket:
            await self._websocket.send(json.dumps({"type": "stop_recording"}))
    
    # Frame processing methods matching Deepgram
    async def _handle_user_speaking(self):
        # Only push interruption frame if bot is actually speaking
        if self._bot_speaking:
            logger.debug("Pushing StartInterruptionFrame - bot was speaking")
            await self.push_frame(StartInterruptionFrame())
        
        if self._user_speaking:
            return

        self._user_speaking = True
        logger.debug("User started speaking")
        await self.push_frame(UserStartedSpeakingFrame())

    async def _handle_user_silence(self):
        if not self._user_speaking:
            return

        self._user_speaking = False
        logger.debug("User stopped speaking")
        await self.push_frame(UserStoppedSpeakingFrame())

    async def _handle_bot_speaking(self):
        logger.debug("Bot started speaking - setting bot_speaking=True")
        self._bot_speaking = True

    async def _handle_bot_silence(self):
        logger.debug("Bot stopped speaking - setting bot_speaking=False")  
        self._bot_speaking = False
    
    # Accumulation handling matching Deepgram
    async def _async_handle_accum_transcription(self, current_time):
        # More aggressive timeouts for faster response (no interim results available)
        timeout_threshold = min(self._on_no_punctuation_seconds, 0.3)  # Cap at 300ms for max speed
        
        if current_time - self._last_time_accum_transcription > timeout_threshold and len(self._accum_transcription_frames):
            logger.debug(f"Sending accum transcription after {timeout_threshold}s timeout")
            await self._send_accum_transcriptions()
        
        # Send immediately if we have accumulated frames and some time has passed
        elif len(self._accum_transcription_frames) > 0 and current_time - self._last_time_accum_transcription > 0.15:
            logger.debug("Fast-sending accum transcription")
            await self._send_accum_transcriptions()

    async def _handle_false_interim(self, current_time):
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
    
    async def _async_handler(self):
        while True:
            await asyncio.sleep(0.05)  # Faster polling for better responsiveness
            
            current_time = time.time()
            await self._async_handle_accum_transcription(current_time)
            await self._handle_false_interim(current_time)
    
    async def _send_accum_transcriptions(self):
        if not len(self._accum_transcription_frames):
            return

        # Send transcriptions with proper user speaking frame handling like Deepgram
        await self._handle_user_speaking()
        
        # Send all frames at once for better compatibility with text aggregators
        for frame in self._accum_transcription_frames:
            await self.push_frame(frame)
        
        # Clear frames before sending user silence to prevent re-processing
        self._accum_transcription_frames = []

        await self._handle_user_silence()
        await self.stop_processing_metrics()

    def _is_accum_transcription(self, text: str):
        END_OF_PHRASE_CHARACTERS = ['.', '?']
        text = text.strip()
        if not text:
            return True
        return not text[-1] in END_OF_PHRASE_CHARACTERS
    
    def _append_accum_transcription(self, frame: TranscriptionFrame):
        self._last_time_accum_transcription = time.time()
        self._accum_transcription_frames.append(frame)

    def _handle_first_message(self, text):
        if self._first_message:
            return 

        self._first_message = text
        self._first_message_time = time.time()

    def _should_ignore_first_repeated_message(self, text):
        if not self._first_message:
            return False
        
        time_since_first_message = time.time() - self._first_message_time
        if time_since_first_message > IGNORE_REPEATED_MSG_AT_START_SECONDS:
            return False
        
        return is_equivalent_basic(text, self._first_message)    
    async def _should_ignore_transcription(self, transcript: str, is_final: bool, confidence: float):
        # Optimized for fastest final results (no interim support in Solaria-1)
        
        # For final results, use aggressive but reasonable thresholds
        if confidence < self._confidence:
            return True

        # For single words, use lower confidence threshold for speed
        if self._transcript_words_count(transcript) == 1 and confidence < 0.4:
            return True
        
        # Allow interruptions with reasonable confidence
        if self._bot_speaking and not self._allow_stt_interruptions:
            if confidence < 0.6:  # Only block low-confidence during bot speech
                logger.debug(f"Ignoring transcription: bot speaking and confidence too low")
                return True
        
        # Skip repeated first messages
        if self._should_ignore_first_repeated_message(transcript):
            return True

        return False
    
    async def _detect_and_handle_voicemail(self, transcript: str):
        if not self.detect_voicemail:
            return False

        logger.debug(transcript)
        logger.debug(self._time_since_init())
        
        if self._time_since_init() > VOICEMAIL_DETECTION_SECONDS and self._was_first_transcript_receipt:
            return False
        
        if not voicemail.is_text_voicemail(transcript):
            return False
        
        logger.debug("Voicemail detected")
        await self.push_frame(VoicemailFrame(transcript))
        logger.debug("Voicemail pushed")
        return True
    
    async def _on_final_transcript_message(self, transcript, language):
        frame = TranscriptionFrame(transcript, "", time_now_iso8601(), language)
        
        self._handle_first_message(frame.text)
        self._was_first_transcript_receipt = True
        
        if not self._user_speaking:
            await self._handle_user_speaking()
            await asyncio.sleep(0.001)
        
        await self.push_frame(frame)
        
        self._last_time_transcription = time.time()
        
        await asyncio.sleep(0.001)
        await self._handle_user_silence()
        
        await self.stop_processing_metrics()
    
    async def _on_interim_transcript_message(self, transcript, language):
        self._last_interim_time = time.time()
        
        # Store interim for potential accumulation like Deepgram
        self._interim_buffer = transcript
        
        # Always trigger user speaking state for interim results
        if not self._user_speaking:
            await self._handle_user_speaking()
        
        # Send interim frame immediately for streaming
        await self.push_frame(
            InterimTranscriptionFrame(transcript, "", time_now_iso8601(), language)
        )
        
        # Count words for streaming optimization
        word_count = self._transcript_words_count(transcript)
        self._streaming_word_count = word_count
        
        # For longer interim results, start accumulating for faster final processing
        if word_count >= 3:  # Start accumulating multi-word interim results
            frame = TranscriptionFrame(transcript, "", time_now_iso8601(), language)
            if self._is_accum_transcription(transcript):
                self._append_accum_transcription(frame)
                logger.debug(f"Accumulated interim transcript for streaming: {transcript}")
        
        logger.debug(f"📢 Interim sent: '{transcript}' (words: {word_count})")

    async def _receive_task_handler(self):
        if not self._restarted:
            return
            
        async for message in self._websocket:
            try:
                content = json.loads(message)
                if content["type"] != "transcript":
                    continue
                    
                utterance = content["data"]["utterance"]
                confidence = utterance.get("confidence", 0)
                transcript = utterance["text"]
                is_final = content["data"]["is_final"]
                
                # Fast path: ignore empty transcripts
                if not transcript.strip():
                    continue
                
                # Stop TTFB metrics as soon as we get any transcript
                if len(transcript) > 0:
                    await self.stop_ttfb_metrics()
                
                # Track performance for final transcripts
                if is_final and self._current_speech_start_time is not None:
                    elapsed = time.perf_counter() - self._current_speech_start_time
                    elapsed_formatted = round(elapsed, 3)
                    self._stt_response_times.append(elapsed_formatted)
                    
                    logger.debug(f"📊 Gladia STT Response Time: {elapsed_formatted}s")
                    logger.debug(f"   📝 Transcript: '{transcript}'")
                    logger.debug(f"   🎯 Confidence: {confidence:.2f}")
                    logger.debug(f"   📦 Audio chunks processed: {self._audio_chunk_count}")
                    
                    self._current_speech_start_time = None
                    self._audio_chunk_count = 0
                
                # Early voicemail detection
                if await self._detect_and_handle_voicemail(transcript):
                    return
                
                # Use optimized transcription filtering
                if await self._should_ignore_transcription(transcript, is_final, confidence):
                    continue
                
                language = self.language
                
                # Process transcripts immediately without additional delays
                if is_final:
                    await self._on_final_transcript_message(transcript, language)
                    self._last_time_transcription = time.time()
                    # Clear interim buffer on final
                    self._interim_buffer = ""
                else:
                    # Process interim immediately
                    await self._on_interim_transcript_message(transcript, language)
                            
            except Exception as e:
                logger.exception(f"{self} unexpected error in _receive_task_handler: {e}")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, BotStartedSpeakingFrame):
            logger.debug("Received bot started speaking on Gladia")
            await self._handle_bot_speaking()

        if isinstance(frame, BotStoppedSpeakingFrame):
            logger.debug("Received bot stopped speaking on Gladia")
            await self._handle_bot_silence()

        if isinstance(frame, STTRestartFrame):
            logger.debug("Received STT Restart Frame")
            self._restarted = True
            self._accum_transcription_frames = []
            await self._disconnect()
            await self._connect()
            return

        if isinstance(frame, UserStartedSpeakingFrame):
            await self.start_ttfb_metrics()
            await self.start_processing_metrics()
            self._current_speech_start_time = time.perf_counter()
            self._audio_chunk_count = 0
            logger.debug(f"🎤 Gladia: User started speaking - resetting timer")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.trace(f"Triggered finalize equivalent on: {frame.name}, {direction}")
            if len(self._accum_transcription_frames) > 0:
                logger.debug("Sending accumulated transcriptions on UserStoppedSpeaking")
                await self._send_accum_transcriptions()
        
        if isinstance(frame, VADInactiveFrame):
            self._vad_active = False
        elif isinstance(frame, VADActiveFrame):
            self._vad_active = True
            fake_interim = InterimTranscriptionFrame(
                "VAD active", "", time_now_iso8601(), self.language
            )
            await self.push_frame(fake_interim)
            
        if isinstance(frame, StartInterruptionFrame):
            logger.debug("Received StartInterruptionFrame - clearing accumulated transcriptions")
            self._accum_transcription_frames = []
            self._last_time_accum_transcription = time.time()
            if self._user_speaking:
                await self._handle_user_silence()

