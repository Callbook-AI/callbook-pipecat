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
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    StartFrame,
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
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._api_key = api_key
        self._websocket = None
        self._receive_task = None
        self._user_speaking = False
        self._language = language
        
        # Optimized settings following Deepgram pattern
        self._settings = {
            "sample_rate": sample_rate or 16000,
            "format_turns": format_turns,
            "language": language,
        }
        
        # Performance tracking
        self._stt_response_times = []
        self._current_speech_start_time = None
        self._connection_active = False
        
        # Accumulation for sentence handling (like Deepgram)
        self._accum_transcription_frames = []
        self._last_time_accum_transcription = time.time()
        
        logger.info(f"AssemblyAI STT Service initialized with language: {language}, sample_rate: {self._settings['sample_rate']}")

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
        Following Deepgram's optimized pattern for efficient audio processing.

        :param audio: Audio data as bytes (PCM 16-bit)
        :yield: None (transcription frames are pushed via callbacks)
        """
        if self._websocket and self._connection_active:
            try:
                await self.start_processing_metrics()
                await self._websocket.send(audio)
                await self.stop_processing_metrics()
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
            }
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
                # Send terminate message
                await self._websocket.send(json.dumps({"type": "Terminate"}))
                await self._websocket.close()
                logger.info(f"{self}: Disconnected from AssemblyAI")
            except Exception as e:
                logger.debug(f"{self}: Error during disconnect: {e}")
            finally:
                self._websocket = None

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
                
                # Stop TTFB metrics on first transcript
                await self.stop_ttfb_metrics()
                
                # Track response time
                if self._current_speech_start_time is not None:
                    elapsed = time.perf_counter() - self._current_speech_start_time
                    self._stt_response_times.append(elapsed)
                    logger.debug(f"{self}: STT response time: {elapsed:.3f}s")
                
                # Check if it's a formatted turn (final) or interim
                is_formatted = data.get('turn_is_formatted', False)
                timestamp = time_now_iso8601()
                
                # Convert language string to Language enum
                language = Language.ES if self._language == "es" else Language.EN
                
                if is_formatted:
                    # This is a final, formatted transcript
                    logger.info(f"{self}: FINAL transcript: '{transcript}'")
                    
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
                    # This is an interim transcript
                    logger.debug(f"{self}: INTERIM transcript: '{transcript}'")
                    
                    await self._handle_user_speaking()
                    
                    frame = InterimTranscriptionFrame(
                        transcript,
                        "",
                        timestamp,
                        language
                    )
                    
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
