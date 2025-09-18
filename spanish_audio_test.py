#!/usr/bin/env python3
"""
AssemblyAI Universal Streaming Test with Spanish Audio File
Plays a Spanish audio file through the Universal Streaming API for transcription
"""

import asyncio
import time
import threading
import sys
import signal
import logging
import queue
import wave
from threading import Thread

try:
    import assemblyai as aai
    from assemblyai.streaming.v3 import (
        BeginEvent,
        StreamingClient,
        StreamingClientOptions,
        StreamingError,
        StreamingEvents,
        StreamingParameters,
        TerminationEvent,
        TurnEvent,
    )
except ImportError:
    print("❌ Error: AssemblyAI v3 streaming not available. Please upgrade: pip install --upgrade assemblyai")
    sys.exit(1)

# Configuration
ASSEMBLY_API_KEY = "50eb210feb72457593d3f811a9f2a698"
AUDIO_FILE_PATH = "/Users/david/Downloads/audiotest1.wav"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFileStream:
    """Stream audio from a WAV file"""
    
    def __init__(self, audio_file_path, chunk_duration_ms=64):
        self.audio_file_path = audio_file_path
        self.chunk_duration_ms = chunk_duration_ms
        self._buffer = queue.Queue()
        self._streaming_thread = None
        self._stop_streaming = threading.Event()
        
        # Audio file properties (will be set when file is opened)
        self.sample_rate = None
        self.channels = None
        self.sample_width = None
        self.total_frames = None
        self.chunk_size = None
        
    def load_audio_file(self):
        """Load and validate the audio file"""
        try:
            with wave.open(self.audio_file_path, 'rb') as wav_file:
                self.sample_rate = wav_file.getframerate()
                self.channels = wav_file.getnchannels()
                self.sample_width = wav_file.getsampwidth()
                self.total_frames = wav_file.getnframes()
                
                # Calculate chunk size based on desired duration
                self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000)
                
                duration = self.total_frames / self.sample_rate
                
                print(f"🎵 Audio file loaded: {self.audio_file_path}")
                print(f"   📊 Sample Rate: {self.sample_rate} Hz")
                print(f"   📊 Channels: {self.channels}")
                print(f"   📊 Sample Width: {self.sample_width} bytes")
                print(f"   📊 Duration: {duration:.1f} seconds")
                print(f"   📊 Total Frames: {self.total_frames}")
                print(f"   📊 Chunk Size: {self.chunk_size} frames ({self.chunk_duration_ms}ms)")
                
                # Validate format for AssemblyAI
                if self.sample_width != 2:
                    print(f"⚠️  Warning: Audio is {self.sample_width * 8}-bit, AssemblyAI expects 16-bit")
                if self.channels != 1:
                    print(f"⚠️  Warning: Audio has {self.channels} channels, AssemblyAI expects mono")
                
                return True
                
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            return False
    
    def start_streaming(self):
        """Start streaming audio file"""
        if not self.load_audio_file():
            return False
            
        self._stop_streaming.clear()
        self._streaming_thread = Thread(target=self._streaming_loop)
        self._streaming_thread.daemon = True
        self._streaming_thread.start()
        return True
    
    def _streaming_loop(self):
        """Stream audio file in chunks"""
        try:
            with wave.open(self.audio_file_path, 'rb') as wav_file:
                print(f"🎵 Starting audio playback...")
                
                frames_read = 0
                chunk_count = 0
                
                while frames_read < self.total_frames and not self._stop_streaming.is_set():
                    # Read chunk from file
                    chunk_frames = min(self.chunk_size, self.total_frames - frames_read)
                    audio_data = wav_file.readframes(chunk_frames)
                    
                    if not audio_data:
                        break
                    
                    # Add to buffer
                    self._buffer.put(audio_data)
                    frames_read += chunk_frames
                    chunk_count += 1
                    
                    # Progress indicator
                    if chunk_count % 50 == 0:  # Every ~3 seconds
                        progress = (frames_read / self.total_frames) * 100
                        print(f"🎵 Progress: {progress:.1f}% ({frames_read}/{self.total_frames} frames)")
                    
                    # Maintain real-time playback speed
                    time.sleep(self.chunk_duration_ms / 1000.0)
                
                print(f"🎵 Audio file streaming completed ({chunk_count} chunks sent)")
                
                # Keep sending empty chunks to maintain connection
                while not self._stop_streaming.is_set():
                    self._buffer.put(b'')
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Audio streaming error: {e}")
    
    def read(self):
        """Read next chunk of audio data"""
        try:
            return self._buffer.get(timeout=0.1)
        except queue.Empty:
            return b''  # Return empty if no data available
    
    def stop_streaming(self):
        """Stop streaming"""
        self._stop_streaming.set()
        if self._streaming_thread:
            self._streaming_thread.join(timeout=1.0)
    
    def close(self):
        """Clean up"""
        self.stop_streaming()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        data = self.read()
        if data is None:
            raise StopIteration
        return data

class SpanishAudioTester:
    def __init__(self, audio_file_path):
        self.audio_file_path = audio_file_path
        
        print("🎤 AssemblyAI Universal Streaming - Spanish Audio Test")
        print("=" * 60)
        print("ℹ️  Using Universal Streaming API (v3)")
        print("🎵 Playing Spanish audio file through streaming API")
        print(f"📁 File: {audio_file_path}")
        print("🌍 Language: Automatic detection (should detect Spanish)")
        print("=" * 60)
        
        self.client = None
        self.audio_stream = None
        
        # Metrics
        self.turns_received = 0
        self.words_transcribed = 0
        self.session_start_time = None
        self.transcripts = []

    def on_begin(self, client, event: BeginEvent):
        """Called when streaming session begins"""
        self.session_start_time = time.time()
        print(f"🔗 Session started: {event.id}")
        print("🎯 Spanish audio transcription will appear below:")
        print("-" * 50)

    def on_turn(self, client, event: TurnEvent):
        """Called when turn data is received"""
        if not event.transcript.strip():
            return
            
        self.turns_received += 1
        
        # Count words
        words = event.transcript.split()
        self.words_transcribed += len(words)
        
        # Store transcript
        transcript_info = {
            'text': event.transcript,
            'end_of_turn': event.end_of_turn,
            'formatted': event.turn_is_formatted,
            'turn_order': event.turn_order,
            'confidence': getattr(event, 'end_of_turn_confidence', None),
            'timestamp': time.time()
        }
        
        if event.end_of_turn:
            self.transcripts.append(transcript_info)
        
        # Show transcript with status
        status = "FINAL" if event.end_of_turn else "PARTIAL"
        formatted_status = "FORMATTED" if event.turn_is_formatted else "RAW"
        confidence = f"{event.end_of_turn_confidence:.2f}" if hasattr(event, 'end_of_turn_confidence') else "N/A"
        
        print(f"🎯 [{status}] {event.transcript}")
        print(f"   📊 Turn #{event.turn_order} | {formatted_status} | Confidence: {confidence} | Words: {len(words)}")
        
        # For Spanish, show if we detect Spanish words
        spanish_indicators = ['es', 'la', 'el', 'de', 'que', 'y', 'en', 'un', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'una', 'estar', 'tener', 'hacer', 'poder', 'decir', 'todo', 'gran', 'otro', 'tiempo', 'muy', 'cuando', 'mucho', 'como', 'gobierno', 'también', 'trabajo', 'vida', 'mundo', 'año', 'día', 'entre', 'tanto', 'hasta', 'país', 'millón', 'nacional', 'sobre', 'español', 'sí', 'hola', 'gracias', 'por favor', 'buenos días']
        
        text_lower = event.transcript.lower()
        spanish_words_found = [word for word in spanish_indicators if word in text_lower]
        if spanish_words_found and event.end_of_turn:
            print(f"   🇪🇸 Spanish indicators detected: {', '.join(spanish_words_found[:5])}")

    def on_terminated(self, client, event: TerminationEvent):
        """Called when session terminates"""
        duration = event.audio_duration_seconds
        print(f"\n🔌 Session terminated: {duration:.1f} seconds of audio processed")

    def on_error(self, client, error: StreamingError):
        """Called on errors"""
        print(f"❌ Streaming error: {error}")

    def create_client(self):
        """Create the streaming client"""
        try:
            self.client = StreamingClient(
                StreamingClientOptions(
                    api_key=ASSEMBLY_API_KEY,
                    api_host="streaming.assemblyai.com",
                )
            )
            
            # Register event handlers
            self.client.on(StreamingEvents.Begin, self.on_begin)
            self.client.on(StreamingEvents.Turn, self.on_turn)
            self.client.on(StreamingEvents.Termination, self.on_terminated)
            self.client.on(StreamingEvents.Error, self.on_error)
            
            print("✅ Streaming client created")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create client: {e}")
            return False

    def start_streaming(self):
        """Start the streaming session"""
        try:
            print("🔄 Connecting to Universal Streaming API...")
            
            # Use the audio file's sample rate if available
            sample_rate = self.audio_stream.sample_rate if self.audio_stream and self.audio_stream.sample_rate else 16000
            
            # Connect with parameters optimized for Spanish transcription
            self.client.connect(
                StreamingParameters(
                    sample_rate=sample_rate,
                    format_turns=True,  # Enable text formatting
                    # Optimized for better Spanish detection
                    min_end_of_turn_silence_when_confident=300,  # Slightly longer for Spanish
                    max_turn_silence=2500,  # Allow for Spanish speech patterns
                    end_of_turn_confidence_threshold=0.5,  # Lower threshold for better detection
                )
            )
            
            print("✅ Connected to Universal Streaming")
            print(f"📊 Using sample rate: {sample_rate} Hz")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_test(self):
        """Run the Spanish audio test"""
        # Create audio file stream
        self.audio_stream = AudioFileStream(self.audio_file_path)
        
        # Create client
        if not self.create_client():
            return False
            
        # Start streaming
        if not self.start_streaming():
            return False
            
        try:
            # Start audio file streaming
            if not self.audio_stream.start_streaming():
                print("❌ Failed to start audio file streaming")
                return False
                
            print("🔄 Streaming Spanish audio to AssemblyAI...")
            
            # Stream the audio file to AssemblyAI
            self.client.stream(self.audio_stream)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            self.cleanup()
            
        return True

    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        if self.client:
            try:
                self.client.disconnect(terminate=True)
                print("✅ Disconnected from streaming service")
            except:
                pass
                
        if self.audio_stream:
            self.audio_stream.close()
        
        # Show final results
        if self.session_start_time:
            elapsed = time.time() - self.session_start_time
            print(f"\n📊 Final Results:")
            print(f"   ⏱️  Duration: {elapsed:.1f}s")
            print(f"   🎯 Total Turns: {self.turns_received}")
            print(f"   🔤 Words Transcribed: {self.words_transcribed}")
            if elapsed > 0:
                print(f"   📈 Throughput: {self.words_transcribed/elapsed:.1f} words/second")
            
            # Show final transcripts
            if self.transcripts:
                print(f"\n📝 Final Spanish Transcripts:")
                print("-" * 50)
                for i, transcript in enumerate(self.transcripts, 1):
                    confidence_str = f" (confidence: {transcript['confidence']:.2f})" if transcript['confidence'] else ""
                    print(f"{i}. {transcript['text']}{confidence_str}")
                print("-" * 50)
            else:
                print("\n⚠️  No final transcripts received")

def signal_handler(signum, frame):
    print("\n🛑 Interrupted")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting AssemblyAI Spanish Audio Streaming Test...")
    
    # Check if audio file exists
    import os
    if not os.path.exists(AUDIO_FILE_PATH):
        print(f"❌ Audio file not found: {AUDIO_FILE_PATH}")
        print("Please ensure the Spanish audio file exists at the specified path.")
        return 1
    
    tester = SpanishAudioTester(AUDIO_FILE_PATH)
    success = tester.run_test()
    
    if not success:
        print("❌ Test failed")
        return 1
        
    print("👋 Spanish audio test completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
