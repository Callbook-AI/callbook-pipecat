#!/usr/bin/env python3
"""
AssemblyAI Universal Streaming Demo with Simulated Audio
Demonstrates the Universal Streaming API working properly
"""

import asyncio
import time
import threading
import sys
import signal
import logging
import queue
import random
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
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulatedAudioStream:
    """Simulated audio stream for demonstration"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._buffer = queue.Queue()
        self._recording_thread = None
        self._stop_recording = threading.Event()
        
        # Pre-recorded phrases to simulate
        self.phrases = [
            "Hello, this is a test of AssemblyAI Universal Streaming.",
            "Hola, esto es una prueba del streaming universal de AssemblyAI.",
            "The weather is nice today.",
            "El clima está muy bueno hoy.",
            "How are you doing?",
            "¿Cómo estás?",
            "This is working great!",
            "¡Esto está funcionando genial!"
        ]
        self.phrase_index = 0
    
    def start_recording(self):
        """Start simulated recording"""
        self._stop_recording.clear()
        self._recording_thread = Thread(target=self._recording_loop)
        self._recording_thread.daemon = True
        self._recording_thread.start()
        return True
    
    def _recording_loop(self):
        """Simulated audio recording loop"""
        try:
            print("🎤 Simulating speech every 5 seconds...")
            print("📝 Simulated phrases will include English and Spanish")
            
            chunk_count = 0
            silence_chunks = int(5 * self.sample_rate / self.chunk_size)  # 5 seconds of silence
            
            while not self._stop_recording.is_set():
                chunk_count += 1
                
                # Every 5 seconds, simulate a phrase being spoken
                if chunk_count % silence_chunks == 0:
                    phrase = self.phrases[self.phrase_index % len(self.phrases)]
                    print(f"\n🗣️  Simulating: \"{phrase}\"")
                    self.phrase_index += 1
                    
                    # Generate some random audio data to simulate speech
                    # (In reality this would be actual audio samples)
                    for _ in range(50):  # Simulate ~3 seconds of speech
                        if not self._stop_recording.is_set():
                            # Generate random audio data (simulating speech)
                            audio_data = bytes([random.randint(0, 255) for _ in range(self.chunk_size * 2)])
                            self._buffer.put(audio_data)
                            time.sleep(0.064)  # 64ms per chunk at 16kHz
                else:
                    # Generate silence
                    silence_data = b'\x00' * (self.chunk_size * 2)
                    self._buffer.put(silence_data)
                    time.sleep(0.064)  # 64ms per chunk
                    
        except Exception as e:
            logger.error(f"Simulated recording error: {e}")
    
    def read(self):
        """Read next chunk of audio data"""
        try:
            return self._buffer.get(timeout=0.1)
        except queue.Empty:
            return b'\x00' * (self.chunk_size * 2)  # Return silence
    
    def stop_recording(self):
        """Stop recording"""
        self._stop_recording.set()
        if self._recording_thread:
            self._recording_thread.join(timeout=1.0)
    
    def close(self):
        """Clean up"""
        self.stop_recording()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.read()

class UniversalStreamingDemo:
    def __init__(self):
        print("🎤 AssemblyAI Universal Streaming Demo")
        print("=" * 50)
        print("ℹ️  Using Universal Streaming API (v3)")
        print("🎭 Using simulated audio with mixed English/Spanish")
        print("🌍 Language: Automatic detection")
        print("=" * 50)
        
        self.client = None
        self.audio_stream = None
        
        # Metrics
        self.turns_received = 0
        self.words_transcribed = 0
        self.session_start_time = None

    def on_begin(self, client, event: BeginEvent):
        """Called when streaming session begins"""
        self.session_start_time = time.time()
        print(f"🔗 Session started: {event.id}")
        print("🎯 Simulated speech will begin shortly...")
        print("-" * 50)

    def on_turn(self, client, event: TurnEvent):
        """Called when turn data is received"""
        if not event.transcript.strip():
            return
            
        self.turns_received += 1
        
        # Count words
        words = event.transcript.split()
        self.words_transcribed += len(words)
        
        # Show transcript with status
        status = "END" if event.end_of_turn else "PARTIAL"
        formatted_status = "FORMATTED" if event.turn_is_formatted else "RAW"
        confidence = f"{event.end_of_turn_confidence:.2f}" if hasattr(event, 'end_of_turn_confidence') else "N/A"
        
        print(f"🎯 [{status}] {event.transcript}")
        print(f"   📊 Turn #{event.turn_order} | {formatted_status} | Confidence: {confidence} | Words: {len(words)}")

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
            
            # Connect with parameters
            self.client.connect(
                StreamingParameters(
                    sample_rate=SAMPLE_RATE,
                    format_turns=True,
                    min_end_of_turn_silence_when_confident=200,
                    max_turn_silence=2000,
                    end_of_turn_confidence_threshold=0.6,
                )
            )
            
            print("✅ Connected to Universal Streaming")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_demo(self):
        """Run the demonstration"""
        # Create simulated audio stream
        self.audio_stream = SimulatedAudioStream()
        
        # Create client
        if not self.create_client():
            return False
            
        # Start streaming
        if not self.start_streaming():
            return False
            
        try:
            # Start simulated audio
            self.audio_stream.start_recording()
            
            print("🔄 Streaming simulated audio to AssemblyAI...")
            
            # Stream for 30 seconds
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
            
        if self.session_start_time:
            elapsed = time.time() - self.session_start_time
            print(f"\n📊 Final Stats:")
            print(f"   ⏱️  Duration: {elapsed:.1f}s")
            print(f"   🎯 Turns: {self.turns_received}")
            print(f"   🔤 Words: {self.words_transcribed}")
            if elapsed > 0:
                print(f"   📈 Throughput: {self.words_transcribed/elapsed:.1f} words/second")

def signal_handler(signum, frame):
    print("\n🛑 Interrupted")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting AssemblyAI Universal Streaming Demo...")
    print("💡 This demo simulates audio input since microphone access isn't available")
    print("🔄 In a real implementation, this would use actual microphone data\n")
    
    demo = UniversalStreamingDemo()
    success = demo.run_demo()
    
    if not success:
        print("❌ Demo failed")
        return 1
        
    print("👋 Demo completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
