#!/usr/bin/env python3
"""
AssemblyAI Universal Streaming Test (v3 API)
Uses the new Universal Streaming API with automatic language detection
Note: Currently optimized for English, Spanish may work with auto-detection
"""

import asyncio
import time
import threading
import sys
import signal
import logging

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

try:
    import pyaudio
except ImportError:
    print("❌ Error: PyAudio not installed. Run: pip install pyaudio")
    sys.exit(1)

# Configuration
ASSEMBLY_API_KEY = "50eb210feb72457593d3f811a9f2a698"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalStreamingTester:
    def __init__(self):
        print("🎤 AssemblyAI Universal Streaming Test")
        print("=" * 50)
        print("ℹ️  Using Universal Streaming API (v3)")
        print("🌍 Language: Automatic detection (optimized for English, may detect Spanish)")
        print("=" * 50)
        
        self.client = None
        self.audio = None
        self.stream = None
        self.is_running = False
        
        # Metrics
        self.turns_received = 0
        self.words_transcribed = 0
        self.session_start_time = None

    def setup_audio(self):
        """Setup audio input"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Find first available input device
            input_device = None
            print("\n🔍 Available audio devices:")
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    print(f"   {i}: {info['name']} ({info['maxInputChannels']} ch)")
                    if input_device is None:
                        input_device = i
            
            if input_device is None:
                print("❌ No audio input device found!")
                return False
            
            device_info = self.audio.get_device_info_by_index(input_device)
            print(f"🎯 Using device: {device_info['name']}")
            
            # Open audio stream
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=input_device,
                frames_per_buffer=CHUNK_SIZE,
            )
            
            print("✅ Audio stream ready")
            return True
            
        except Exception as e:
            print(f"❌ Audio setup failed: {e}")
            return False

    def on_begin(self, client, event: BeginEvent):
        """Called when streaming session begins"""
        self.session_start_time = time.time()
        print(f"🔗 Session started: {event.id}")
        print("🎯 Speak now - transcription will appear below:")
        print("💡 Try speaking in Spanish to test language detection")
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
        
        # Auto-format if end of turn and not formatted
        if event.end_of_turn and not event.turn_is_formatted:
            print("🔄 Requesting formatting...")

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
            
            # Connect with parameters optimized for real-time transcription
            self.client.connect(
                StreamingParameters(
                    sample_rate=SAMPLE_RATE,
                    format_turns=True,  # Enable text formatting
                    # Optimize for responsiveness
                    min_end_of_turn_silence_when_confident=200,  # Reduced from default 160ms
                    max_turn_silence=2000,  # Reduced from default 2400ms  
                    end_of_turn_confidence_threshold=0.6,  # Slightly lower threshold
                )
            )
            
            print("✅ Connected to Universal Streaming")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            import traceback
            traceback.print_exc()
            return False

    def audio_loop(self):
        """Audio capture and streaming loop"""
        print("🎤 Starting audio capture...")
        
        try:
            # Create microphone stream using AssemblyAI's helper
            # Use default parameters for MicrophoneStream
            mic_stream = aai.extras.MicrophoneStream(
                sample_rate=SAMPLE_RATE
            )
            
            print("🔄 Streaming audio...")
            self.client.stream(mic_stream)
            
        except Exception as e:
            print(f"❌ Audio streaming error: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main execution"""
        # Setup audio (for device info, but we'll use AssemblyAI's mic stream)
        if not self.setup_audio():
            return False
            
        # Create client
        if not self.create_client():
            return False
            
        # Start streaming
        if not self.start_streaming():
            return False
            
        # Run audio streaming
        try:
            self.audio_loop()
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
                
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if self.audio:
            self.audio.terminate()
            
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
    
    print("🚀 Starting AssemblyAI Universal Streaming Test...")
    
    tester = UniversalStreamingTester()
    success = tester.run()
    
    if not success:
        print("❌ Test failed")
        return 1
        
    print("👋 Test completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
