#!/usr/bin/env python3
"""
AssemblyAI Universal Streaming Test with Custom Microphone (v3 API)
Uses the new Universal Streaming API with automatic language detection
Includes robust PyAudio device selection and streaming logic
"""

import asyncio
import time
import threading
import sys
import signal
import logging
import queue
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

class CustomMicrophoneStream:
    """Custom microphone stream with robust device selection"""
    
    def __init__(self, sample_rate=SAMPLE_RATE, chunk_size=CHUNK_SIZE, channels=CHANNELS):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio = None
        self.stream = None
        self._buffer = queue.Queue()
        self._recording_thread = None
        self._stop_recording = threading.Event()
    
    def setup_audio(self):
        """Setup audio input with robust device selection"""
        try:
            self.audio = pyaudio.PyAudio()
            
            print("\n🔍 Available audio input devices:")
            available_devices = []
            default_device = None
            
            for i in range(self.audio.get_device_count()):
                try:
                    info = self.audio.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        print(f"   {i}: {info['name']} ({info['maxInputChannels']} channels) - Rate: {int(info['defaultSampleRate'])}")
                        available_devices.append((i, info))
                        
                        # Check if this is a good default (prefer built-in, avoid Teams)
                        if default_device is None and "teams" not in info['name'].lower():
                            if "built-in" in info['name'].lower() or "default" in info['name'].lower() or "microphone" in info['name'].lower():
                                default_device = i
                except:
                    continue
            
            if not available_devices:
                print("❌ No audio input devices found!")
                return False
            
            # Try to find the best device to use
            selected_device = None
            
            # First, try the system default
            try:
                default_info = self.audio.get_default_input_device_info()
                selected_device = default_info['index']
                print(f"\n🎯 Using system default device: {default_info['name']}")
            except:
                # If no system default, use our detected default or first available
                if default_device is not None:
                    selected_device = default_device
                    device_info = self.audio.get_device_info_by_index(selected_device)
                    print(f"\n🎯 Using detected default device: {device_info['name']}")
                else:
                    # Use first available device (even if it's Teams Audio)
                    selected_device = available_devices[0][0]
                    device_info = available_devices[0][1]
                    print(f"\n🎯 Using first available device: {device_info['name']}")
                    print("⚠️  Note: Using Microsoft Teams Audio - make sure your mic is not muted in Teams")
            
            # Test the selected device with supported parameters
            device_info = self.audio.get_device_info_by_index(selected_device)
            device_sample_rate = int(device_info['defaultSampleRate'])
            
            # Try different configurations
            configs_to_try = [
                (self.sample_rate, selected_device),  # Preferred config
                (device_sample_rate, selected_device),  # Device native rate
                (44100, selected_device),  # Common rate
                (48000, selected_device),  # Another common rate
            ]
            
            for sample_rate, device_id in configs_to_try:
                try:
                    print(f"🔧 Trying device {device_id} at {sample_rate} Hz...")
                    
                    # Test if we can open the stream
                    test_stream = self.audio.open(
                        format=FORMAT,
                        channels=self.channels,
                        rate=sample_rate,
                        input=True,
                        input_device_index=device_id,
                        frames_per_buffer=self.chunk_size,
                        stream_callback=None
                    )
                    test_stream.close()  # Close test stream
                    
                    # If we get here, it worked! Open the real stream
                    self.stream = self.audio.open(
                        format=FORMAT,
                        channels=self.channels,
                        rate=sample_rate,
                        input=True,
                        input_device_index=device_id,
                        frames_per_buffer=self.chunk_size,
                        stream_callback=None
                    )
                    
                    # Update sample rate if we had to change it
                    if sample_rate != self.sample_rate:
                        print(f"📊 Adjusted sample rate from {self.sample_rate} to {sample_rate} Hz")
                        self.sample_rate = sample_rate
                    
                    print(f"✅ Audio stream initialized successfully!")
                    print(f"   Device: {device_info['name']}")
                    print(f"   Sample Rate: {sample_rate} Hz")
                    print(f"   Channels: {self.channels}")
                    print(f"   Format: 16-bit PCM")
                    return True
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    continue
            
            print("❌ Could not initialize audio with any configuration!")
            return False
            
        except Exception as e:
            print(f"❌ Error setting up audio: {e}")
            return False
    
    def start_recording(self):
        """Start recording audio in a separate thread"""
        if not self.stream:
            return False
            
        self._stop_recording.clear()
        self._recording_thread = Thread(target=self._recording_loop)
        self._recording_thread.daemon = True
        self._recording_thread.start()
        return True
    
    def _recording_loop(self):
        """Audio recording loop"""
        try:
            chunk_count = 0
            while not self._stop_recording.is_set():
                if self.stream.is_active():
                    try:
                        # Read audio data
                        data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                        if data:
                            self._buffer.put(data)
                            chunk_count += 1
                            # Print progress every 100 chunks (about every 6 seconds at 16kHz)
                            if chunk_count % 100 == 0:
                                print(f"🎤 Audio chunks captured: {chunk_count}")
                    except Exception as e:
                        logger.warning(f"Audio read error: {e}")
                        break
        except Exception as e:
            logger.error(f"Recording loop error: {e}")
    
    def read(self):
        """Read next chunk of audio data"""
        try:
            # Use a timeout to avoid blocking indefinitely
            return self._buffer.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def stop_recording(self):
        """Stop recording"""
        self._stop_recording.set()
        if self._recording_thread:
            self._recording_thread.join(timeout=1.0)
    
    def close(self):
        """Clean up audio resources"""
        self.stop_recording()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()
    
    def __iter__(self):
        """Iterator interface for streaming"""
        return self
    
    def __next__(self):
        """Iterator interface - get next audio chunk"""
        data = self.read()
        if data is None:
            # Return silence if no data available (keeps stream alive)
            return b'\x00' * (self.chunk_size * 2)  # 2 bytes per sample for 16-bit
        return data

class UniversalStreamingTester:
    def __init__(self):
        print("🎤 AssemblyAI Universal Streaming Test (Custom Microphone)")
        print("=" * 60)
        print("ℹ️  Using Universal Streaming API (v3)")
        print("🎙️ Using custom PyAudio microphone with robust device selection")
        print("🌍 Language: Automatic detection (optimized for English, may detect Spanish)")
        print("=" * 60)
        
        self.client = None
        self.microphone = None
        self.is_running = False
        
        # Metrics
        self.turns_received = 0
        self.words_transcribed = 0
        self.session_start_time = None

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
                    sample_rate=self.microphone.sample_rate,
                    format_turns=True,  # Enable text formatting
                    # Optimize for responsiveness
                    min_end_of_turn_silence_when_confident=200,  # Reduced from default
                    max_turn_silence=2000,  # Reduced from default
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
            # Start recording
            if not self.microphone.start_recording():
                print("❌ Failed to start recording")
                return
                
            print("🔄 Streaming audio to AssemblyAI...")
            
            # Stream the microphone data to AssemblyAI
            self.client.stream(self.microphone)
            
        except Exception as e:
            print(f"❌ Audio streaming error: {e}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main execution"""
        # Setup custom microphone
        self.microphone = CustomMicrophoneStream()
        if not self.microphone.setup_audio():
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
                
        if self.microphone:
            self.microphone.close()
            
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
    
    print("🚀 Starting AssemblyAI Universal Streaming Test with Custom Microphone...")
    
    tester = UniversalStreamingTester()
    success = tester.run()
    
    if not success:
        print("❌ Test failed")
        return 1
        
    print("👋 Test completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
