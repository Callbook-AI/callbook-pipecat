#!/usr/bin/env python3
"""
AssemblyAI Real-time Transcription Test Script
This script tests AssemblyAI's real-time transcription capabilities in Spanish
with microphone input and performance metrics.
"""

import asyncio
import time
import threading
from typing import Dict, List
from dataclasses import dataclass
from collections import deque
import statistics
import sys
import signal

try:
    import assemblyai as aai
    from assemblyai import AudioEncoding
except ImportError:
    print("❌ Error: AssemblyAI not installed. Run: pip install assemblyai")
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

@dataclass
class TranscriptMetrics:
    """Metrics for tracking transcription performance"""
    interim_count: int = 0
    final_count: int = 0
    response_times: List[float] = None
    word_counts: List[int] = None
    confidence_scores: List[float] = None
    start_time: float = None
    
    def __post_init__(self):
        if self.response_times is None:
            self.response_times = []
        if self.word_counts is None:
            self.word_counts = []
        if self.confidence_scores is None:
            self.confidence_scores = []
        if self.start_time is None:
            self.start_time = time.time()

class AssemblyAITester:
    def __init__(self):
        self.transcriber = None
        self.audio = None
        self.stream = None
        self.is_running = False
        self.metrics = TranscriptMetrics()
        self.audio_chunks_sent = 0
        self.last_audio_time = None
        self.session_start_time = None
        self.sample_rate = SAMPLE_RATE  # Store as instance variable
        
        # Setup AssemblyAI
        aai.settings.api_key = ASSEMBLY_API_KEY
        
        # Recent transcripts for display
        self.recent_transcripts = deque(maxlen=10)
        
        print("🎤 AssemblyAI Real-time Spanish Transcription Test")
        print("=" * 60)
        print(f"📊 Configuration:")
        print(f"   • Language: Spanish (es)")
        print(f"   • Sample Rate: {self.sample_rate} Hz")
        print(f"   • Chunk Size: {CHUNK_SIZE} samples")
        print(f"   • Audio Format: 16-bit PCM")
        print(f"   • API Key: {ASSEMBLY_API_KEY[:20]}...")
        print("=" * 60)
        
        # macOS permission reminder
        import platform
        if platform.system() == "Darwin":
            print("🍎 macOS Note: Make sure Terminal has microphone permissions!")
            print("   Go to: System Preferences > Security & Privacy > Privacy > Microphone")
            print("   Enable access for Terminal or your terminal app.")
            print("=" * 60)

    def setup_audio(self):
        """Initialize PyAudio for microphone input"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # List available input devices
            print("\n🔍 Available audio input devices:")
            available_devices = []
            default_device = None
            
            for i in range(self.audio.get_device_count()):
                try:
                    info = self.audio.get_device_info_by_index(i)
                    if info['maxInputChannels'] > 0:
                        print(f"   {i}: {info['name']} ({info['maxInputChannels']} channels) - Rate: {int(info['defaultSampleRate'])}")
                        available_devices.append((i, info))
                        
                        # Check if this is a good default (not Microsoft Teams)
                        if default_device is None and "teams" not in info['name'].lower():
                            if "built-in" in info['name'].lower() or "default" in info['name'].lower():
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
                    selected_device = available_devices[0][0]
                    device_info = available_devices[0][1]
                    print(f"\n🎯 Using first available device: {device_info['name']}")
            
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
                        channels=CHANNELS,
                        rate=sample_rate,
                        input=True,
                        input_device_index=device_id,
                        frames_per_buffer=CHUNK_SIZE,
                        stream_callback=None
                    )
                    test_stream.close()  # Close test stream
                    
                    # If we get here, it worked! Open the real stream
                    self.stream = self.audio.open(
                        format=FORMAT,
                        channels=CHANNELS,
                        rate=sample_rate,
                        input=True,
                        input_device_index=device_id,
                        frames_per_buffer=CHUNK_SIZE,
                        stream_callback=None
                    )
                    
                    # Update sample rate if we had to change it
                    if sample_rate != self.sample_rate:
                        print(f"📊 Adjusted sample rate from {self.sample_rate} to {sample_rate} Hz")
                        self.sample_rate = sample_rate
                    
                    print(f"✅ Audio stream initialized successfully!")
                    print(f"   Device: {device_info['name']}")
                    print(f"   Sample Rate: {sample_rate} Hz")
                    print(f"   Channels: {CHANNELS}")
                    print(f"   Format: 16-bit PCM")
                    return True
                    
                except Exception as e:
                    print(f"   ❌ Failed: {e}")
                    continue
            
            print("❌ Could not initialize audio with any configuration!")
            return False
            
        except Exception as e:
            print(f"❌ Error setting up audio: {e}")
            if self.audio:
                self.audio.terminate()
            return False

    def on_open(self, session_opened: aai.RealtimeSessionOpened):
        """Callback when AssemblyAI connection opens"""
        self.session_start_time = time.time()
        print(f"\n🔗 Connected to AssemblyAI!")
        print(f"   • Session ID: {session_opened.session_id}")
        print(f"   • Connection time: {time.strftime('%H:%M:%S')}")
        print(f"\n🎯 Ready to transcribe! Start speaking in Spanish...")
        print(f"   Press Ctrl+C to stop")
        print("-" * 60)

    def on_data(self, transcript: aai.RealtimeTranscript):
        """Callback for transcription results"""
        if not transcript.text:
            return
            
        current_time = time.time()
        
        # Calculate response time (time since last audio chunk)
        if self.last_audio_time:
            response_time = current_time - self.last_audio_time
            self.metrics.response_times.append(response_time)
        
        # Count words
        word_count = len(transcript.text.split())
        self.metrics.word_counts.append(word_count)
        
        if isinstance(transcript, aai.RealtimeFinalTranscript):
            self.metrics.final_count += 1
            confidence = getattr(transcript, 'confidence', 0.0)
            self.metrics.confidence_scores.append(confidence)
            
            # Store recent transcript
            self.recent_transcripts.append({
                'text': transcript.text,
                'type': 'FINAL',
                'confidence': confidence,
                'words': word_count,
                'time': time.strftime('%H:%M:%S')
            })
            
            # Display final transcript with metrics
            print(f"\n🎯 FINAL: '{transcript.text}'")
            print(f"   📊 Words: {word_count} | Confidence: {confidence:.2f} | Time: {time.strftime('%H:%M:%S')}")
            
        else:  # Interim transcript
            self.metrics.interim_count += 1
            
            # Display interim with less detail
            print(f"💭 interim: '{transcript.text}' ({word_count} words)")
        
        # Show real-time metrics every 10 transcripts
        total_transcripts = self.metrics.interim_count + self.metrics.final_count
        if total_transcripts > 0 and total_transcripts % 10 == 0:
            self.display_metrics()

    def on_error(self, error: aai.RealtimeError):
        """Callback for errors"""
        print(f"\n❌ AssemblyAI Error: {error}")

    def on_close(self):
        """Callback when connection closes"""
        print(f"\n🔌 Disconnected from AssemblyAI")

    def start_transcription(self):
        """Start the real-time transcription"""
        try:
            print(f"🔄 Connecting to AssemblyAI Universal Streaming with sample rate: {self.sample_rate} Hz...")
            print("🔄 Configuring for Spanish language streaming...")
            
            # Use the new Universal Streaming API
            self.transcriber = aai.RealtimeTranscriber(
                sample_rate=self.sample_rate,
                on_data=self.on_data,
                on_error=self.on_error,
                on_open=self.on_open,
                on_close=self.on_close,
                # Language will be detected automatically with universal model
                end_utterance_silence_threshold=700,  # ms
                disable_partial_transcripts=False,    # Enable interim results
            )
            
            print("🔗 Establishing connection with Universal Streaming API...")
            self.transcriber.connect()
            return True
            
        except Exception as e:
            print(f"❌ Error starting transcription: {e}")
            print(f"ℹ️  Trying with minimal configuration...")
            
            try:
                # Minimal configuration approach
                self.transcriber = aai.RealtimeTranscriber(
                    sample_rate=self.sample_rate,
                    on_data=self.on_data,
                    on_error=self.on_error,
                    on_open=self.on_open,
                    on_close=self.on_close,
                )
                
                self.transcriber.connect()
                print("✅ Connected with minimal configuration")
                return True
                
            except Exception as e2:
                print(f"❌ Minimal config also failed: {e2}")
                
                # Last attempt with just basic parameters
                try:
                    print("🔄 Final attempt with basic setup...")
                    self.transcriber = aai.RealtimeTranscriber(
                        on_data=self.on_data,
                        on_error=self.on_error,
                        on_open=self.on_open,
                        on_close=self.on_close,
                    )
                    
                    self.transcriber.connect()
                    print("✅ Connected with basic configuration")
                    return True
                    
                except Exception as e3:
                    print(f"❌ All connection attempts failed: {e3}")
                    import traceback
                    traceback.print_exc()
                    return False

    def audio_loop(self):
        """Audio capture loop running in separate thread"""
        print("🎤 Starting audio capture...")
        
        while self.is_running:
            try:
                # Read audio chunk
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                if self.transcriber and data:
                    # Send to AssemblyAI
                    self.transcriber.stream(data)
                    self.audio_chunks_sent += 1
                    self.last_audio_time = time.time()
                    
                    # Log every 100 chunks (about every 3 seconds at 1024 samples/16kHz)
                    if self.audio_chunks_sent % 100 == 0:
                        elapsed = time.time() - self.session_start_time if self.session_start_time else 0
                        print(f"📡 Audio chunks sent: {self.audio_chunks_sent} | Session time: {elapsed:.1f}s")
                        
            except Exception as e:
                if self.is_running:  # Only log if we're supposed to be running
                    print(f"⚠️ Audio capture error: {e}")
                break

    def display_metrics(self):
        """Display current performance metrics"""
        print("\n" + "=" * 60)
        print("📊 REAL-TIME METRICS")
        print("=" * 60)
        
        # Session duration
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        print(f"⏱️  Session Duration: {session_duration:.1f}s")
        print(f"📡 Audio Chunks Sent: {self.audio_chunks_sent}")
        
        # Transcript counts
        print(f"📝 Transcripts: {self.metrics.final_count} final, {self.metrics.interim_count} interim")
        
        # Response times
        if self.metrics.response_times:
            avg_response = statistics.mean(self.metrics.response_times)
            min_response = min(self.metrics.response_times)
            max_response = max(self.metrics.response_times)
            print(f"⚡ Response Time: {avg_response:.3f}s avg (min: {min_response:.3f}s, max: {max_response:.3f}s)")
        
        # Word counts
        if self.metrics.word_counts:
            total_words = sum(self.metrics.word_counts)
            avg_words = statistics.mean(self.metrics.word_counts)
            print(f"🔤 Words: {total_words} total, {avg_words:.1f} avg per transcript")
        
        # Confidence scores
        if self.metrics.confidence_scores:
            avg_confidence = statistics.mean(self.metrics.confidence_scores)
            print(f"🎯 Average Confidence: {avg_confidence:.2f}")
        
        # Throughput
        if session_duration > 0:
            transcripts_per_minute = (self.metrics.final_count / session_duration) * 60
            print(f"📈 Throughput: {transcripts_per_minute:.1f} final transcripts/minute")
        
        print("=" * 60)

    def display_recent_transcripts(self):
        """Display recent transcripts"""
        if not self.recent_transcripts:
            return
            
        print("\n📋 RECENT TRANSCRIPTS:")
        print("-" * 60)
        for i, transcript in enumerate(list(self.recent_transcripts)[-5:], 1):
            print(f"{i}. [{transcript['time']}] {transcript['type']}: '{transcript['text']}'")
            print(f"   📊 {transcript['words']} words, confidence: {transcript.get('confidence', 'N/A')}")
        print("-" * 60)

    async def run(self):
        """Main execution loop"""
        # Setup audio
        if not self.setup_audio():
            return
        
        # Start transcription
        if not self.start_transcription():
            return
        
        # Wait for connection
        await asyncio.sleep(2)
        
        # Start audio capture in separate thread
        self.is_running = True
        audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        audio_thread.start()
        
        try:
            # Main loop - display metrics periodically
            while self.is_running:
                await asyncio.sleep(10)  # Update every 10 seconds
                self.display_metrics()
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping transcription...")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        
        if self.transcriber:
            self.transcriber.close()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("\n" + "=" * 60)
        print("🏁 FINAL SUMMARY")
        print("=" * 60)
        
        session_duration = time.time() - self.session_start_time if self.session_start_time else 0
        print(f"⏱️  Total Session Time: {session_duration:.1f}s")
        print(f"📡 Total Audio Chunks: {self.audio_chunks_sent}")
        print(f"📝 Final Transcripts: {self.metrics.final_count}")
        print(f"💭 Interim Transcripts: {self.metrics.interim_count}")
        
        if self.metrics.response_times:
            avg_response = statistics.mean(self.metrics.response_times)
            print(f"⚡ Average Response Time: {avg_response:.3f}s")
        
        if self.metrics.confidence_scores:
            avg_confidence = statistics.mean(self.metrics.confidence_scores)
            print(f"🎯 Average Confidence: {avg_confidence:.2f}")
        
        total_words = sum(self.metrics.word_counts) if self.metrics.word_counts else 0
        print(f"🔤 Total Words Transcribed: {total_words}")
        
        self.display_recent_transcripts()
        
        print("\n✅ Test completed successfully!")
        print("=" * 60)

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print(f"\n🛑 Received signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Setup signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚀 Starting AssemblyAI Real-time Transcription Test...")
    
    # Create and run tester
    tester = AssemblyAITester()
    await tester.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
