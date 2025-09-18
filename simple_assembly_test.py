#!/usr/bin/env python3
"""
AssemblyAI Simple Spanish Streaming Test
Updated for current Universal Streaming API
"""

import asyncio
import time
import threading
import statistics
import sys
import signal

try:
    import assemblyai as aai
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

class SimpleAssemblyTester:
    def __init__(self):
        print("🎤 AssemblyAI Spanish Streaming Test")
        print("=" * 50)
        
        # Set API key
        aai.settings.api_key = ASSEMBLY_API_KEY
        
        self.transcriber = None
        self.audio = None
        self.stream = None
        self.is_running = False
        
        # Metrics
        self.final_transcripts = 0
        self.interim_transcripts = 0
        self.start_time = None
        
        print("✅ Configured for Spanish streaming")

    def setup_audio(self):
        """Setup audio input"""
        try:
            self.audio = pyaudio.PyAudio()
            
            # Find first available input device
            input_device = None
            for i in range(self.audio.get_device_count()):
                info = self.audio.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    input_device = i
                    print(f"🎯 Using device: {info['name']}")
                    break
            
            if input_device is None:
                print("❌ No audio input device found!")
                return False
            
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

    def on_open(self, session_opened):
        """Called when connection opens"""
        self.start_time = time.time()
        print(f"🔗 Connected! Session: {session_opened.session_id}")
        print("🎯 Speak in Spanish - transcription will appear below:")
        print("-" * 50)

    def on_data(self, transcript):
        """Called when transcript data arrives"""
        if not transcript.text:
            return
            
        if transcript.message_type == "FinalTranscript":
            self.final_transcripts += 1
            print(f"🎯 FINAL: {transcript.text}")
        else:
            self.interim_transcripts += 1
            print(f"💭 interim: {transcript.text}")

    def on_error(self, error):
        """Called on errors"""
        print(f"❌ Error: {error}")

    def on_close(self):
        """Called when connection closes"""
        print("🔌 Disconnected")

    def start_streaming(self):
        """Start the streaming transcription"""
        try:
            print("🔄 Creating streaming connection...")
            
            # Create transcriber with minimal config
            self.transcriber = aai.RealtimeTranscriber(
                on_data=self.on_data,
                on_error=self.on_error,
                on_open=self.on_open,
                on_close=self.on_close,
                sample_rate=SAMPLE_RATE,
            )
            
            print("🔄 Connecting...")
            self.transcriber.connect()
            return True
            
        except Exception as e:
            print(f"❌ Failed to start streaming: {e}")
            return False

    def audio_loop(self):
        """Audio capture and streaming loop"""
        print("🎤 Starting audio capture...")
        
        while self.is_running:
            try:
                # Read audio data
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # Send to AssemblyAI
                if self.transcriber and data:
                    self.transcriber.stream(data)
                    
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ Audio error: {e}")
                break

    def run(self):
        """Main execution"""
        # Setup audio
        if not self.setup_audio():
            return False
            
        # Start streaming
        if not self.start_streaming():
            return False
            
        # Wait for connection
        time.sleep(2)
        
        # Start audio in thread
        self.is_running = True
        audio_thread = threading.Thread(target=self.audio_loop, daemon=True)
        audio_thread.start()
        
        try:
            # Keep running until interrupted
            while self.is_running:
                time.sleep(1)
                
                # Show stats every 10 seconds
                if self.start_time and (time.time() - self.start_time) % 10 < 1:
                    elapsed = time.time() - self.start_time
                    print(f"\n📊 Stats: {self.final_transcripts} final, {self.interim_transcripts} interim transcripts in {elapsed:.0f}s")
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            self.cleanup()
            
        return True

    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        
        if self.transcriber:
            try:
                self.transcriber.close()
            except:
                pass
                
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        if self.audio:
            self.audio.terminate()
            
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"\n✅ Session complete: {elapsed:.1f}s, {self.final_transcripts} final transcripts")

def signal_handler(signum, frame):
    print("\n🛑 Interrupted")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    tester = SimpleAssemblyTester()
    success = tester.run()
    
    if not success:
        print("❌ Test failed")
        return 1
        
    print("👋 Goodbye!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
