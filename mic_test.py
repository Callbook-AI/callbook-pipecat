#!/usr/bin/env python3
"""
Simple microphone test to check if we're getting audio input
"""

import pyaudio
import time
import sys
import struct

# Configuration
SAMPLE_RATE = 16000
CHUNK_SIZE = 1024
CHANNELS = 1
FORMAT = pyaudio.paInt16

def test_microphone():
    audio = pyaudio.PyAudio()
    
    print("🔍 Available audio input devices:")
    available_devices = []
    
    for i in range(audio.get_device_count()):
        try:
            info = audio.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                print(f"   {i}: {info['name']} ({info['maxInputChannels']} channels) - Rate: {int(info['defaultSampleRate'])}")
                available_devices.append((i, info))
        except:
            continue
    
    if not available_devices:
        print("❌ No audio input devices found!")
        return
    
    # Use first available device
    device_id, device_info = available_devices[0]
    print(f"\n🎯 Testing device: {device_info['name']}")
    
    try:
        # Open stream
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=device_id,
            frames_per_buffer=CHUNK_SIZE,
        )
        
        print("🎤 Listening for audio... (speak now)")
        print("📊 Audio levels will show below:")
        
        start_time = time.time()
        max_level = 0
        chunk_count = 0
        
        while time.time() - start_time < 10:  # Test for 10 seconds
            try:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # Convert to 16-bit integers and calculate volume (RMS)
                audio_data = struct.unpack(f'{CHUNK_SIZE}h', data)
                sum_squares = sum(sample ** 2 for sample in audio_data)
                volume = (sum_squares / len(audio_data)) ** 0.5
                max_level = max(max_level, volume)
                
                # Show visual level indicator
                level_bars = int(volume / 100)  # Scale down
                level_display = "█" * min(level_bars, 50)
                
                chunk_count += 1
                if chunk_count % 16 == 0:  # Update display every ~1 second
                    print(f"\r📊 Level: {level_display:<50} ({volume:.0f})    ", end="")
                    
            except Exception as e:
                print(f"❌ Read error: {e}")
                break
        
        stream.stop_stream()
        stream.close()
        audio.terminate()
        
        print(f"\n\n📈 Test Results:")
        print(f"   Duration: 10 seconds")
        print(f"   Chunks processed: {chunk_count}")
        print(f"   Max audio level: {max_level:.0f}")
        
        if max_level > 50:
            print("✅ Good audio levels detected - microphone is working!")
        elif max_level > 10:
            print("⚠️  Low audio levels - check microphone volume")
        else:
            print("❌ No significant audio detected - microphone may be muted or not working")
            
    except Exception as e:
        print(f"❌ Stream error: {e}")
        audio.terminate()

if __name__ == "__main__":
    test_microphone()
