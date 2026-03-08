# Deepgram Service Documentation

## Overview
This file (`src/pipecat/services/deepgram.py`) implements speech-to-text (STT) and text-to-speech (TTS) services using Deepgram API, with an intelligent backup system using Gladia.

## Main Components

### 1. **DeepgramTTSService**
Text-to-Speech service using Deepgram's Aura voices.

**Key Features:**
- Configurable voice selection (default: "aura-helios-en")
- Streaming audio generation
- TTFB (Time To First Byte) metrics tracking
- Error handling with fallback

**Main Methods:**
- `run_tts(text: str)` - Generates audio from text, yields `TTSAudioRawFrame` chunks

### 2. **DeepgramSiDetector**
Specialized detector for Spanish "si/sí" confirmation detection.

**Key Features:**
- Spanish-only transcription
- Pattern matching for "si"/"sí" variations
- Callback-based detection notification
- Deduplication using start times

**Main Methods:**
- `start()` - Opens Deepgram websocket connection
- `send_audio(chunk: bytes)` - Streams audio for detection
- `stop()` - Gracefully closes connection

### 3. **DeepgramGladiaDetector**
Intelligent backup STT system using Gladia's Solaria-1 model.

**Key Features:**
- Acts as backup when Deepgram fails or is slow
- Ultra-reliable with maximum sensitivity (low thresholds)
- VAD-based filtering to prevent false transcripts
- Coordinated timeout system with Deepgram
- Deduplication and similarity checking

**Configuration:**
- `confidence`: 0.1 (lower for max sensitivity)
- `endpointing`: 0.2 (more sensitive)
- `speech_threshold`: 0.3 (lower for max detection)
- `deepgram_wait_timeout`: 1.0s (reduced from 2.5s to minimize race conditions)
- `_vad_backup_window`: 1.5s (only allows backups within this window after VAD inactive)

**Main Methods:**
- `start()` - Initializes Gladia websocket
- `send_audio(chunk: bytes)` - Sends audio to backup service
- `notify_deepgram_final(transcript, timestamp)` - Cancels pending backups when Deepgram responds
- `update_vad_inactive_time(timestamp)` - Updates VAD reference for filtering
- `_process_backup_transcript()` - Handles backup transcript with intelligent coordination
- `_backup_timeout_handler()` - Activates backup if Deepgram doesn't respond in time

**Filtering Logic:**
- Ignores backups outside VAD window
- Checks similarity to recent Deepgram transcripts
- Prevents duplicates using transcript keys
- Suppresses transcripts that appear after bot starts speaking

### 4. **DeepgramSTTService** (Main Service)
Primary STT service with intelligent backup integration.

**Key Features:**
- Real-time transcription via Deepgram websocket
- Interim and final transcript handling
- VAD (Voice Activity Detection) support
- Voicemail detection (first 10 seconds)
- Interruption handling (configurable)
- Fast response mode
- Accumulated transcription buffering
- Multiple backup API key failover
- STT response time tracking
- Intelligent Gladia backup integration

**Configuration Options:**
- `api_key`: Primary Deepgram API key
- `backup_api_keys`: List of fallback Deepgram keys
- `url`: Deepgram websocket URL
- `sample_rate`: Audio sample rate
- `on_no_punctuation_seconds`: 2s (default timeout for sending accumulated transcripts)
- `detect_voicemail`: True (enables voicemail detection)
- `allow_interruptions`: True (allows STT during bot speech)
- `gladia_api_key`: Gladia API key for intelligent backup
- `gladia_timeout`: 1.8s (timeout for backup activation)
- `fast_response`: False (enables faster response mode)

**Frame Handling:**
- Processes `BotStartedSpeakingFrame` / `BotStoppedSpeakingFrame`
- Handles `VADActiveFrame` / `VADInactiveFrame`
- Responds to `STTRestartFrame`
- Emits `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame`
- Generates `TranscriptionFrame` / `InterimTranscriptionFrame`
- Can emit `VoicemailFrame`

**Main Methods:**
- `start(frame)` - Connects to Deepgram and starts backup services
- `stop(frame)` - Disconnects from all services
- `run_stt(audio: bytes)` - Sends audio to Deepgram and backup
- `process_frame(frame, direction)` - Handles various frame types
- `intelligent_backup_handler()` - Processes backup transcriptions from Gladia
- `_on_message()` - Main Deepgram message handler
- `_process_transcript_message()` - Routes interim/final transcripts
- `_process_final_transcript()` - Handles final transcripts with timing metrics
- `_should_ignore_transcription()` - Filtering logic for transcripts

**Performance Tracking:**
- `get_stt_response_times()` - Returns list of response durations
- `get_average_stt_response_time()` - Returns average response time
- `get_stt_stats()` - Returns comprehensive statistics
- `get_backup_stats()` - Returns backup system statistics
- `get_comprehensive_stt_stats()` - Returns all stats including backup
- `log_comprehensive_stt_performance()` - Logs detailed performance report

## Important Constants

```python
DEFAULT_ON_NO_PUNCTUATION_SECONDS = 2  # Timeout for accumulated transcripts
IGNORE_REPEATED_MSG_AT_START_SECONDS = 4  # Ignore repeated initial messages
VOICEMAIL_DETECTION_SECONDS = 10  # Window for voicemail detection
FALSE_INTERIM_SECONDS = 1.3  # Threshold for false interim detection
```

## Intelligent Backup System

The backup system provides ultra-high reliability by coordinating between Deepgram (primary) and Gladia (backup):

1. **Audio Streaming**: Audio sent to both services simultaneously
2. **Primary Response**: Deepgram typically responds first
3. **Timeout Monitoring**: Backup waits 1.0s for Deepgram
4. **Backup Activation**: If Deepgram fails/delays, Gladia transcript is used
5. **Deduplication**: Prevents duplicate transcripts through coordination
6. **VAD Filtering**: Only processes backups within VAD window
7. **Bot Speech Protection**: Prevents false interruptions during bot speech

## Transcription Flow

### Normal Flow:
1. Audio → Deepgram → Interim transcripts → User speaking events
2. Audio → Deepgram → Final transcript → Accumulated or sent immediately
3. Punctuation/timeout triggers sending accumulated transcripts

### Fast Response Flow:
- Short sentences: Send on punctuation or 2s timeout
- Long sentences: Send on punctuation + 2s timeout OR 4s timeout without punctuation

### Backup Flow:
1. Audio → Deepgram (primary) + Gladia (backup)
2. Gladia receives transcript → Stores with timeout
3. If Deepgram responds → Cancel Gladia timeout
4. If Deepgram silent → Gladia activates after 1.0s
5. Backup transcript processed as final transcript

## Filtering & Deduplication

**Transcripts are ignored if:**
- Interim with confidence < 0.7
- First word within 1 second (fast greeting)
- Repeated first message within 4 seconds
- Interim when VAD inactive
- Bot speaking and interruptions disabled
- Bot speaking with single word (for both primary and backup)
- Backup: Bot speaking + low word count (≤2) or confidence (<0.95)
- Backup: Too soon after bot started speaking (<1.5s)
- Backup: Too soon after last transcript (<2.0s)
- Outside VAD backup window (>1.5s since VAD inactive)
- Similar to recent Deepgram transcript (<3s)

## Voicemail Detection

- Active during first 10 seconds only
- Uses pattern matching from `pipecat.utils.text.voicemail`
- Emits `VoicemailFrame` when detected
- Stops processing after detection

## Error Handling

- Connection errors trigger API key rotation (backup keys)
- Maximum error count = number of backup keys
- Emits `DeepgramError` for recoverable errors
- Emits `DeepgramFatalError` when all keys exhausted
- Backup system provides fallback when Deepgram unavailable

## Supported Languages

**Gladia Backup Supports:**
- Spanish (es)
- English (en)
- French (fr)
- Portuguese (pt)
- Catalan (ca)
- German (de)
- Italian (it)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)
- Russian (ru)
- Arabic (ar)
- Hindi (hi)

## Key Implementation Details

### Response Time Tracking:
- `_current_speech_start_time`: Marks when speech detection begins
- `_stt_response_times`: List of all response durations
- Tracks both primary and backup response times
- Logs comprehensive performance metrics

### Bot Speech Tracking:
- `_bot_speaking`: Flag for bot speech state
- `_bot_started_speaking_time`: Timestamp when bot starts speaking
- Used to prevent false interruptions from backup system

### VAD Integration:
- `_vad_active`: Tracks Voice Activity Detection state
- `_last_vad_inactive_time`: Reference time for backup filtering
- Triggers finalize on VAD inactive
- Sends accumulated transcripts on VAD inactive with comma ending

### Accumulated Transcription:
- `_accum_transcription_frames`: Buffer for incomplete phrases
- Sent when punctuation detected or timeout expires
- Improves conversation flow by grouping related speech

## Usage Example

```python
stt_service = DeepgramSTTService(
    api_key="your_deepgram_key",
    backup_api_keys=["backup_key_1", "backup_key_2"],
    gladia_api_key="your_gladia_key",  # Optional for backup
    sample_rate=16000,
    detect_voicemail=True,
    allow_interruptions=True,
    fast_response=False,
    live_options=LiveOptions(
        language=Language.ES,
        model="nova-3-general",
        interim_results=True,
        vad_events=True,
        smart_format=True,
        punctuate=True
    )
)

# Get performance stats
stats = stt_service.get_comprehensive_stt_stats()
stt_service.log_comprehensive_stt_performance()

# Check connection status
status = stt_service.get_connection_status()
```

## Architecture Benefits

1. **Ultra-High Reliability**: Dual STT system ensures no transcripts lost
2. **Automatic Failover**: Seamless switch to backup when primary fails
3. **Low Latency**: Optimized coordination minimizes delays
4. **Smart Filtering**: Prevents duplicates and false positives
5. **Performance Tracking**: Detailed metrics for monitoring
6. **Flexible Configuration**: Highly customizable behavior
7. **Multi-language Support**: Extensive language coverage
