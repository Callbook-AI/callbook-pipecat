import os

from dotenv import load_dotenv
import data
from pipecat.services.openai import OpenAILLMService
# from pipecat.services.anthropic import AnthropicLLMService
from pipecat.services.elevenlabs import ElevenLabsTTSService
# from custom_pipecat.elevenlabs import ElevenLabsTTSService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.deepgram import DeepgramSTTService, LiveOptions, DeepgramFatalError, DeepgramError
from pipecat.services.gladia import GladiaSTTService
from pipecat.processors.transcript_processor import TranscriptProcessor
# from pipecat.services.gemini_multimodal_live.gemini import (
#     GeminiMultimodalLiveLLMService,
#     GeminiMultimodalModalities,
#     InputParams
# )
from fast_text_aggregator import FastTextAggregator
from gladia_optimized_aggregator import create_gladia_optimized_aggregators
import recording
import user_idle
from loguru import logger
import instance

load_dotenv(override=True)


OPENAI_API_KEY = os.getenv("OPENAI_API_ENV")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
ANTHROPIC_API_KEY = "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"
ANTHROPIC_MODEL = "claude-3-5-haiku-20241022"
GEMINI_API_KEY = "AIzaSyBu3_HqlvLQ6xc7jkaPqnQYX3TojHdTv7k"
GEMINI_MODEL = "models/gemini-2.5-flash-lite"
GLADIA_API_KEY = "d0875e4f-4126-42d0-aeed-dc04e5709032"


def get_llm_service(call):

    if call['model']['provider'] != 'openai': return None

    model = call['model']['model']

    llm = OpenAILLMService(api_key=OPENAI_API_KEY, model=model)
    # llm = AnthropicLLMService(api_key=ANTHROPIC_API_KEY)

    # llm = GeminiMultimodalLiveLLMService(
    #     api_key=GEMINI_API_KEY,
    #     params=InputParams(modalities=GeminiMultimodalModalities.TEXT)
    # )

    return llm

def get_llm_context(call):

    messages = call['model']['messages']
    context = OpenAILLMContext(messages)

    return context

def get_stt_service(call):

    def on_connection_error(error):
        nonlocal call
        print("callid in on_connection_error:", call['id'])
        if isinstance(error, DeepgramFatalError):
            logger.error(f"Deepgram fatal error detected: {error.text}")
            call = data.update_call(call['id'], {
                'deepgram_error': 'fatal'
            })
        
        if isinstance(error, DeepgramError):
            logger.error(f"Deepgram error detected: {error.text}")
            call = data.update_call(call['id'], {
                'deepgram_error': 'error'
            })
        logger.error(f"Deepgram connection error: {error}")


    language = call['transcriber']['language']
    allow_interruptions = call['transcriber'].get('allow_interruptions', True)
    fast_response = call.get('fast_response', False)
    # allow_interruptions = True  
    stt_provider = call['transcriber'].get('provider', 'deepgram')
    # stt_provider = "dual"
    model = "nova-2-general" if stt_provider == 'deepgram' or stt_provider == 'dual' else "solaria-1"

    should_detect_voicemail = not (call.get("type") == "inboundPhoneCall")

    deepgram_key = instance.INSTANCE.get("deepgram_key")
    backup_deepgram_keys = instance.INSTANCE.get("backup_deepgram_keys", [])

    stt_provider = "gladia"
    if stt_provider == 'gladia':
        stt = GladiaSTTService(
            api_key=GLADIA_API_KEY,
            confidence=0.3,  # Reduced from 0.99 to be more responsive to actual speech
            on_no_punctuation_seconds=0.5,  # Much faster timeout for immediate response
            params=GladiaSTTService.InputParams(
                language=language,
                allow_interruptions=allow_interruptions,
                model="solaria-1",
                detect_voicemail=should_detect_voicemail,
                region="us-east",
                endpointing=0.05,           # Very aggressive endpointing for fastest response
                maximum_duration_without_endpointing=6,  # Shorter timeout
                speech_threshold=0.2,       # Lower threshold for faster detection
                audio_enhancer=False,       # Disable for speed
                code_switching=True,
                words_accurate_timestamps=False,  # Disable for speed
            )
        )
    elif stt_provider == 'deepgram':
        stt = DeepgramSTTService(
            api_key=deepgram_key,
            fast_response=fast_response,
            bakckup_api_keys=backup_deepgram_keys,
            on_connection_error=on_connection_error,
            live_options=LiveOptions(
                language=language, 
                vad_events=True, 
                model=model, 
                filler_words=True, 
                profanity_filter=False
            ),
            audio_passthrough=True,
            detect_voicemail=should_detect_voicemail,
            allow_interruptions=allow_interruptions,
        )
    elif stt_provider == 'dual':
        print("Using dual STT with Deepgram key:", deepgram_key)
        stt = DeepgramSTTService(
            api_key=deepgram_key, 
            bakckup_api_keys=backup_deepgram_keys,
            on_connection_error=on_connection_error,
            gladia_api_key=GLADIA_API_KEY,  
            gladia_timeout=1.5,
            live_options=LiveOptions(
                language=language, 
                vad_events=True, 
                model=model, 
                filler_words=True, 
                profanity_filter=False
            ),
            audio_passthrough=True,
            detect_voicemail=should_detect_voicemail,
            allow_interruptions=allow_interruptions,
        )


    return stt


def get_tts_service(call):
    logger.debug(f"TTS: {call['voice']}")
    voice = call['voice']
    voice_id = voice['voiceId']
    language = call['voice']['language']
    speed = voice.get('speed', 1.0)
    model = voice.get('model', 'eleven_flash_v2_5')
    fast_response = call.get('fast_response', False)
    
    logger.debug(f"fast_response: {fast_response}")
    stop_frame_timeout_s = 1.0
    text_aggregator = None
    # fast_response = True
    if fast_response:
        stop_frame_timeout_s = 0.1
        text_aggregator = FastTextAggregator()

    tts = ElevenLabsTTSService(
        api_key=ELEVEN_LABS_API_KEY,
        voice_id=voice_id,
        model=model,
        stop_frame_timeout_s=stop_frame_timeout_s,
        text_aggregator=text_aggregator,
        params=ElevenLabsTTSService.InputParams(
            similarity_boost=voice['similarityBoost'],
            stability=voice['stability'],
            speed=speed,
            language=language,
            auto_mode=False,
        )
    )

    return tts


def get_transcript_processor( manage_voicemail_message: bool = False):

    transcript_processor = TranscriptProcessor()
    transcript_processor.manage_voicemail_message = manage_voicemail_message    

    return transcript_processor


def get_pipeline_items(call, manage_voicemail_message: bool = False):

    llm = get_llm_service(call)
    logger.debug("got llm")

    llm_context = get_llm_context(call)
    logger.debug("got llm_context")

    # Check if we're using Gladia STT for optimization
    stt_provider = call['transcriber'].get('provider', 'deepgram')
    stt_provider = "gladia"  # Force to gladia for testing - should match get_stt_service()
    
    if stt_provider == 'gladia':
        # Use optimized aggregators for Gladia
        logger.info("🚀 Using Gladia-optimized aggregators for improved performance")
        llm_context_aggregator = create_gladia_optimized_aggregators(
            llm_context,
            user_kwargs={
                'aggregation_timeout': 0.05,  # Very fast aggregation
                'bot_interruption_timeout': 0.1,
                'immediate_mode': True  # Process transcripts immediately
            },
            assistant_kwargs={
                'expect_stripped_words': True,
                'min_chars_for_streaming': 15,  # Start TTS after 15 chars
                'streaming_enabled': True
            }
        )
        logger.info("✅ Gladia-optimized aggregators created successfully")
    else:
        # Use standard aggregators for Deepgram/Dual
        logger.info("📝 Using standard aggregators for Deepgram/Dual")
        llm_context_aggregator = llm.create_context_aggregator(
            llm_context,
            assistant_kwargs={"expect_stripped_words": True},
            user_kwargs={ 'aggregation_timeout': 0, 'bot_interruption_timeout': 0 }
        )
    
    logger.debug("got llm_context_aggregator")

    stt = get_stt_service(call)
    logger.debug("got stt")

    tts = get_tts_service(call)
    logger.debug("got tts")

    transcript_processor = get_transcript_processor(manage_voicemail_message)
    logger.debug("got transcript_processor")
            
    audiobuffer = recording.get_audio_buffer_processor(call['provider_id'])
    logger.debug("got audiobuffer")
    user_idle_processor = user_idle.get_user_idle_processor(call)
    logger.debug("got user_idle_processor")

    return (
        stt,
        user_idle_processor,
        transcript_processor,
        llm_context_aggregator,
        llm,
        tts,
        audiobuffer,
    )
