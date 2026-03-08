#
# Copyright (c) 2024-2025 Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call

import pytest

from pipecat.frames.frames import (
    TranscriptionFrame,
    InterimTranscriptionFrame,
    VoicemailFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.services.deepgram import DeepgramSTTService
from pipecat.transcriptions.language import Language


# Mock LiveResultResponse structure
class MockWord:
    def __init__(self, start=0.0):
        self.start = start


class MockAlternative:
    def __init__(self, transcript="", confidence=1.0, words=None, languages=None):
        self.transcript = transcript
        self.confidence = confidence
        self.words = words or []
        self.languages = languages or []


class MockChannel:
    def __init__(self, alternatives=None):
        self.alternatives = alternatives or []


class MockLiveResultResponse:
    def __init__(
        self,
        transcript="",
        confidence=1.0,
        is_final=True,
        speech_final=True,
        start=0.0,
        words=None,
        languages=None,
    ):
        self.is_final = is_final
        self.speech_final = speech_final
        self.start = start
        self.channel = MockChannel(
            [MockAlternative(transcript, confidence, words or [], languages or [])]
        )


# ============================================================================
# 1. VOICEMAIL DETECTION TESTS (via _on_message)
# ============================================================================


class TestVoicemailDetection(unittest.TestCase):
    """Test suite for voicemail detection through _on_message."""

    def setUp(self):
        """Set up test fixtures."""
        self.stt_service = DeepgramSTTService(
            api_key="test_key", detect_voicemail=True, sample_rate=16000
        )
        self.stt_service.push_frame = AsyncMock()
        self.stt_service._restarted = True  # Required for _on_message to process

    @pytest.mark.asyncio
    async def test_voicemail_detection_disabled(self):
        """Test 1.1: Voicemail detection disabled."""
        self.stt_service.detect_voicemail = False
        self.stt_service.start_time = time.time() - 5

        result = MockLiveResultResponse(
            transcript="please leave a message",
            is_final=True,
            confidence=0.9,
            words=[MockWord(2.0)],
        )

        with patch("pipecat.utils.text.voicemail.is_text_voicemail", return_value=True):
            await self.stt_service._on_message(result=result)

        # Should not push VoicemailFrame
        voicemail_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], VoicemailFrame)
        ]
        self.assertEqual(len(voicemail_frames), 0)

    @pytest.mark.asyncio
    async def test_voicemail_detection_after_timeout_window(self):
        """Test 1.2: Voicemail detection after timeout window."""
        self.stt_service.detect_voicemail = True
        self.stt_service.start_time = time.time() - 15  # > 10 seconds
        self.stt_service._was_first_transcript_receipt = True

        result = MockLiveResultResponse(
            transcript="please leave a message after the beep",
            is_final=True,
            confidence=0.9,
            words=[MockWord(2.0)],
        )

        with patch("pipecat.utils.text.voicemail.is_text_voicemail", return_value=True):
            await self.stt_service._on_message(result=result)

        # Should not push VoicemailFrame (outside time window)
        voicemail_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], VoicemailFrame)
        ]
        self.assertEqual(len(voicemail_frames), 0)

    @pytest.mark.asyncio
    async def test_voicemail_not_detected_normal_text(self):
        """Test 1.3: Voicemail not detected - normal text."""
        self.stt_service.detect_voicemail = True
        self.stt_service.start_time = time.time() - 5  # < 10 seconds
        self.stt_service._was_first_transcript_receipt = False

        result = MockLiveResultResponse(
            transcript="hello how are you", is_final=True, confidence=0.9, words=[MockWord(2.0)]
        )

        with patch("pipecat.utils.text.voicemail.is_text_voicemail", return_value=False):
            await self.stt_service._on_message(result=result)

        # Should not push VoicemailFrame
        voicemail_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], VoicemailFrame)
        ]
        self.assertEqual(len(voicemail_frames), 0)

    @pytest.mark.asyncio
    async def test_voicemail_detected_successfully(self):
        """Test 1.4: Voicemail detected successfully."""
        self.stt_service.detect_voicemail = True
        self.stt_service.start_time = time.time() - 5  # < 10 seconds
        self.stt_service._was_first_transcript_receipt = False
        self.stt_service.language = "en"

        result = MockLiveResultResponse(
            transcript="please leave a message after the beep",
            is_final=True,
            confidence=0.9,
            words=[MockWord(2.0)],
            languages=["en"],
        )

        with patch("pipecat.utils.text.voicemail.is_text_voicemail", return_value=True):
            await self.stt_service._on_message(result=result)

        # Should push VoicemailFrame
        voicemail_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], VoicemailFrame)
        ]
        self.assertGreater(len(voicemail_frames), 0)

    @pytest.mark.asyncio
    async def test_voicemail_detection_first_transcript_not_received(self):
        """Test 1.5: Voicemail detection - first transcript not received."""
        self.stt_service.detect_voicemail = True
        self.stt_service.start_time = time.time() - 3  # < 10 seconds
        self.stt_service._was_first_transcript_receipt = False
        self.stt_service.language = "en"

        result = MockLiveResultResponse(
            transcript="please leave a message",
            is_final=True,
            confidence=0.9,
            words=[MockWord(2.0)],
            languages=["en"],
        )

        with patch("pipecat.utils.text.voicemail.is_text_voicemail", return_value=True):
            await self.stt_service._on_message(result=result)

        # Should push VoicemailFrame
        voicemail_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], VoicemailFrame)
        ]
        self.assertGreater(len(voicemail_frames), 0)

    @pytest.mark.asyncio
    async def test_voicemail_detection_edge_of_time_window(self):
        """Test 1.6: Voicemail detection - edge of time window."""
        self.stt_service.detect_voicemail = True
        self.stt_service.start_time = time.time() - 10.0  # Exactly at boundary
        self.stt_service._was_first_transcript_receipt = True

        result = MockLiveResultResponse(
            transcript="please leave a message",
            is_final=True,
            confidence=0.9,
            words=[MockWord(2.0)],
        )

        with patch("pipecat.utils.text.voicemail.is_text_voicemail", return_value=True):
            await self.stt_service._on_message(result=result)

        # Condition is `>`, so 10.0 is NOT > 10, should not detect
        voicemail_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], VoicemailFrame)
        ]
        self.assertEqual(len(voicemail_frames), 0)


# ============================================================================
# 2. TRANSCRIPTION FILTERING TESTS (via _on_message)
# ============================================================================


class TestTranscriptionFiltering(unittest.TestCase):
    """Test suite for transcription filtering through _on_message."""

    def setUp(self):
        """Set up test fixtures."""
        self.stt_service = DeepgramSTTService(
            api_key="test_key",
            allow_interruptions=True,
            sample_rate=16000,
        )
        self.stt_service.push_frame = AsyncMock()
        self.stt_service._restarted = True
        self.stt_service._vad_active = True
        self.stt_service._bot_speaking = False
        self.stt_service._first_message = None
        self.stt_service._first_message_time = None
        self.stt_service._last_time_transcription = 0

    @pytest.mark.asyncio
    async def test_ignore_interim_low_confidence(self):
        """Test 2.1: Ignore interim - low confidence."""
        result = MockLiveResultResponse(
            transcript="hello", confidence=0.5, is_final=False, words=[MockWord(2.0)]
        )

        await self.stt_service._on_message(result=result)

        # Should not push any transcription frames
        transcript_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], (TranscriptionFrame, InterimTranscriptionFrame))
        ]
        self.assertEqual(len(transcript_frames), 0)

    @pytest.mark.asyncio
    async def test_accept_interim_high_confidence(self):
        """Test 2.2: Accept interim - high confidence."""
        result = MockLiveResultResponse(
            transcript="hello",
            confidence=0.85,
            is_final=False,
            words=[MockWord(2.0)],
            languages=["en"],
        )

        await self.stt_service._on_message(result=result)

        # Should push interim transcription frame
        interim_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], InterimTranscriptionFrame)
        ]
        self.assertGreater(len(interim_frames), 0)

    @pytest.mark.asyncio
    async def test_accept_final_any_confidence(self):
        """Test 2.3: Accept final - any confidence."""
        result = MockLiveResultResponse(
            transcript="hello",
            confidence=0.3,
            is_final=True,
            words=[MockWord(2.0)],
            languages=["en"],
        )

        await self.stt_service._on_message(result=result)

        # Should push transcription frame (accumulated)
        self.stt_service.push_frame.assert_called()

    @pytest.mark.asyncio
    async def test_ignore_fast_greeting_single_word_early(self):
        """Test 2.4: Ignore fast greeting - single word early."""
        result = MockLiveResultResponse(transcript="hi", confidence=0.9, words=[MockWord(0.5)])

        await self.stt_service._on_message(result=result)

        # Should not push transcription frames (filtered)
        transcript_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], TranscriptionFrame)
        ]
        self.assertEqual(len(transcript_frames), 0)

    @pytest.mark.asyncio
    async def test_accept_fast_multi_word(self):
        """Test 2.5: Accept fast multi-word."""
        result = MockLiveResultResponse(
            transcript="hi there",
            confidence=0.9,
            words=[MockWord(0.5), MockWord(0.6)],
            languages=["en"],
        )

        await self.stt_service._on_message(result=result)

        # Should process (not filtered as fast greeting)
        self.stt_service.push_frame.assert_called()

    @pytest.mark.asyncio
    async def test_accept_single_word_after_1_second(self):
        """Test 2.6: Accept single word after 1 second."""
        result = MockLiveResultResponse(
            transcript="hello", confidence=0.9, words=[MockWord(1.5)], languages=["en"]
        )

        await self.stt_service._on_message(result=result)

        # Should process (time threshold passed)
        self.stt_service.push_frame.assert_called()

    @pytest.mark.asyncio
    async def test_ignore_repeated_first_message(self):
        """Test 2.7: Ignore repeated first message."""
        self.stt_service._first_message = "hello"
        self.stt_service._first_message_time = time.time() - 2  # 2 seconds ago

        result = MockLiveResultResponse(
            transcript="hello", confidence=0.9, words=[MockWord(2.0)], languages=["en"]
        )

        with patch("pipecat.utils.string.is_equivalent_basic", return_value=True):
            await self.stt_service._on_message(result=result)

        # Should be filtered (repeated first message within window)
        transcript_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], TranscriptionFrame)
        ]
        self.assertEqual(len(transcript_frames), 0)

    @pytest.mark.asyncio
    async def test_accept_repeated_first_message_after_timeout(self):
        """Test 2.8: Accept repeated first message after timeout."""
        self.stt_service._first_message = "hello"
        self.stt_service._first_message_time = time.time() - 5  # 5 seconds ago

        result = MockLiveResultResponse(
            transcript="hello", confidence=0.9, words=[MockWord(2.0)], languages=["en"]
        )

        with patch("pipecat.utils.string.is_equivalent_basic", return_value=True):
            await self.stt_service._on_message(result=result)

        # Should process (timeout exceeded)
        self.stt_service.push_frame.assert_called()

    @pytest.mark.asyncio
    async def test_ignore_interim_when_vad_inactive(self):
        """Test 2.9: Ignore interim when VAD inactive."""
        self.stt_service._vad_active = False

        result = MockLiveResultResponse(
            transcript="hello", confidence=0.85, is_final=False, words=[MockWord(2.0)]
        )

        await self.stt_service._on_message(result=result)

        # Should not push interim frames
        interim_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], InterimTranscriptionFrame)
        ]
        self.assertEqual(len(interim_frames), 0)

    @pytest.mark.asyncio
    async def test_accept_final_when_vad_inactive(self):
        """Test 2.10: Accept final when VAD inactive."""
        self.stt_service._vad_active = False

        result = MockLiveResultResponse(
            transcript="hello",
            confidence=0.9,
            is_final=True,
            words=[MockWord(2.0)],
            languages=["en"],
        )

        await self.stt_service._on_message(result=result)

        # Should process final transcripts
        self.stt_service.push_frame.assert_called()

    @pytest.mark.asyncio
    async def test_ignore_when_bot_speaking_and_interruptions_disabled(self):
        """Test 2.11: Ignore when bot speaking and interruptions disabled."""
        self.stt_service._bot_speaking = True
        self.stt_service._allow_stt_interruptions = False

        result = MockLiveResultResponse(
            transcript="hello world", confidence=0.9, words=[MockWord(2.0)], languages=["en"]
        )

        await self.stt_service._on_message(result=result)

        # Should not process
        transcript_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], TranscriptionFrame)
        ]
        self.assertEqual(len(transcript_frames), 0)

    @pytest.mark.asyncio
    async def test_accept_when_bot_speaking_and_interruptions_enabled(self):
        """Test 2.12: Accept when bot speaking and interruptions enabled."""
        self.stt_service._bot_speaking = True
        self.stt_service._allow_stt_interruptions = True

        result = MockLiveResultResponse(
            transcript="hello world", confidence=0.9, words=[MockWord(2.0)], languages=["en"]
        )

        await self.stt_service._on_message(result=result)

        # Should process
        self.stt_service.push_frame.assert_called()

    @pytest.mark.asyncio
    async def test_ignore_bot_speaking_single_word_primary_source(self):
        """Test 2.13: Ignore bot speaking single word (primary source)."""
        self.stt_service._bot_speaking = True
        self.stt_service._allow_stt_interruptions = True

        result = MockLiveResultResponse(transcript="yes", confidence=0.9, words=[MockWord(2.0)])

        await self.stt_service._on_message(result=result, backup_source=False)

        # Should filter single word during bot speech
        transcript_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], TranscriptionFrame)
        ]
        self.assertEqual(len(transcript_frames), 0)

    @pytest.mark.asyncio
    async def test_backup_source_bot_speaking_low_confidence(self):
        """Test 2.14: Backup source - bot speaking low confidence."""
        self.stt_service._bot_speaking = True
        self.stt_service._allow_stt_interruptions = True

        result = MockLiveResultResponse(
            transcript="hello there", confidence=0.90, words=[MockWord(2.0)]
        )

        await self.stt_service._on_message(result=result, backup_source=True)

        # Should filter (backup needs higher confidence)
        transcript_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], TranscriptionFrame)
        ]
        self.assertEqual(len(transcript_frames), 0)

    @pytest.mark.asyncio
    async def test_backup_source_bot_speaking_high_confidence_and_word_count(self):
        """Test 2.15: Backup source - bot speaking high confidence and word count."""
        self.stt_service._bot_speaking = True
        self.stt_service._allow_stt_interruptions = True
        self.stt_service._bot_started_speaking_time = time.time() - 2  # 2 seconds ago
        self.stt_service._last_time_transcription = time.time() - 3  # 3 seconds ago

        result = MockLiveResultResponse(
            transcript="hello there friend",
            confidence=0.96,
            words=[MockWord(2.0)],
            languages=["en"],
        )

        await self.stt_service._on_message(result=result, backup_source=True)

        # Should process (meets all backup criteria)
        self.stt_service.push_frame.assert_called()

    @pytest.mark.asyncio
    async def test_backup_source_too_soon_after_bot_started_speaking(self):
        """Test 2.16: Backup source - too soon after bot started speaking."""
        self.stt_service._bot_speaking = True
        self.stt_service._allow_stt_interruptions = True
        self.stt_service._bot_started_speaking_time = time.time() - 0.5  # 0.5 seconds ago

        result = MockLiveResultResponse(
            transcript="hello there friend", confidence=0.96, words=[MockWord(2.0)]
        )

        await self.stt_service._on_message(result=result, backup_source=True)

        # Should filter (too soon after bot started)
        transcript_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], TranscriptionFrame)
        ]
        self.assertEqual(len(transcript_frames), 0)

    @pytest.mark.asyncio
    async def test_backup_source_too_soon_after_last_transcript(self):
        """Test 2.17: Backup source - too soon after last transcript."""
        self.stt_service._bot_speaking = True
        self.stt_service._allow_stt_interruptions = True
        self.stt_service._bot_started_speaking_time = time.time() - 2  # 2 seconds ago
        self.stt_service._last_time_transcription = time.time() - 1.0  # 1 second ago

        result = MockLiveResultResponse(
            transcript="hello there friend", confidence=0.96, words=[MockWord(2.0)]
        )

        await self.stt_service._on_message(result=result, backup_source=True)

        # Should filter (too soon after last transcript)
        transcript_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], TranscriptionFrame)
        ]
        self.assertEqual(len(transcript_frames), 0)

    @pytest.mark.asyncio
    async def test_backup_source_not_bot_speaking(self):
        """Test 2.18: Backup source - not bot speaking."""
        self.stt_service._bot_speaking = False

        result = MockLiveResultResponse(
            transcript="hi", confidence=0.9, words=[MockWord(2.0)], languages=["en"]
        )

        await self.stt_service._on_message(result=result, backup_source=True)

        # Should process (backup restrictions only during bot speech)
        self.stt_service.push_frame.assert_called()


# ============================================================================
# 3. FAST RESPONSE ACCUMULATION TESTS (via _on_message)
# ============================================================================


class TestFastResponseAccumulation(unittest.TestCase):
    """Test suite for fast response accumulation through _on_message."""

    def setUp(self):
        """Set up test fixtures."""
        self.stt_service = DeepgramSTTService(
            api_key="test_key", fast_response=True, sample_rate=16000
        )
        self.stt_service.push_frame = AsyncMock()
        self.stt_service._restarted = True
        self.stt_service._on_no_punctuation_seconds = 2.0
        self.stt_service._vad_active = False

    @pytest.mark.asyncio
    async def test_fast_response_disabled(self):
        """Test 3.1: Fast response disabled."""
        self.stt_service._fast_response = False
        self.stt_service._accum_transcription_frames = []

        # Send two final transcripts
        result1 = MockLiveResultResponse(
            transcript="Hello", is_final=True, words=[MockWord(1.0)], languages=["en"]
        )
        result2 = MockLiveResultResponse(
            transcript="World", is_final=True, words=[MockWord(2.0)], languages=["en"]
        )

        await self.stt_service._on_message(result=result1)
        await self.stt_service._on_message(result=result2)

        # With fast response disabled, should use normal accumulation logic
        # Verify frames were pushed (exact count depends on punctuation logic)
        self.stt_service.push_frame.assert_called()

    @pytest.mark.asyncio
    async def test_vad_active_do_not_send(self):
        """Test 3.3: VAD active - do not send."""
        self.stt_service._vad_active = True
        self.stt_service._accum_transcription_frames = []

        result = MockLiveResultResponse(
            transcript="Hello", is_final=True, words=[MockWord(1.0)], languages=["en"]
        )

        await self.stt_service._on_message(result=result)

        # Should accumulate but not send while VAD active
        # Transcripts should be in accumulation buffer
        self.assertGreater(len(self.stt_service._accum_transcription_frames), 0)

    @pytest.mark.asyncio
    async def test_short_sentence_with_end_punctuation(self):
        """Test 3.4: Short sentence with end punctuation."""
        self.stt_service._fast_response = True
        self.stt_service._vad_active = False
        self.stt_service._accum_transcription_frames = []

        result = MockLiveResultResponse(
            transcript="Hello.", is_final=True, words=[MockWord(1.0)], languages=["en"]
        )

        await self.stt_service._on_message(result=result)

        # Should send immediately (short sentence with punctuation)
        user_stopped_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], UserStoppedSpeakingFrame)
        ]
        self.assertGreater(len(user_stopped_frames), 0)

    @pytest.mark.asyncio
    async def test_short_sentence_without_punctuation_needs_timeout(self):
        """Test 3.5/3.6: Short sentence without punctuation - timeout behavior."""
        self.stt_service._fast_response = True
        self.stt_service._vad_active = False
        self.stt_service._accum_transcription_frames = []
        self.stt_service._last_time_accum_transcription = time.time() - 1.0

        result = MockLiveResultResponse(
            transcript="Hello", is_final=True, words=[MockWord(1.0)], languages=["en"]
        )

        await self.stt_service._on_message(result=result)

        # Should accumulate (waiting for timeout)
        self.assertGreater(len(self.stt_service._accum_transcription_frames), 0)

        # Simulate timeout by updating time
        self.stt_service._last_time_accum_transcription = time.time() - 3.0

        # Trigger async handler logic
        await self.stt_service._fast_response_send_accum_transcriptions()

        # Should have sent after timeout
        user_stopped_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], UserStoppedSpeakingFrame)
        ]
        self.assertGreater(len(user_stopped_frames), 0)

    @pytest.mark.asyncio
    async def test_long_sentence_with_punctuation_and_timeout(self):
        """Test 3.7: Long sentence with punctuation and timeout."""
        self.stt_service._fast_response = True
        self.stt_service._vad_active = False
        self.stt_service._accum_transcription_frames = []

        # Send 3 finals to make it "long"
        results = [
            MockLiveResultResponse(
                transcript="Hello", is_final=True, words=[MockWord(1.0)], languages=["en"]
            ),
            MockLiveResultResponse(
                transcript="there", is_final=True, words=[MockWord(2.0)], languages=["en"]
            ),
            MockLiveResultResponse(
                transcript="friend.", is_final=True, words=[MockWord(3.0)], languages=["en"]
            ),
        ]

        for result in results:
            await self.stt_service._on_message(result=result)

        # Simulate timeout
        self.stt_service._last_time_accum_transcription = time.time() - 3.0
        await self.stt_service._fast_response_send_accum_transcriptions()

        # Should send (long sentence with punctuation and timeout)
        user_stopped_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], UserStoppedSpeakingFrame)
        ]
        self.assertGreater(len(user_stopped_frames), 0)

    @pytest.mark.asyncio
    async def test_long_sentence_with_punctuation_before_timeout(self):
        """Test 3.8: Long sentence with punctuation - before timeout."""
        self.stt_service._fast_response = True
        self.stt_service._vad_active = False
        self.stt_service._accum_transcription_frames = []
        self.stt_service._last_time_accum_transcription = time.time() - 1.0

        # Send 3 finals
        results = [
            MockLiveResultResponse(
                transcript="Hello", is_final=True, words=[MockWord(1.0)], languages=["en"]
            ),
            MockLiveResultResponse(
                transcript="there", is_final=True, words=[MockWord(2.0)], languages=["en"]
            ),
            MockLiveResultResponse(
                transcript="friend.", is_final=True, words=[MockWord(3.0)], languages=["en"]
            ),
        ]

        for result in results:
            await self.stt_service._on_message(result=result)

        # Should accumulate (needs timeout even with punctuation for long sentences)
        # Check that frames are accumulated
        self.assertGreaterEqual(len(self.stt_service._accum_transcription_frames), 1)

    @pytest.mark.asyncio
    async def test_long_sentence_without_punctuation_double_timeout(self):
        """Test 3.9/3.10: Long sentence without punctuation - double timeout."""
        self.stt_service._fast_response = True
        self.stt_service._vad_active = False
        self.stt_service._accum_transcription_frames = []

        # Send 3 finals without punctuation
        results = [
            MockLiveResultResponse(
                transcript="Hello", is_final=True, words=[MockWord(1.0)], languages=["en"]
            ),
            MockLiveResultResponse(
                transcript="there", is_final=True, words=[MockWord(2.0)], languages=["en"]
            ),
            MockLiveResultResponse(
                transcript="friend", is_final=True, words=[MockWord(3.0)], languages=["en"]
            ),
        ]

        for result in results:
            await self.stt_service._on_message(result=result)

        # Before double timeout (3 seconds < 4 seconds)
        self.stt_service._last_time_accum_transcription = time.time() - 3.0
        await self.stt_service._fast_response_send_accum_transcriptions()

        initial_stop_count = sum(
            1
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], UserStoppedSpeakingFrame)
        )

        # After double timeout (5 seconds > 4 seconds)
        self.stt_service._last_time_accum_transcription = time.time() - 5.0
        await self.stt_service._fast_response_send_accum_transcriptions()

        final_stop_count = sum(
            1
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], UserStoppedSpeakingFrame)
        )

        # Should send after double timeout
        self.assertGreater(final_stop_count, initial_stop_count)

    @pytest.mark.asyncio
    async def test_time_calculation_with_last_interim(self):
        """Test 3.11: Time calculation with last interim."""
        self.stt_service._fast_response = True
        self.stt_service._vad_active = False
        self.stt_service._accum_transcription_frames = []
        self.stt_service._last_interim_time = time.time() - 4.0
        self.stt_service._last_time_accum_transcription = time.time() - 1.0

        result = MockLiveResultResponse(
            transcript="Hello", is_final=True, words=[MockWord(1.0)], languages=["en"]
        )

        await self.stt_service._on_message(result=result)
        await self.stt_service._fast_response_send_accum_transcriptions()

        # Should use max(4.0, 1.0) = 4.0, so timeout exceeded
        user_stopped_frames = [
            call_args[0][0]
            for call_args in self.stt_service.push_frame.call_args_list
            if isinstance(call_args[0][0], UserStoppedSpeakingFrame)
        ]
        self.assertGreater(len(user_stopped_frames), 0)

    @pytest.mark.asyncio
    async def test_boundary_exactly_2_frames(self):
        """Test 3.12: Boundary - exactly 2 frames."""
        self.stt_service._fast_response = True
        self.stt_service._vad_active = False
        self.stt_service._accum_transcription_frames = []

        results = [
            MockLiveResultResponse(
                transcript="Hello", is_final=True, words=[MockWord(1.0)], languages=["en"]
            ),
            MockLiveResultResponse(
                transcript="there", is_final=True, words=[MockWord(2.0)], languages=["en"]
            ),
        ]

        for result in results:
            await self.stt_service._on_message(result=result)

        # 2 frames = short sentence
        self.assertEqual(len(self.stt_service._accum_transcription_frames), 2)

    @pytest.mark.asyncio
    async def test_boundary_exactly_3_frames(self):
        """Test 3.13: Boundary - exactly 3 frames."""
        self.stt_service._fast_response = True
        self.stt_service._vad_active = False
        self.stt_service._accum_transcription_frames = []

        results = [
            MockLiveResultResponse(
                transcript="Hello", is_final=True, words=[MockWord(1.0)], languages=["en"]
            ),
            MockLiveResultResponse(
                transcript="there", is_final=True, words=[MockWord(2.0)], languages=["en"]
            ),
            MockLiveResultResponse(
                transcript="friend", is_final=True, words=[MockWord(3.0)], languages=["en"]
            ),
        ]

        for result in results:
            await self.stt_service._on_message(result=result)

        # 3 frames = long sentence
        self.assertEqual(len(self.stt_service._accum_transcription_frames), 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
