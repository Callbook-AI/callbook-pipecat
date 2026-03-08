#
# TTS Service Contract Tests
#
# These tests define the behavioral contract that any TTS service must satisfy
# to work correctly in the Pipecat pipeline. They use a mock TTS service that
# simulates real audio generation. When implementing a new TTS provider, ensure
# your service passes equivalent behavior.
#

import asyncio
import unittest
from typing import Any, AsyncGenerator, Dict, Mapping, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    LLMFullResponseEndFrame,
    StartFrame,
    StartInterruptionFrame,
    TextFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.ai_services import TTSService
from pipecat.tests.utils import SleepFrame, run_test


# ---------------------------------------------------------------------------
# Mock TTS Service — simulates what a real provider (e.g. ElevenLabs) does
# ---------------------------------------------------------------------------

class MockTTSService(TTSService):
    """A mock TTS service that produces deterministic audio frames.

    This mimics the behavior of ElevenLabs' HTTP streaming variant:
    - Yields TTSStartedFrame at the beginning
    - Yields one or more TTSAudioRawFrame chunks
    - Yields TTSStoppedFrame at the end
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        fail_on_text: Optional[str] = None,
        audio_chunk_size: int = 320,
        num_chunks: int = 2,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)
        self.fail_on_text = fail_on_text
        self.audio_chunk_size = audio_chunk_size
        self.num_chunks = num_chunks
        self.run_tts_calls: list[str] = []
        self._settings: Dict[str, Any] = {
            "language": None,
        }

    def can_generate_metrics(self) -> bool:
        return True

    def set_voice(self, voice: str):
        self._voice_id = voice

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        self.run_tts_calls.append(text)

        if self.fail_on_text and text == self.fail_on_text:
            yield ErrorFrame(error=f"TTS failed for: {text}")
            return

        yield TTSStartedFrame()
        for _ in range(self.num_chunks):
            audio = b"\x00" * self.audio_chunk_size
            yield TTSAudioRawFrame(audio=audio, sample_rate=self.sample_rate, num_channels=1)
        yield TTSStoppedFrame()


# ---------------------------------------------------------------------------
# 1. Core TTS Contract Tests
# ---------------------------------------------------------------------------

class TestTTSServiceContract(unittest.IsolatedAsyncioTestCase):
    """Tests the core contract that ANY TTS service must satisfy."""

    async def test_single_text_frame_produces_started_audio_stopped(self):
        """Sending a TextFrame with a complete sentence triggers TTS immediately
        due to sentence aggregation. The sentence boundary (period) causes the
        aggregator to flush without waiting for LLMFullResponseEndFrame.
        """
        tts = MockTTSService()

        frames_to_send = [TextFrame(text="Hello world.")]
        expected_down = [
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
        ]
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)

    async def test_tts_speak_frame(self):
        """TTSSpeakFrame should trigger TTS generation directly."""
        tts = MockTTSService()

        frames_to_send = [TTSSpeakFrame(text="Say this.")]
        expected_down = [
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
        ]
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)

    async def test_audio_frame_contains_correct_sample_rate(self):
        """Audio frames must carry the correct sample rate."""
        tts = MockTTSService(sample_rate=24000)

        frames_to_send = [TTSSpeakFrame(text="Check sample rate.")]
        expected_down = [
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
        ]
        (down, _) = await run_test(
            tts, frames_to_send=frames_to_send, expected_down_frames=expected_down
        )

        for frame in down:
            if isinstance(frame, TTSAudioRawFrame):
                assert frame.sample_rate == 24000, (
                    f"Expected sample_rate 24000, got {frame.sample_rate}"
                )
                assert frame.num_channels == 1

    async def test_empty_text_is_not_sent_to_tts(self):
        """Whitespace-only text should be filtered out before reaching run_tts."""
        tts = MockTTSService()

        frames_to_send = [TTSSpeakFrame(text="   ")]
        expected_down = []
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)
        assert len(tts.run_tts_calls) == 0, "run_tts should not be called for whitespace-only text"

    async def test_error_in_run_tts_yields_error_frame(self):
        """If run_tts yields an ErrorFrame, process_generator sends it upstream
        (via push_error). The TTSTextFrame still goes downstream since
        push_text_frames=True.
        """
        tts = MockTTSService(fail_on_text="fail me")

        frames_to_send = [TTSSpeakFrame(text="fail me")]
        expected_down = [TTSTextFrame]
        expected_up = [ErrorFrame]
        await run_test(
            tts,
            frames_to_send=frames_to_send,
            expected_down_frames=expected_down,
            expected_up_frames=expected_up,
        )


# ---------------------------------------------------------------------------
# 2. Sentence Aggregation Tests
# ---------------------------------------------------------------------------

class TestTTSSentenceAggregation(unittest.IsolatedAsyncioTestCase):
    """Tests sentence aggregation behavior — how token-by-token LLM output is
    batched into sentences before being sent to TTS."""

    async def test_aggregation_batches_tokens_into_sentence(self):
        """Individual tokens should be aggregated until a sentence boundary."""
        tts = MockTTSService(aggregate_sentences=True)

        # Simulate LLM streaming tokens one-by-one, ending with LLMFullResponseEndFrame
        frames_to_send = [
            TextFrame(text="Hello "),
            TextFrame(text="world."),
            LLMFullResponseEndFrame(),
        ]
        # "Hello world." gets aggregated and sent as one TTS call
        expected_down = [
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
            LLMFullResponseEndFrame,
        ]
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)

    async def test_no_aggregation_sends_each_token(self):
        """With aggregation disabled, each TextFrame triggers run_tts independently."""
        tts = MockTTSService(aggregate_sentences=False)

        frames_to_send = [
            TextFrame(text="Hello"),
            TextFrame(text="world"),
            LLMFullResponseEndFrame(),
        ]
        # Each token produces its own started/audio/stopped cycle
        expected_down = [
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
            LLMFullResponseEndFrame,
        ]
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)


# ---------------------------------------------------------------------------
# 3. Interruption Handling Tests
# ---------------------------------------------------------------------------

class TestTTSInterruptionHandling(unittest.IsolatedAsyncioTestCase):
    """Tests that TTS services properly handle interruptions."""

    async def test_interruption_is_forwarded(self):
        """StartInterruptionFrame should pass through the TTS service."""
        tts = MockTTSService()

        frames_to_send = [StartInterruptionFrame()]
        expected_down = [StartInterruptionFrame]
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)

    async def test_text_after_interruption_still_works(self):
        """TTS should recover and process text normally after an interruption."""
        tts = MockTTSService()

        frames_to_send = [
            StartInterruptionFrame(),
            TTSSpeakFrame(text="After interruption."),
        ]
        expected_down = [
            StartInterruptionFrame,
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
        ]
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)


# ---------------------------------------------------------------------------
# 4. Settings Update Tests
# ---------------------------------------------------------------------------

class TestTTSSettingsUpdate(unittest.IsolatedAsyncioTestCase):
    """Tests dynamic settings updates via TTSUpdateSettingsFrame."""

    async def test_update_voice_setting(self):
        """TTSUpdateSettingsFrame with 'voice' should update the voice."""
        tts = MockTTSService()
        tts.set_voice("original-voice")

        frames_to_send = [
            TTSUpdateSettingsFrame(settings={"voice": "new-voice"}),
            TTSSpeakFrame(text="With new voice."),
        ]
        expected_down = [
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
        ]
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)
        assert tts._voice_id == "new-voice"

    async def test_update_unknown_setting_does_not_crash(self):
        """Unknown settings should be logged but not cause exceptions."""
        tts = MockTTSService()

        frames_to_send = [
            TTSUpdateSettingsFrame(settings={"nonexistent_param": 42}),
            TTSSpeakFrame(text="Should still work."),
        ]
        expected_down = [
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
        ]
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)


# ---------------------------------------------------------------------------
# 5. Multiple Utterances Test
# ---------------------------------------------------------------------------

class TestTTSMultipleUtterances(unittest.IsolatedAsyncioTestCase):
    """Tests that sequential TTS requests work correctly."""

    async def test_two_sequential_speak_frames(self):
        """Two TTSSpeakFrames should each produce their own started/audio/stopped cycle."""
        tts = MockTTSService()

        frames_to_send = [
            TTSSpeakFrame(text="First sentence."),
            TTSSpeakFrame(text="Second sentence."),
        ]
        expected_down = [
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
            TTSTextFrame,
        ]
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)
        assert tts.run_tts_calls == ["First sentence.", "Second sentence."]


# ---------------------------------------------------------------------------
# 6. push_text_frames behavior
# ---------------------------------------------------------------------------

class TestTTSTextFramePushing(unittest.IsolatedAsyncioTestCase):
    """Tests the push_text_frames flag behavior."""

    async def test_push_text_frames_disabled(self):
        """When push_text_frames=False, no TTSTextFrame should be emitted."""
        tts = MockTTSService(push_text_frames=False)

        frames_to_send = [TTSSpeakFrame(text="No text frame.")]
        expected_down = [
            TTSStartedFrame,
            TTSAudioRawFrame,
            TTSAudioRawFrame,
            TTSStoppedFrame,
        ]
        await run_test(tts, frames_to_send=frames_to_send, expected_down_frames=expected_down)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
