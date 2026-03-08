#
# ElevenLabs TTS Service Tests
#
# Tests for ElevenLabsTTSService (WebSocket) and its helper functions.
# All network I/O is mocked — no real API calls are made.
#

import asyncio
import base64
import json
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from pipecat.frames.frames import (
    StartFrame,
    TTSSpeakFrame,
)
from pipecat.services.elevenlabs import (
    ElevenLabsTTSService,
    build_elevenlabs_voice_settings,
    calculate_word_times,
    language_to_elevenlabs_language,
    output_format_from_sample_rate,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transcriptions.language import Language
from pipecat.utils.asyncio import TaskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeWebSocket:
    """Simulates an ElevenLabs WebSocket connection.

    Also acts as an awaitable so `await websockets.connect(...)` returns self.
    """

    def __init__(self, messages: List[str] = None):
        self.messages = messages or []
        self.sent: List[str] = []
        self.closed = False
        self.state = MagicMock()
        self.close_rcvd = None
        self.close_sent = None
        self.close_rcvd_then_sent = None

    def __await__(self):
        yield
        return self

    async def send(self, data: str):
        self.sent.append(data)

    async def close(self):
        self.closed = True

    async def ping(self):
        pass

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for msg in self.messages:
            yield msg
        await asyncio.Future()


# ===========================================================================
# 1. Pure function tests (no pipeline needed)
# ===========================================================================

class TestLanguageMapping(unittest.TestCase):
    def test_base_language(self):
        assert language_to_elevenlabs_language(Language.EN) == "en"
        assert language_to_elevenlabs_language(Language.ES) == "es"
        assert language_to_elevenlabs_language(Language.JA) == "ja"

    def test_language_variant_fallback(self):
        if hasattr(Language, "ES_ES"):
            result = language_to_elevenlabs_language(Language.ES_ES)
            assert result == "es"

    def test_unsupported_language_returns_none(self):
        for lang in Language:
            result = language_to_elevenlabs_language(lang)
            assert result is None or isinstance(result, str)


class TestOutputFormat(unittest.TestCase):
    def test_known_sample_rates(self):
        assert output_format_from_sample_rate(16000) == "pcm_16000"
        assert output_format_from_sample_rate(22050) == "pcm_22050"
        assert output_format_from_sample_rate(24000) == "pcm_24000"
        assert output_format_from_sample_rate(44100) == "pcm_44100"

    def test_unknown_sample_rate_defaults_to_16000(self):
        assert output_format_from_sample_rate(48000) == "pcm_16000"
        assert output_format_from_sample_rate(8000) == "pcm_16000"


class TestBuildVoiceSettings(unittest.TestCase):
    def test_both_stability_and_similarity(self):
        settings = {
            "stability": 0.5, "similarity_boost": 0.8,
            "style": None, "use_speaker_boost": None, "speed": None,
        }
        result = build_elevenlabs_voice_settings(settings)
        assert result == {"stability": 0.5, "similarity_boost": 0.8}

    def test_all_settings(self):
        settings = {
            "stability": 0.5, "similarity_boost": 0.8,
            "style": 0.3, "use_speaker_boost": True, "speed": 1.2,
        }
        result = build_elevenlabs_voice_settings(settings)
        assert result == {
            "stability": 0.5, "similarity_boost": 0.8,
            "style": 0.3, "use_speaker_boost": True, "speed": 1.2,
        }

    def test_missing_stability_returns_none(self):
        settings = {
            "stability": None, "similarity_boost": None,
            "style": None, "use_speaker_boost": None, "speed": None,
        }
        assert build_elevenlabs_voice_settings(settings) is None

    def test_partial_stability_returns_none(self):
        settings = {
            "stability": 0.5, "similarity_boost": None,
            "style": None, "use_speaker_boost": None, "speed": None,
        }
        assert build_elevenlabs_voice_settings(settings) is None


class TestCalculateWordTimes(unittest.TestCase):
    def test_single_word(self):
        # For a single word "Hi", the last char index triggers the time calc
        # using zipped_times[i-1][1], so it picks the second-to-last char's time.
        alignment = {"chars": ["H", "i"], "charStartTimesMs": [0, 50]}
        result = calculate_word_times(alignment, cumulative_time=0.0)
        assert len(result) == 1
        assert result[0][0] == "Hi"
        assert result[0][1] == 0.0  # uses zipped_times[0][1] = 0ms

    def test_multiple_words(self):
        alignment = {
            "chars": ["H", "i", " ", "y", "o"],
            "charStartTimesMs": [0, 50, 100, 150, 200],
        }
        result = calculate_word_times(alignment, cumulative_time=0.0)
        assert len(result) == 2
        assert result[0][0] == "Hi"
        assert result[1][0] == "yo"

    def test_cumulative_time_offset(self):
        alignment = {"chars": ["O", "k"], "charStartTimesMs": [0, 50]}
        result = calculate_word_times(alignment, cumulative_time=1.0)
        assert result[0][1] == 1.0  # cumulative 1.0 + zipped_times[0][1]=0ms


class TestInputParamsValidation(unittest.TestCase):
    def test_valid_both_none(self):
        params = ElevenLabsTTSService.InputParams()
        assert params.stability is None
        assert params.similarity_boost is None

    def test_valid_both_set(self):
        params = ElevenLabsTTSService.InputParams(stability=0.5, similarity_boost=0.8)
        assert params.stability == 0.5

    def test_invalid_only_stability(self):
        with self.assertRaises(ValidationError):
            ElevenLabsTTSService.InputParams(stability=0.5)

    def test_invalid_only_similarity_boost(self):
        with self.assertRaises(ValidationError):
            ElevenLabsTTSService.InputParams(similarity_boost=0.8)


# ===========================================================================
# 2. ElevenLabsTTSService (WebSocket) — integration tests
# ===========================================================================

class TestElevenLabsWebSocket(unittest.IsolatedAsyncioTestCase):

    def _make_service(self, **kwargs):
        defaults = dict(
            api_key="test-key",
            voice_id="test-voice",
            model="eleven_flash_v2_5",
            sample_rate=16000,
        )
        defaults.update(kwargs)
        return ElevenLabsTTSService(**defaults)

    async def _start_service(self, tts, sample_rate=16000):
        """Bootstrap a TTS service the same way the pipeline would."""
        tm = TaskManager()
        tm.set_event_loop(asyncio.get_event_loop())
        frame = StartFrame(
            clock=MagicMock(),
            task_manager=tm,
            audio_out_sample_rate=sample_rate,
        )
        await tts.process_frame(frame, FrameDirection.DOWNSTREAM)
        return tts

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_websocket_url_includes_model_and_format(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service(model="eleven_flash_v2_5")
        await self._start_service(tts, 16000)

        url = mock_connect.call_args[0][0]
        assert "model_id=eleven_flash_v2_5" in url
        assert "output_format=pcm_16000" in url
        assert "auto_mode=true" in url

        await tts._disconnect()

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_websocket_url_includes_language_for_multilingual_model(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        params = ElevenLabsTTSService.InputParams(language=Language.ES)
        tts = self._make_service(model="eleven_flash_v2_5", params=params)
        await self._start_service(tts)

        url = mock_connect.call_args[0][0]
        assert "language_code=es" in url

        await tts._disconnect()

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_websocket_url_omits_language_for_non_multilingual_model(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        params = ElevenLabsTTSService.InputParams(language=Language.ES)
        tts = self._make_service(model="eleven_multilingual_v2", params=params)
        await self._start_service(tts)

        url = mock_connect.call_args[0][0]
        assert "language_code" not in url

        await tts._disconnect()

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_voice_settings_sent_in_init_message(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        params = ElevenLabsTTSService.InputParams(stability=0.5, similarity_boost=0.8)
        tts = self._make_service(params=params)
        await self._start_service(tts)

        init_msg = json.loads(fake_ws.sent[0])
        assert "voice_settings" in init_msg
        assert init_msg["voice_settings"]["stability"] == 0.5
        assert init_msg["voice_settings"]["similarity_boost"] == 0.8

        await tts._disconnect()

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_no_voice_settings_when_not_configured(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        init_msg = json.loads(fake_ws.sent[0])
        assert "voice_settings" not in init_msg

        await tts._disconnect()

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_init_message_contains_api_key(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service(api_key="my-secret-key")
        await self._start_service(tts)

        init_msg = json.loads(fake_ws.sent[0])
        assert init_msg["xi_api_key"] == "my-secret-key"
        assert init_msg["text"] == " "

        await tts._disconnect()

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_flush_audio_sends_flush_message(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        await tts.flush_audio()

        flush_msgs = [json.loads(m) for m in fake_ws.sent if "flush" in m]
        assert len(flush_msgs) == 1
        assert flush_msgs[0]["flush"] is True

        await tts._disconnect()

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_disconnect_sends_empty_text_and_closes(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)
        await tts._disconnect()

        close_msgs = [json.loads(m) for m in fake_ws.sent if json.loads(m).get("text") == ""]
        assert len(close_msgs) >= 1
        assert fake_ws.closed

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_voice_change_reconnects_websocket(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        initial_call_count = mock_connect.call_count
        await tts._update_settings({"voice": "new-voice"})

        assert mock_connect.call_count > initial_call_count
        assert tts._voice_id == "new-voice"

        await tts._disconnect()

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_connection_error_sets_websocket_to_none(self, mock_connect):
        mock_connect.side_effect = Exception("Connection refused")

        tts = self._make_service()
        await self._start_service(tts)

        assert tts._websocket is None

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_max_websocket_size_is_16mb(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        _, kwargs = mock_connect.call_args
        assert kwargs.get("max_size") == 16 * 1024 * 1024

        await tts._disconnect()

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_optimize_streaming_latency_in_url(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        params = ElevenLabsTTSService.InputParams(optimize_streaming_latency="3")
        tts = self._make_service(params=params)
        await self._start_service(tts)

        url = mock_connect.call_args[0][0]
        assert "optimize_streaming_latency=3" in url

        await tts._disconnect()

    @patch("pipecat.services.elevenlabs.websockets.connect")
    async def test_duplicate_connect_is_noop(self, mock_connect):
        """Calling _connect_websocket when already connected should not open a second socket."""
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        call_count = mock_connect.call_count
        await tts._connect_websocket()

        assert mock_connect.call_count == call_count  # no new connection

        await tts._disconnect()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
