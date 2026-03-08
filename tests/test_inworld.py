#
# Inworld TTS Service Tests
#
# Tests for InworldTTSService (WebSocket) and its helper functions.
# All network I/O is mocked — no real API calls are made.
#

import asyncio
import base64
import json
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from pipecat.frames.frames import StartFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.inworld import (
    InworldTTSService,
    calculate_word_times,
    output_encoding_from_sample_rate,
)
from pipecat.utils.asyncio import TaskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _context_created_response(context_id: str = "ctx-test-1") -> str:
    return json.dumps({
        "result": {
            "contextId": context_id,
            "contextCreated": {
                "voiceId": "test-voice",
                "modelId": "v1.5",
            },
            "status": {"code": 0, "message": "OK"},
        }
    })


def _context_created_error_response() -> str:
    return json.dumps({
        "result": {
            "status": {"code": 3, "message": "Invalid voice"},
        }
    })


class FakeWebSocket:
    """Simulates an Inworld WebSocket connection.

    Supports being awaited (websockets.connect returns an awaitable).
    First recv() returns the context_created response.
    Iteration yields pre-configured messages then blocks.
    """

    def __init__(self, messages: List[str] = None, context_response: str = None):
        self.messages = messages or []
        self.sent: List[str] = []
        self.closed = False
        self._context_response = context_response or _context_created_response()
        self._recv_called = False

    def __await__(self):
        yield
        return self

    async def send(self, data: str):
        self.sent.append(data)

    async def recv(self):
        """Return context created response on first call."""
        if not self._recv_called:
            self._recv_called = True
            return self._context_response
        await asyncio.Future()  # block forever

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
# 1. Pure function tests
# ===========================================================================

class TestOutputEncoding(unittest.TestCase):
    def test_always_returns_linear16(self):
        assert output_encoding_from_sample_rate(16000) == "LINEAR16"
        assert output_encoding_from_sample_rate(24000) == "LINEAR16"
        assert output_encoding_from_sample_rate(48000) == "LINEAR16"


class TestCalculateWordTimes(unittest.TestCase):
    def test_single_word(self):
        alignment = {"words": ["Hello"], "wordStartTimeSeconds": [0.1]}
        result = calculate_word_times(alignment, cumulative_time=0.0)
        assert len(result) == 1
        assert result[0] == ("Hello", 0.1)

    def test_multiple_words(self):
        alignment = {
            "words": ["Hello", "world"],
            "wordStartTimeSeconds": [0.1, 0.5],
        }
        result = calculate_word_times(alignment, cumulative_time=0.0)
        assert len(result) == 2
        assert result[0] == ("Hello", 0.1)
        assert result[1] == ("world", 0.5)

    def test_cumulative_time_offset(self):
        alignment = {"words": ["Ok"], "wordStartTimeSeconds": [0.2]}
        result = calculate_word_times(alignment, cumulative_time=1.0)
        assert result[0] == ("Ok", 1.2)

    def test_empty_alignment(self):
        alignment = {"words": [], "wordStartTimeSeconds": []}
        result = calculate_word_times(alignment, cumulative_time=0.0)
        assert result == []

    def test_missing_keys(self):
        result = calculate_word_times({}, cumulative_time=0.0)
        assert result == []


class TestInputParamsValidation(unittest.TestCase):
    def test_defaults(self):
        params = InworldTTSService.InputParams()
        assert params.speaking_rate is None
        assert params.temperature is None
        assert params.auto_mode is True

    def test_custom_values(self):
        params = InworldTTSService.InputParams(
            speaking_rate=1.2, temperature=0.8, auto_mode=False
        )
        assert params.speaking_rate == 1.2
        assert params.temperature == 0.8
        assert params.auto_mode is False


# ===========================================================================
# 2. InworldTTSService (WebSocket) — integration tests
# ===========================================================================

class TestInworldWebSocket(unittest.IsolatedAsyncioTestCase):

    def _make_service(self, **kwargs):
        defaults = dict(
            api_key="test-api-key",
            voice_id="test-voice",
            model="v1.5",
            sample_rate=16000,
        )
        defaults.update(kwargs)
        return InworldTTSService(**defaults)

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

    # --- Connection & URL ---

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_websocket_connects_to_correct_url(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        url = mock_connect.call_args[0][0]
        assert url == "wss://api.inworld.ai/tts/v1/voice:streamBidirectional"

        await tts._disconnect()

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_websocket_sends_auth_header(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service(api_key="my-secret-key")
        await self._start_service(tts)

        _, kwargs = mock_connect.call_args
        headers = kwargs.get("extra_headers", {})
        assert headers["authorization"] == "Basic my-secret-key"

        await tts._disconnect()

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_websocket_max_size_16mb(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        _, kwargs = mock_connect.call_args
        assert kwargs.get("max_size") == 16 * 1024 * 1024

        await tts._disconnect()

    # --- Create context message ---

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_create_message_contains_voice_and_model(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service(voice_id="narrator-v2", model="v1.5")
        await self._start_service(tts)

        create_msg = json.loads(fake_ws.sent[0])
        assert create_msg["create"]["voiceId"] == "narrator-v2"
        assert create_msg["create"]["modelId"] == "v1.5"

        await tts._disconnect()

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_create_message_audio_config(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service(sample_rate=24000)
        await self._start_service(tts, sample_rate=24000)

        create_msg = json.loads(fake_ws.sent[0])
        audio_config = create_msg["create"]["audioConfig"]
        assert audio_config["audioEncoding"] == "LINEAR16"
        assert audio_config["sampleRateHertz"] == 24000

        await tts._disconnect()

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_create_message_includes_speaking_rate(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        params = InworldTTSService.InputParams(speaking_rate=1.3)
        tts = self._make_service(params=params)
        await self._start_service(tts)

        create_msg = json.loads(fake_ws.sent[0])
        assert create_msg["create"]["audioConfig"]["speakingRate"] == 1.3

        await tts._disconnect()

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_create_message_includes_temperature(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        params = InworldTTSService.InputParams(temperature=0.5)
        tts = self._make_service(params=params)
        await self._start_service(tts)

        create_msg = json.loads(fake_ws.sent[0])
        assert create_msg["create"]["temperature"] == 0.5

        await tts._disconnect()

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_create_message_auto_mode_and_timestamps(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        create_msg = json.loads(fake_ws.sent[0])
        assert create_msg["create"]["autoMode"] is True
        assert create_msg["create"]["timestampType"] == "WORD"

        await tts._disconnect()

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_create_message_no_speaking_rate_when_none(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        create_msg = json.loads(fake_ws.sent[0])
        assert "speakingRate" not in create_msg["create"]["audioConfig"]

        await tts._disconnect()

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_create_message_no_temperature_when_none(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        create_msg = json.loads(fake_ws.sent[0])
        assert "temperature" not in create_msg["create"]

        await tts._disconnect()

    # --- Context ID ---

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_context_id_stored_after_create(self, mock_connect):
        fake_ws = FakeWebSocket(context_response=_context_created_response("ctx-abc"))
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        assert tts._context_id == "ctx-abc"

        await tts._disconnect()

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_failed_context_creation_sets_websocket_none(self, mock_connect):
        fake_ws = FakeWebSocket(context_response=_context_created_error_response())
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        assert tts._websocket is None
        assert tts._context_id is None

        await tts._disconnect()

    # --- Flush ---

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_flush_audio_sends_flush_context(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        await tts.flush_audio()

        flush_msgs = [json.loads(m) for m in fake_ws.sent if "flush_context" in m and "create" not in m]
        assert len(flush_msgs) == 1
        assert flush_msgs[0]["contextId"] == tts._context_id

        await tts._disconnect()

    # --- Disconnect ---

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_disconnect_sends_close_context_and_closes(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        context_id = tts._context_id
        await tts._disconnect()

        close_msgs = [json.loads(m) for m in fake_ws.sent if "close_context" in m]
        assert len(close_msgs) == 1
        assert close_msgs[0]["contextId"] == context_id
        assert fake_ws.closed

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_disconnect_clears_context_id(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)
        await tts._disconnect()

        assert tts._context_id is None
        assert tts._websocket is None

    # --- Duplicate connect ---

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_duplicate_connect_is_noop(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        call_count = mock_connect.call_count
        await tts._connect_websocket()

        assert mock_connect.call_count == call_count

        await tts._disconnect()

    # --- Connection error ---

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_connection_error_sets_websocket_to_none(self, mock_connect):
        mock_connect.side_effect = Exception("Connection refused")

        tts = self._make_service()
        await self._start_service(tts)

        assert tts._websocket is None

    # --- Send text ---

    @patch("pipecat.services.inworld.websockets.connect")
    async def test_send_text_format(self, mock_connect):
        fake_ws = FakeWebSocket()
        mock_connect.return_value = fake_ws

        tts = self._make_service()
        await self._start_service(tts)

        await tts._send_text("Hello world!")

        text_msgs = [json.loads(m) for m in fake_ws.sent if "send_text" in m]
        assert len(text_msgs) == 1
        assert text_msgs[0]["send_text"]["text"] == "Hello world!"
        assert "flush_context" in text_msgs[0]["send_text"]
        assert text_msgs[0]["contextId"] == tts._context_id

        await tts._disconnect()

    # --- Build create message directly ---

    def test_build_create_message_minimal(self):
        tts = self._make_service()
        tts._sample_rate = 16000
        msg = tts._build_create_message()

        assert msg["create"]["voiceId"] == "test-voice"
        assert msg["create"]["modelId"] == "v1.5"
        assert msg["create"]["audioConfig"]["audioEncoding"] == "LINEAR16"
        assert msg["create"]["audioConfig"]["sampleRateHertz"] == 16000
        assert msg["create"]["timestampType"] == "WORD"
        assert msg["create"]["autoMode"] is True
        assert "temperature" not in msg["create"]
        assert "speakingRate" not in msg["create"]["audioConfig"]

    def test_build_create_message_all_params(self):
        params = InworldTTSService.InputParams(
            speaking_rate=1.2, temperature=0.7, auto_mode=False
        )
        tts = self._make_service(params=params)
        tts._sample_rate = 48000
        msg = tts._build_create_message()

        assert msg["create"]["audioConfig"]["sampleRateHertz"] == 48000
        assert msg["create"]["audioConfig"]["speakingRate"] == 1.2
        assert msg["create"]["temperature"] == 0.7
        assert msg["create"]["autoMode"] is False


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
