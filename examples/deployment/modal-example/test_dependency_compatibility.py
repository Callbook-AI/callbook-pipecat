"""Construct the real upgraded voice pipeline without joining a room or calling AI APIs."""

import importlib
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import bot
import modal

from pipecat.transports.daily import utils as daily_utils


class ModalDependencyCompatibility(unittest.IsolatedAsyncioTestCase):
    async def test_real_pipeline_retains_audio_vad_context_and_participant_lifecycle(self):
        captured = {}
        real_transport = bot.DailyTransport
        real_pair = bot.LLMContextAggregatorPair

        def transport_factory(*args, **kwargs):
            captured["transport"] = real_transport(*args, **kwargs)
            captured["params"] = args[3]
            return captured["transport"]

        def pair_factory(context, **kwargs):
            captured["context"] = context
            captured["user_params"] = kwargs["user_params"]
            return real_pair(context, **kwargs)

        async def run_pipeline(worker):
            captured["worker"] = worker
            transport = captured["transport"]
            with (
                patch.object(
                    transport, "capture_participant_transcription", new_callable=AsyncMock
                ) as capture,
                patch.object(worker, "queue_frames", new_callable=AsyncMock) as queue,
                patch.object(worker, "cancel", new_callable=AsyncMock) as cancel,
            ):
                for handler in transport._event_handlers["on_first_participant_joined"].handlers:
                    await handler(transport, {"id": "synthetic-participant"})
                capture.assert_awaited_once_with("synthetic-participant")
                self.assertIsInstance(queue.await_args.args[0][0], bot.LLMRunFrame)
                self.assertIn(
                    "introduce yourself", captured["context"].get_messages()[-1]["content"]
                )
                for handler in transport._event_handlers["on_participant_left"].handlers:
                    await handler(transport, {"id": "synthetic-participant"}, "left")
                cancel.assert_awaited_once()

        with (
            patch.dict(
                os.environ, {"CARTESIA_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic"}
            ),
            patch.object(bot, "DailyTransport", side_effect=transport_factory),
            patch.object(bot, "LLMContextAggregatorPair", side_effect=pair_factory),
            patch.object(bot.PipelineRunner, "run", side_effect=run_pipeline),
        ):
            await bot.main("https://synthetic.daily.co/room", "synthetic-room-token")
        params = captured["params"]
        self.assertTrue(params.audio_in_enabled)
        self.assertTrue(params.audio_out_enabled)
        self.assertTrue(params.audio_in_passthrough)
        self.assertTrue(params.transcription_enabled)
        user_params = captured["user_params"]
        self.assertIsInstance(user_params.vad_analyzer, bot.SileroVADAnalyzer)
        self.assertIsInstance(
            user_params.user_turn_strategies.stop[0], bot.SpeechTimeoutUserTurnStopStrategy
        )
        self.assertIsInstance(captured["worker"], bot.PipelineWorker)

    async def test_endpoint_creates_room_and_returns_named_token_without_logging_it(self):
        with patch.object(
            modal.Secret, "from_dotenv", return_value=modal.Secret.from_dict({"SYNTHETIC": "true"})
        ):
            app = importlib.import_module("app")
        with (
            patch.object(daily_utils, "DailyRESTHelper") as helper,
            patch.object(app.launch_bot_process, "spawn") as spawn,
            patch.object(app.logger, "info") as log,
        ):
            helper.return_value.create_room = AsyncMock(
                return_value=SimpleNamespace(url="https://synthetic.daily.co/room")
            )
            helper.return_value.get_token = AsyncMock(return_value="synthetic-room-token")
            response = await app.start.get_raw_f()()
        self.assertEqual(
            json.loads(response.body),
            {"room_url": "https://synthetic.daily.co/room", "token": "synthetic-room-token"},
        )
        spawn.assert_called_once_with(
            room_url="https://synthetic.daily.co/room", token="synthetic-room-token"
        )
        helper.return_value.get_token.assert_awaited_once_with(
            "https://synthetic.daily.co/room", app.MAX_SESSION_TIME
        )
        self.assertNotIn("synthetic-room-token", str(log.call_args_list))


if __name__ == "__main__":
    unittest.main()
