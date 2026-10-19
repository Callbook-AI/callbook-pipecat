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
    async def test_real_pipeline_retains_microphone_transcription_and_audio_output(self):
        with (
            patch.dict(
                os.environ, {"CARTESIA_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic"}
            ),
            patch.object(bot, "DailyTransport", wraps=bot.DailyTransport) as transport,
            patch.object(bot.PipelineRunner, "run", new_callable=AsyncMock) as run,
        ):
            await bot.main("https://synthetic.daily.co/room", "synthetic-room-token")
        params = transport.call_args.args[3]
        self.assertTrue(params.audio_in_enabled)
        self.assertTrue(params.audio_out_enabled)
        self.assertTrue(params.transcription_enabled)
        self.assertIsInstance(params.vad_analyzer, bot.SileroVADAnalyzer)
        task = run.await_args.args[0]
        self.assertIsInstance(task, bot.PipelineTask)

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
