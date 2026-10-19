"""Offline construction checks for local faster-whisper capture (no model download)."""

import unittest
from unittest.mock import AsyncMock, patch

import bot
import faster_whisper

from pipecat.transports.local import audio as local_audio


class LocalTranscriptionCompatibility(unittest.IsolatedAsyncioTestCase):
    async def test_cuda_transcription_pipeline_uses_selected_devices(self):
        with (
            patch.object(faster_whisper, "WhisperModel") as model,
            patch.object(local_audio.pyaudio, "PyAudio"),
            patch.object(bot, "LocalAudioTransport", wraps=bot.LocalAudioTransport) as transport,
            patch.object(bot.PipelineRunner, "run", new_callable=AsyncMock) as run,
        ):
            await bot.main(7, 9)
        params = transport.call_args.args[0]
        self.assertTrue(params.audio_in_enabled)
        self.assertFalse(params.audio_out_enabled)
        self.assertEqual(params.input_device_index, 7)
        self.assertEqual(params.output_device_index, 9)
        self.assertEqual(model.call_args.kwargs["device"], "cuda")
        self.assertEqual(model.call_args.args[0], bot.Model.LARGE.value)
        self.assertIsInstance(run.await_args.args[0], bot.PipelineTask)


if __name__ == "__main__":
    unittest.main()
