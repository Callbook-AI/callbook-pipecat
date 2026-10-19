"""Offline construction checks for local faster-whisper capture (no model download)."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import bot

from pipecat.frames.frames import (
    EndFrame,
    InputAudioRawFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.services.whisper import stt as whisper_stt
from pipecat.transports.local import audio as local_audio


class LocalTranscriptionCompatibility(unittest.IsolatedAsyncioTestCase):
    async def test_cuda_transcription_pipeline_uses_selected_devices(self):
        with (
            patch.object(whisper_stt, "WhisperModel") as model,
            patch.object(local_audio.pyaudio, "PyAudio"),
            patch.object(bot, "LocalAudioTransport", wraps=bot.LocalAudioTransport) as transport,
            patch.object(bot.PipelineRunner, "run", new_callable=AsyncMock) as run,
            patch.object(bot, "VADProcessor", wraps=bot.VADProcessor) as vad,
            patch.object(bot, "Pipeline", wraps=bot.Pipeline) as pipeline,
        ):
            await bot.main(7, 9)
        params = transport.call_args.args[0]
        self.assertTrue(params.audio_in_enabled)
        self.assertFalse(params.audio_out_enabled)
        self.assertEqual(params.input_device_index, 7)
        self.assertEqual(params.output_device_index, 9)
        self.assertEqual(model.call_args.kwargs["device"], "cuda")
        self.assertEqual(model.call_args.args[0], bot.Model.LARGE.value)
        self.assertIsInstance(run.await_args.args[0], bot.PipelineWorker)
        self.assertIsInstance(vad.call_args.kwargs["vad_analyzer"], bot.SileroVADAnalyzer)
        self.assertIsInstance(pipeline.call_args.args[0][1], bot.VADProcessor)
        self.assertIsInstance(pipeline.call_args.args[0][2], bot.WhisperSTTService)

    async def test_segmented_pcm_produces_logged_transcription(self):
        with patch.object(whisper_stt, "WhisperModel") as model, patch("builtins.print") as output:
            model.return_value.transcribe.return_value = (
                [SimpleNamespace(text="Synthetic spoken phrase.", no_speech_prob=0.01)],
                None,
            )
            stt = bot.WhisperSTTService(device="cuda", model=bot.Model.LARGE, no_speech_prob=0.3)
            worker = bot.PipelineWorker(bot.Pipeline([stt, bot.TranscriptionLogger()]))
            await worker.queue_frames(
                [
                    VADUserStartedSpeakingFrame(),
                    InputAudioRawFrame(audio=b"\x01\x00" * 800, sample_rate=16000, num_channels=1),
                    VADUserStoppedSpeakingFrame(),
                    EndFrame(),
                ]
            )
            await asyncio.wait_for(bot.PipelineRunner(handle_sigint=False).run(worker), timeout=5)
        model.return_value.transcribe.assert_called_once()
        output.assert_any_call("Transcription: Synthetic spoken phrase. ")
        self.assertEqual(len(model.return_value.transcribe.call_args.args[0]), 800)


if __name__ == "__main__":
    unittest.main()
