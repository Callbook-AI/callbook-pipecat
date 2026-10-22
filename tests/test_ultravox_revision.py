"""Test model-loading boundaries without downloading weights or requiring a GPU."""

import asyncio
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def load_ultravox():
    """Load the real service with only optional inference dependencies substituted."""
    frames = types.ModuleType("pipecat.frames.frames")
    for name in (
        "AudioRawFrame",
        "CancelFrame",
        "EndFrame",
        "ErrorFrame",
        "Frame",
        "StartFrame",
        "TranscriptionFrame",
        "UserStartedSpeakingFrame",
        "UserStoppedSpeakingFrame",
    ):
        setattr(frames, name, type(name, (), {}))
    modules = {
        "numpy": types.SimpleNamespace(ndarray=list),
        "huggingface_hub": types.SimpleNamespace(login=Mock()),
        "loguru": types.SimpleNamespace(logger=Mock()),
        "transformers": types.SimpleNamespace(AutoTokenizer=Mock()),
        "vllm": types.SimpleNamespace(AsyncLLMEngine=Mock(), SamplingParams=Mock()),
        "vllm.engine.arg_utils": types.SimpleNamespace(AsyncEngineArgs=Mock()),
        "pipecat.frames.frames": frames,
        "pipecat.processors.frame_processor": types.SimpleNamespace(FrameDirection=object),
        "pipecat.services.ai_services": types.SimpleNamespace(AIService=object),
        "pipecat.utils.time": types.SimpleNamespace(time_now_iso8601=Mock()),
    }
    source = Path(__file__).parents[1] / "src/pipecat/services/ultravox.py"
    spec = importlib.util.spec_from_file_location("ultravox_revision_under_test", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(module)
    return module


class TestUltravoxRevision(unittest.TestCase):
    def setUp(self):
        self.module = load_ultravox()

    def assert_pinned(self, name, revision):
        module = self.module
        module.AsyncEngineArgs.assert_called_once_with(
            model=name,
            revision=revision,
            code_revision=revision,
            tokenizer_revision=revision,
            gpu_memory_utilization=0.9,
            max_model_len=8192,
            trust_remote_code=True,
        )
        module.AutoTokenizer.from_pretrained.assert_called_once_with(name, revision=revision)
        module.AsyncLLMEngine.from_engine_args.assert_called_once_with(
            module.AsyncEngineArgs.return_value
        )

    def test_default_pins_engine_remote_code_and_both_tokenizers(self):
        self.module.UltravoxModel()
        self.assert_pinned(
            "fixie-ai/ultravox-v0_4_1-llama-3_1-8b",
            "d1d4298148a6584e085c15111f3348ca0cd47a19",
        )

    def test_custom_model_uses_its_own_revision(self):
        self.module.UltravoxModel("example/custom", revision="a" * 40)
        self.assert_pinned("example/custom", "a" * 40)

    def test_mutable_or_missing_custom_revision_fails_before_loading(self):
        for revision in (None, "main", "v1.0", "abc1234", "z" * 40, "a" * 40 + "\n", ""):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                self.module.UltravoxModel("example/custom", revision=revision)
        self.module.AsyncEngineArgs.assert_not_called()
        self.module.AutoTokenizer.from_pretrained.assert_not_called()

    def test_default_revision_override_must_also_be_immutable(self):
        with self.assertRaises(ValueError):
            self.module.UltravoxModel(revision="main")
        self.module.AsyncEngineArgs.assert_not_called()

    def test_local_model_directory_remains_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            self.module.UltravoxModel(directory)
            self.assert_pinned(directory, None)

    def test_service_forwards_revision_and_retains_generation_settings(self):
        service = self.module.UltravoxSTTService(
            model_size="example/custom", revision="b" * 40, temperature=0.2, max_tokens=55
        )
        self.assert_pinned("example/custom", "b" * 40)
        self.assertEqual(service._temperature, 0.2)
        self.assertEqual(service._max_tokens, 55)

    def test_prompt_audio_and_streaming_generation_are_preserved(self):
        module = self.module
        model = module.UltravoxModel()
        messages = [{"role": "user", "content": "<|audio|>\n"}]
        model.tokenizer.apply_chat_template.return_value = "formatted prompt"
        audio = [1, 2, 3]

        async def outputs():
            for text, reason in [("hello", None), ("hello world", "stop")]:
                yield types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(text=text, finish_reason=reason)]
                )

        model.engine.generate.return_value = outputs()

        async def collect():
            return [json.loads(chunk) async for chunk in model.generate(messages, 0.2, 55, audio)]

        chunks = asyncio.run(collect())
        model.tokenizer.apply_chat_template.assert_called_once_with(
            messages, tokenize=False, add_generation_prompt=True
        )
        module.SamplingParams.assert_called_once_with(
            temperature=0.2, max_tokens=55, stop_token_ids=None
        )
        self.assertEqual(
            model.engine.generate.call_args.args[0],
            {"prompt": "formatted prompt", "multi_modal_data": {"audio": audio}},
        )
        self.assertEqual(
            [chunk["choices"][0]["delta"]["content"] for chunk in chunks], ["hello", " world"]
        )
        self.assertEqual(chunks[0]["choices"][0]["delta"]["role"], "assistant")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")
