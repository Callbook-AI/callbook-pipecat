"""Offline compatibility checks; run with `python -m unittest discover -s .`."""

import asyncio
import io
import os
import subprocess
import sys
import unittest
from unittest.mock import AsyncMock, patch

import studypal
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from pipecat.frames.frames import (
    EndFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


class Response:
    def __init__(self, *, status=200, body=b"", text=""):
        self.status = status
        self.body = body
        self.html = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def read(self):
        return self.body

    async def text(self):
        return self.html


class Session:
    def __init__(self, response):
        self.response = response
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        return self.response


class StudyPalDependencyCompatibility(unittest.IsolatedAsyncioTestCase):
    def test_malformed_pdf_rejection_is_bounded_in_a_separate_process(self):
        # The EOF-xref regression is the synthetic fixture from upstream #4170.
        # A subprocess timeout also bounds a regression that blocks Python itself.
        body = (
            b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Kids[]/Count 0>>endobj\n"
        )
        tail = b"\nstartxref\n%d\n%%%%EOF" % len(body)
        truncated_xref = body + b"xref\n0 3\n0000000000 65535 f \n" + tail
        code = (
            "import io,sys; from pypdf import PdfReader; "
            "PdfReader(io.BytesIO(sys.stdin.buffer.read()))"
        )
        for data, expected_error in (
            (b"not a PDF", b"PdfStreamError"),
            (truncated_xref, b"PdfReadError"),
        ):
            with self.subTest(data=data[:16]):
                result = subprocess.run(
                    [sys.executable, "-c", code], input=data,
                    capture_output=True, timeout=5, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected_error, result.stderr)
        self.assertIn(b"Unexpected EOF in Xref table", result.stderr)

    def test_extracts_real_pdf_and_normalizes_arxiv_url(self):
        writer = PdfWriter()
        page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(b"BT /F1 12 Tf 20 200 Td (Synthetic study article.) Tj ET")
        page[NameObject("/Contents")] = stream
        pdf = io.BytesIO()
        writer.write(pdf)
        session = Session(Response(body=pdf.getvalue()))
        text = asyncio.run(studypal.get_article_content("https://arxiv.org/abs/1234.5678", session))
        self.assertIn("Synthetic study article.", text)
        self.assertEqual(session.urls, ["https://arxiv.org/pdf/1234.5678.pdf"])

    def test_extracts_html_article(self):
        session = Session(Response(text='<div class="mw-parser-output"><p>Article text.</p></div>'))
        text = asyncio.run(studypal.get_article_content("https://example.invalid/wiki", session))
        self.assertEqual(text, "Article text.")

    def test_pdf_download_failure_does_not_parse_error_body(self):
        session = Session(Response(status=403, body=b"not a PDF"))
        self.assertEqual(
            asyncio.run(studypal.get_arxiv_content("https://arxiv.org/pdf/1234", session)),
            "Failed to download arXiv PDF.",
        )

    def test_token_limit_preserves_short_and_truncates_long_content(self):
        # Supply a deterministic local encoding to avoid downloading tiktoken assets.
        class Encoding:
            def encode(self, text):
                return list(text.encode())

            def decode(self, tokens):
                return bytes(tokens).decode()

        with patch.object(studypal.tiktoken, "encoding_for_model", return_value=Encoding()):
            self.assertEqual(studypal.truncate_content("Article", "gpt-4o-mini"), "Article")
            self.assertEqual(len(studypal.truncate_content("a" * 12000, "gpt-4o-mini")), 10000)

    async def test_real_pipeline_retains_audio_vad_context_and_participant_lifecycle(self):
        captured = {}
        real_transport = studypal.DailyTransport
        real_pair = studypal.LLMContextAggregatorPair

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
                self.assertIsInstance(queue.await_args.args[0][0], studypal.LLMRunFrame)
                self.assertIn("ready to discuss", captured["context"].get_messages()[-1]["content"])
                for handler in transport._event_handlers["on_participant_left"].handlers:
                    await handler(transport, {"id": "synthetic-participant"}, "left")
                cancel.assert_awaited_once()

        with (
            patch("builtins.input", return_value="https://example.invalid/article"),
            patch.object(
                studypal,
                "configure",
                new_callable=AsyncMock,
                return_value=("https://synthetic.daily.co/room", "synthetic-room-token"),
            ),
            patch.object(
                studypal,
                "get_article_content",
                new_callable=AsyncMock,
                return_value="Synthetic article text.",
            ),
            patch.object(studypal, "truncate_content", return_value="Synthetic article text."),
            patch.dict(
                os.environ, {"CARTESIA_API_KEY": "synthetic", "OPENAI_API_KEY": "synthetic"}
            ),
            patch.object(studypal, "DailyTransport", side_effect=transport_factory),
            patch.object(studypal, "LLMContextAggregatorPair", side_effect=pair_factory),
            patch.object(studypal.PipelineRunner, "run", side_effect=run_pipeline),
        ):
            await studypal.main()
        params = captured["params"]
        self.assertTrue(params.audio_in_enabled)
        self.assertTrue(params.audio_out_enabled)
        self.assertTrue(params.audio_in_passthrough)
        self.assertTrue(params.transcription_enabled)
        user_params = captured["user_params"]
        self.assertIsInstance(user_params.vad_analyzer, studypal.SileroVADAnalyzer)
        self.assertIsInstance(
            user_params.user_turn_strategies.stop[0], studypal.SpeechTimeoutUserTurnStopStrategy
        )
        self.assertIsInstance(captured["worker"], studypal.PipelineWorker)

        self.assertIn("Synthetic article text.", captured["context"].get_messages()[0]["content"])

    async def test_universal_context_preserves_real_user_and_assistant_turns(self):
        context = studypal.LLMContext([{"role": "system", "content": "Synthetic study session."}])
        response_seen = asyncio.Event()
        assertions = self

        class OfflineLLM(FrameProcessor):
            async def process_frame(self, frame, direction):
                await super().process_frame(frame, direction)
                if isinstance(frame, LLMContextFrame) and direction == FrameDirection.DOWNSTREAM:
                    assertions.assertEqual(
                        frame.context.get_messages()[-1],
                        {"role": "user", "content": "Synthetic user question."},
                    )
                    await self.push_frame(LLMFullResponseStartFrame())
                    await self.push_frame(TextFrame("Synthetic assistant answer."))
                    await self.push_frame(LLMFullResponseEndFrame())
                    response_seen.set()
                else:
                    await self.push_frame(frame, direction)

        pair = studypal.LLMContextAggregatorPair(
            context,
            user_params=studypal.LLMUserAggregatorParams(
                user_turn_strategies=studypal.UserTurnStrategies(
                    stop=[studypal.SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.01)]
                )
            ),
        )
        worker = studypal.PipelineWorker(
            studypal.Pipeline([pair.user(), OfflineLLM(), pair.assistant()])
        )
        running = asyncio.create_task(studypal.PipelineRunner(handle_sigint=False).run(worker))
        try:
            await worker.queue_frames(
                [
                    TranscriptionFrame(
                        "Synthetic user question.", "synthetic", "2026-10-19T00:00:00Z"
                    )
                ]
            )
            await asyncio.wait_for(response_seen.wait(), timeout=5)
            await worker.queue_frames([EndFrame()])
            await asyncio.wait_for(running, timeout=5)
        finally:
            if not running.done():
                await worker.cancel()
                await running
        self.assertEqual(
            context.get_messages()[-1],
            {"role": "assistant", "content": "Synthetic assistant answer."},
        )

    async def test_first_spoken_sentences_use_bundled_tokenizer(self):
        aggregator = SimpleTextAggregator()
        sentences = []
        for chunk in ["Hello, Dr. ", "Medina. ", "Your balance is $29.50. ", "Next sentence. "]:
            async for aggregation in aggregator.aggregate(chunk):
                sentences.append(aggregation.text)
        final = await aggregator.flush()
        if final:
            sentences.append(final.text)
        self.assertEqual(
            sentences, ["Hello, Dr. Medina.", "Your balance is $29.50.", "Next sentence."]
        )


if __name__ == "__main__":
    unittest.main()
