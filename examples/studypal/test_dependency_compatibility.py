"""Offline compatibility checks; run with `python -m unittest discover -s .`."""

import asyncio
import io
import unittest
from unittest.mock import patch

import studypal
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


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


class StudyPalDependencyCompatibility(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
