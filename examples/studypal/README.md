# studypal

### Have a conversation about any article on the web

studypal is a fast conversational AI built using [Daily](https://www.daily.co/) for real-time media transport and [Cartesia](https://cartesia.ai) for text-to-speech. Everything is orchestrated together (VAD -> STT -> LLM -> TTS) using [Pipecat](https://www.pipecat.ai/).

## Setup

1. Clone the repository
2. Copy `env.example` to a `.env` file and add API keys
3. Install the required packages: `pip install -r requirements.txt`
4. Run `python3 studypal.py` from your command line.
5. While the app is running, go to the `https://<yourdomain>.daily.co/<room_url>` set in `DAILY_SAMPLE_ROOM_URL` and talk to studypal!

Use Python 3.11 or 3.12 (Python 3.10 is no longer supported) with the pinned Pipecat 1.13.0 dependencies.

Run the offline dependency compatibility checks from this example directory:

```bash
python -m unittest test_dependency_compatibility.py
```

Pipecat 1.13.0 uses the installed sentencex tokenizer for sentence boundaries, including the first spoken response.

## PDF dependency security (2026-10-22)

The example pins pypdf 6.21.1, the smallest version fixing all five newly
reported findings: GHSA-7mh2-4gg2-979f and GHSA-39mj-jf6j-2qj3 (fixed 6.20.0),
GHSA-9mx8-4j7g-633p and GHSA-qfv9-64jg-mh7p (6.20.1), and
[GHSA-5hx5-vpf9-5wcv](https://github.com/advisories/GHSA-5hx5-vpf9-5wcv)
(6.21.1). No voice, model, provider, authentication or application code changes.

Eight offline contracts pass on the example's separately installed Python 3.12
SDK graph: real synthetic PDF text extraction, arXiv URL normalization, HTML,
error responses, token limits, participant lifecycle, audio/VAD/context, sentence
segmentation and malformed PDFs. A five-second subprocess timeout bounds the
upstream #4170 EOF-xref regression even if parsing blocks the interpreter.
Existing Linux CI installs and validates this example independently of the fork.

This is an opt-in example, not part of the root library's setuptools src package
or its runtime requirements. There is no dev branch or StudyPal deployment
workflow; staging has unrelated history. Merge the example fix to main without
publishing the voice library, changing downstream pins or restarting production
workers. No actual provider call, microphone or customer message is exercised.
Rollback restores only this example's requirements from the preceding main.
