# October 2026 dependency remediation

The bundled examples use their own package manifests and are not the dependency
set installed by a consumer of this framework. Update and test each example in
its directory. The Python examples now use upstream pipecat-ai 1.13.0, including
its service and transport import paths; they do not install this older fork.

## LiveKit serializer migration

`pipecat.serializers.livekit.LivekitFrameSerializer` has been removed, matching
upstream's security fix in 0.0.94 for
[GHSA-c2jg-5cp7-6wc7](https://github.com/pipecat-ai/pipecat/security/advisories/GHSA-c2jg-5cp7-6wc7).
It used Python pickle to deserialize network input and could execute arbitrary
code. Neither this repository's examples nor the checked callbook-core main
source imports it.

Use `LiveKitTransport` for LiveKit audio. It handles native SDK audio frames
without this serializer. For clients with a custom binary WebSocket protocol,
use `ProtobufFrameSerializer` on both ends. Pickle payloads are not compatible
with that protocol and must not be accepted or automatically retried with pickle.

The framework's aiohttp lower bound is also updated to the patched 3.14 series.
Other voice service implementations, media framing and transports are retained.

These changes do not repin downstream applications or restart a voice service.
callbook-core's inspected main requirements pin the fork to commit
558f39742ea31a08865f3070d596b469ebb3107b. A downstream pin change needs separate
compatibility validation and delivery through that application's deployment path.


## Remaining mobile tooling advisories

The supported Expo 57 / Daily native SDK stack still includes `braces@3.0.3`
([GHSA-vfj7-8cjw-p6xm](https://github.com/advisories/GHSA-vfj7-8cjw-p6xm))
and `node-forge@1.4.0`
([GHSA-86w9-cpqp-85rv](https://github.com/advisories/GHSA-86w9-cpqp-85rv)).
At verification time neither advisory has a released fixed version. Braces is
used by the Metro/build glob tooling; node-forge is pulled by Expo CLI tooling.
These dependencies are retained and their findings remain open. npm's additional
parent findings describe these same two underlying advisories. Recheck released
fixes before removing the compatibility notes or claiming a clean mobile audit.

## Verification limits and existing framework failures

Six web example frozen installs, builds and audits passed; mobile iOS/Android
Hermes bundles and native project generation passed. The three Python examples
have eleven offline compatibility tests. No live provider session, actual native
compilation, device audio or downstream voice deployment is established by them.

The original framework CI cannot collect Deepgram tests because its test extras
omit that dependency. The test requirements now declare Deepgram and ElevenLabs.
With these prerequisites, the original source has 210 passing tests, 44 failures
and four skips. The candidate has 214 passing tests, the same 44 failing test IDs
and four skips. The failures remain visible; they were not quarantined. Existing
whole-repository import/format violations also remain; changed Python files pass
the pinned checks.

Build distributions from a clean checkout. Setuptools can retain removed modules
in an old `build/lib` directory. The build workflow installs the actual wheel in
isolation and checks that the removed serializer is absent while protobuf audio
still round-trips; checking the source tree alone does not verify the artifact.


## Patched SDK follow-up

Version 0.0.94 fixes the removed pickle serializer but is still affected by
[development-runner call control](https://github.com/advisories/GHSA-j8cv-x86q-rj85)
and [file traversal](https://github.com/advisories/GHSA-3363-2ph6-35wh).
All three standalone Python examples use the current stable 1.13.0 release.
This requires Python 3.11 or newer. Their context, VAD and lifecycle contracts
run in separate CI jobs using each example's own dependency set on Python 3.12.
The fork's runtime dependency set is separate and retains its existing API.

Pipecat 1.5 is also affected by
[development-runner exposure](https://github.com/advisories/GHSA-hqm2-cfx2-2vgh),
whose boundary is clarified starting in 1.6. The development runner intentionally
has unauthenticated endpoints and must remain local; a production application
needs an authenticated dispatcher. An SDK upgrade does not add production
authentication to that development tool.

The final 1.13 SDK uses bundled Sentencex tokenization. Its clean dependency
installation does not require NLTK or a startup tokenizer download. The tests
exercise first-utterance sentence boundaries directly; the older SDK's NLTK
image bootstrap and CI download step are removed.


Pipecat 1.x moves voice-activity handling into the user aggregator. The examples
keep an explicit speech-timeout turn strategy rather than adopting a new turn
model implicitly. The CUDA Whisper example includes a VAD processor and forwards
control frames through its transcript logger; otherwise transcription or worker
startup can stall. The lifecycle tests process actual synthetic frames through
those paths, with provider/model inference and audio hardware boundaries mocked.
