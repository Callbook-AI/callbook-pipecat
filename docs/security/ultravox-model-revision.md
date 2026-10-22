# Ultravox model revision

The optional local Ultravox service retains its existing default model,
`fixie-ai/ultravox-v0_4_1-llama-3_1-8b`, and pins its Hub revision to
`d1d4298148a6584e085c15111f3348ca0cd47a19`. The official Hub API returned that
exact current commit on 2026-10-21 (last modified 2025-05-06). This freezes the
model that an uncached default load would select at verification time; it does
not assert what an older deployment or local cache previously loaded.

Both vLLM's model, remote-code and internal-tokenizer revisions and the separate
Transformers tokenizer revision use the same commit. Remote code remains enabled
because the existing model uses Ultravox custom configuration/model classes.
The pinned configuration's `auto_map` points to files inside the same repository,
not another Hub code repository. A commit pin prevents later changes to this
repository from being silently loaded; it is not a full security audit of those
files or a pin of every transitive model dependency they might reference.

Custom Hub models remain supported through an explicit full 40-character commit
SHA, for example `UltravoxSTTService(model_size="owner/model", revision=sha)`.
Missing revisions, mutable branches/tags and abbreviated hashes fail before model
loading. An override for the default model must also be immutable. Existing local
model directories remain supported without a Hub revision; their contents and any
external references are the caller's responsibility. Authentication, generation
settings, prompts, PCM handling and streaming behavior are unchanged.

Seven offline contracts load the real service with inference dependencies
substituted. They check all four revision boundaries, custom and local models,
failures before download, service argument forwarding and prompt/audio/streaming
behavior. CI also builds and verifies the installed wheel. These checks do not
establish GPU inference quality or execute the model's downloaded custom code.
No weights were downloaded, package published, consumer pin changed, worker
restarted or live call made as part of this source-only remediation.

Sources:

- [Official model snapshot](https://huggingface.co/fixie-ai/ultravox-v0_4_1-llama-3_1-8b/tree/d1d4298148a6584e085c15111f3348ca0cd47a19)
- [Official Hub metadata](https://huggingface.co/api/models/fixie-ai/ultravox-v0_4_1-llama-3_1-8b)
- [vLLM 0.7.3 engine revision arguments](https://docs.vllm.ai/en/v0.7.3/serving/engine_args.html)
- [Transformers 4.48 custom-code revision guidance](https://huggingface.co/docs/transformers/v4.48.0/en/custom_models#using-a-model-with-custom-code)
