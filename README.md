# llm-vertex-genai

An [LLM](https://llm.datasette.io/) plugin for Gemini models on **Google Cloud Vertex AI**, built on the `google-genai` SDK.

It is for Vertex AI, billed to a GCP project and authenticated with `gcloud`. For Google AI Studio API keys, use [llm-gemini](https://github.com/simonw/llm-gemini) instead.

Compared with the plugins it grew from ([justyns/llm-vertex](https://github.com/justyns/llm-vertex), then [avoidik/llm-vertex](https://github.com/avoidik/llm-vertex)), it adds:

- **attachments**: images, audio, video, PDF and text, plus `gs://` URIs;
- **real conversation history** for `llm -c` and `llm chat`, model turns included;
- **any model id**, through `LLM_VERTEX_MODELS`, so a new Gemini release needs no plugin update;
- a `thinking_level` option, token usage in `llm logs`, and the `global` location by default;
- no Google auth at registration, so other `llm` commands do not slow down.

## Installation

```terminal
llm install https://github.com/NightMachinery/llm-vertex-genai/archive/refs/heads/main.zip
```

It registers the same `vertex-…` model ids as `llm-vertex`, so uninstall that first: `llm uninstall llm-vertex`.

## Use

Authenticate and pick a project:

```terminal
gcloud auth application-default login
export VERTEX_PROJECT_ID=my-gcp-project   # or GOOGLE_CLOUD_PROJECT
export VERTEX_LOCATION=global             # optional; global is the default
```

Then:

```terminal
llm -m vertex-gemini-3.8-flash "What's one clever name for a pet pelican?"
llm -m vertex-gemini-4-argon -a photo.jpg "Describe this image."
llm -m vertex-gemini-3.8-flash -a memo.wav "Transcribe this."
llm -m vertex-gemini-4-argon -o thinking_level high "Prove that sqrt(2) is irrational."
llm chat -m vertex-gemini-3.8-flash
```

Built-in models: `gemini-4-argon`, `gemini-3.8-flash`, `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`. Register more, comma-separated, each one becoming `vertex-<id>`:

```terminal
export LLM_VERTEX_MODELS=gemini-next-preview,another-model-id
```

Options: `temperature`, `top_p`, `top_k`, `max_output_tokens`, `thinking_level` (`minimal`, `low`, `medium`, `high`).

## Development

```terminal
uv run --with pytest --with-editable . python -m pytest
```

The tests build requests offline and never call Vertex.

## License

Apache-2.0, as upstream. See `LICENSE`. The changes from upstream are listed at the top of `llm_vertex_genai.py` and in the git history.
