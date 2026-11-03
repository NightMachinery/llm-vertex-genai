# Modified by NightMachinery from avoidik/llm-vertex (itself a fork of
# justyns/llm-vertex): attachments, conversation history, configurable model
# list, lazy client, `global` location default, usage reporting.
import os
from typing import Optional

import llm
from pydantic import Field

# Source: https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models
DEFAULT_MODELS = [
    'gemini-4-argon',
    'gemini-3.8-flash',
    'gemini-3.1-pro-preview',
    'gemini-3-flash-preview',
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
]

ATTACHMENT_TYPES = {
    # Images
    'image/png', 'image/jpeg', 'image/webp', 'image/heic', 'image/heif',
    # Audio
    'audio/wav', 'audio/x-wav', 'audio/mp3', 'audio/mpeg', 'audio/aiff',
    'audio/aac', 'audio/ogg', 'audio/flac', 'audio/webm', 'application/ogg',
    # Video
    'video/mp4', 'video/mpeg', 'video/quicktime', 'video/x-msvideo',
    'video/x-flv', 'video/mpg', 'video/webm', 'video/wmv', 'video/3gpp',
    # Documents
    'application/pdf', 'text/plain',
}

#: MIME types Vertex rejects, mapped to the equivalent it accepts.
MIME_ALIASES = {
    'application/ogg': 'audio/ogg',
    'audio/x-wav': 'audio/wav',
}


def model_names():
    """The default models plus any listed in `LLM_VERTEX_MODELS`, comma-separated."""
    extra = [m.strip() for m in os.getenv('LLM_VERTEX_MODELS', '').split(',')]
    names = list(DEFAULT_MODELS)
    for m in extra:
        if m and m not in names:
            names.append(m)
    return names


@llm.hookimpl
def register_models(register):
    for model in model_names():
        register(Vertex(f'vertex-{model}'))


_clients = {}


def get_client():
    """One client per (project, location), created on first use.

    Creating it at registration time would make every `llm` invocation pay for
    Google auth, even ones that never touch Vertex.
    """
    from google import genai

    project = os.getenv('VERTEX_PROJECT_ID') or os.getenv('GOOGLE_CLOUD_PROJECT')
    location = os.getenv('VERTEX_LOCATION') or 'global'
    key = (project, location)
    if key not in _clients:
        _clients[key] = genai.Client(vertexai=True, project=project, location=location)
    return _clients[key]


class Vertex(llm.Model):
    can_stream = True
    attachment_types = ATTACHMENT_TYPES

    class Options(llm.Options):
        max_output_tokens: Optional[int] = None
        temperature: Optional[float] = None
        top_p: Optional[float] = None
        top_k: Optional[int] = None
        thinking_level: Optional[str] = Field(
            default=None,
            description='minimal, low, medium or high (Gemini 3 and later)',
        )

    def __init__(self, model_id):
        self.model_id = model_id
        self.model_name = model_id.removeprefix('vertex-')

    def __str__(self):
        return f'Vertex: {self.model_id}'

    def execute(self, prompt, stream, response, conversation):
        client = get_client()
        contents = self.build_contents(prompt, conversation)
        config = self.build_config(prompt)

        last = None
        if stream:
            for chunk in client.models.generate_content_stream(
                model=self.model_name, contents=contents, config=config
            ):
                last = chunk
                if chunk.text:
                    yield chunk.text
        else:
            last = client.models.generate_content(
                model=self.model_name, contents=contents, config=config
            )
            yield last.text or ''

        self.set_usage(response, last)

    def build_contents(self, prompt, conversation):
        contents = []
        if conversation:
            for prev in conversation.responses:
                contents.append(self.user_content(prev.prompt))
                contents.append({'role': 'model', 'parts': [{'text': prev.text_or_raise()}]})
        contents.append(self.user_content(prompt))
        return contents

    def user_content(self, prompt):
        from google.genai import types

        parts = []
        for attachment in prompt.attachments or []:
            mime_type = attachment.resolve_type()
            mime_type = MIME_ALIASES.get(mime_type, mime_type)
            if attachment.url and attachment.url.startswith('gs://'):
                parts.append(types.Part.from_uri(file_uri=attachment.url, mime_type=mime_type))
            else:
                parts.append(types.Part.from_bytes(data=attachment.content_bytes(), mime_type=mime_type))
        if prompt.prompt:
            parts.append(types.Part.from_text(text=prompt.prompt))
        return types.Content(role='user', parts=parts)

    def build_config(self, prompt):
        from google.genai import types

        opts = prompt.options.model_dump()
        thinking_level = opts.pop('thinking_level', None)
        config = {k: v for k, v in opts.items() if v is not None}
        if thinking_level:
            config['thinking_config'] = types.ThinkingConfig(thinking_level=thinking_level.upper())
        if prompt.system:
            config['system_instruction'] = prompt.system
        return types.GenerateContentConfig(**config)

    def set_usage(self, response, last):
        usage = getattr(last, 'usage_metadata', None)
        if not usage:
            return
        output = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        response.set_usage(input=usage.prompt_token_count, output=output)
