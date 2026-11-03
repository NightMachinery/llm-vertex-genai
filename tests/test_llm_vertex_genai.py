# Tests for llm-vertex-genai. They build requests offline; nothing calls Vertex.
from types import SimpleNamespace

import llm_vertex_genai
from llm_vertex_genai import Vertex


def registered(monkeypatch, extra=None):
    if extra is None:
        monkeypatch.delenv('LLM_VERTEX_MODELS', raising=False)
    else:
        monkeypatch.setenv('LLM_VERTEX_MODELS', extra)
    models = []
    llm_vertex_genai.register_models(models.append)
    return [m.model_id for m in models]


def test_default_models_registered(monkeypatch):
    ids = registered(monkeypatch)
    assert 'vertex-gemini-3.8-flash' in ids
    assert all(i.startswith('vertex-') for i in ids)


def test_extra_models_from_env(monkeypatch):
    ids = registered(monkeypatch, ' gemini-9-test , gemini-3.8-flash,')
    assert ids.count('vertex-gemini-3.8-flash') == 1
    assert ids[-1] == 'vertex-gemini-9-test'


def test_model_name():
    assert Vertex('vertex-gemini-2.0-flash-001').model_name == 'gemini-2.0-flash-001'


def test_registration_does_not_create_client(monkeypatch):
    llm_vertex_genai._clients.clear()
    registered(monkeypatch)
    assert llm_vertex_genai._clients == {}


class FakeAttachment:
    def __init__(self, mime_type, data=b'x', url=None):
        self.mime_type, self.data, self.url = mime_type, data, url

    def resolve_type(self):
        return self.mime_type

    def content_bytes(self):
        return self.data


def fake_prompt(text, attachments=(), system=None, **options):
    return SimpleNamespace(
        prompt=text, attachments=list(attachments), system=system,
        options=Vertex.Options(**options),
    )


def test_attachments_and_mime_aliases():
    content = Vertex('vertex-x').user_content(
        fake_prompt('hi', [FakeAttachment('image/png'), FakeAttachment('application/ogg')]))
    assert [p.inline_data.mime_type for p in content.parts[:2]] == ['image/png', 'audio/ogg']
    assert content.parts[2].text == 'hi'


def test_gcs_attachment_uses_uri():
    content = Vertex('vertex-x').user_content(
        fake_prompt('', [FakeAttachment('video/mp4', url='gs://b/v.mp4')]))
    assert content.parts[0].file_data.file_uri == 'gs://b/v.mp4'
    assert len(content.parts) == 1


def test_history_includes_model_turns():
    prev = SimpleNamespace(prompt=fake_prompt('first'), text_or_raise=lambda: 'answer')
    contents = Vertex('vertex-x').build_contents(
        fake_prompt('second'), SimpleNamespace(responses=[prev]))
    assert [c.role if hasattr(c, 'role') else c['role'] for c in contents] == ['user', 'model', 'user']
    assert contents[1]['parts'][0]['text'] == 'answer'


def test_config():
    config = Vertex('vertex-x').build_config(
        fake_prompt('q', system='be brief', temperature=0.2, thinking_level='low'))
    assert config.temperature == 0.2
    assert config.system_instruction == 'be brief'
    assert config.thinking_config.thinking_level.value == 'LOW'
    assert config.top_k is None
