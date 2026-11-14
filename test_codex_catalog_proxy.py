
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

module_path = Path(__file__).with_name('codex_catalog_proxy.py')
spec = importlib.util.spec_from_file_location('codex_catalog_proxy', module_path)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


def capability(chat=True, opts=None, param='reasoning_effort',
               switch='chat_template_kwargs.enable_thinking',
               think=True, default=True, effort_default=None):
    return {
        'chat_supported': chat,
        'modality': 'text_chat' if chat else 'embedding',
        'thinking': {
            'supported': think,
            'toggle': ({'parameter': switch, 'on': True, 'off': False}
                       if switch else None),
            'default': default,
        },
        'reasoning_effort': {'parameter': param, 'options': opts, 'default': effort_default},
    }


CAPABILITIES = {
    'Qwen/Qwen3.8-27B': capability(opts=['low', 'medium', 'xhigh'], effort_default='xhigh'),
    'deepseek-ai/DeepSeek-V4-Flash': capability(opts=['low', 'high', 'max'],
                                               switch='reasoning_effort', default=None),
    'google/gemma-4-26B-A4B-it': capability(opts=None, default=False, effort_default=None),
    'RiVault/Reasoning-Small': capability(opts=None, switch=None, think=None, default=None),
    'ByteDance-Seed/Seed-OSS-36B-Instruct': capability(opts=[], param=None,
        switch='thinking_budget', default=True),
    'meta-llama/Llama-3.2-3B-Instruct': capability(opts=[], param=None,
        switch=None, think=False, default=None),
    'RiVault/Embedding-Small': capability(chat=False, opts=[], param=None,
        switch=None, think=False, default=None),
    'unknown-chat-model': capability(chat=None, opts=None, param=None,
        switch=None, think=None, default=None),
}

UPSTREAM = {'data': [{'id': k} for k in CAPABILITIES] + [{'id': 'not-in-json'}]}
DEBUG = [{'slug': 'codex-demo', 'supported_reasoning_levels': [], 'display_name': 'Demo'}]


def build(upstream=UPSTREAM, overrides=None, caps=CAPABILITIES):
    return proxy.convert_models(upstream, DEBUG, overrides or {}, caps, mode='capabilities')['models']


def test_chat_only_filter():
    models = build()
    ids = [m['slug'] for m in models]
    assert ids == [
        'Qwen/Qwen3.8-27B', 'deepseek-ai/DeepSeek-V4-Flash',
        'google/gemma-4-26B-A4B-it', 'RiVault/Reasoning-Small',
        'ByteDance-Seed/Seed-OSS-36B-Instruct', 'meta-llama/Llama-3.2-3B-Instruct',
    ]
    assert [m['priority'] for m in models] == list(range(1, len(ids) + 1))


def test_effort_options_and_default():
    models = {m['slug']: m for m in build()}
    def levels(name):
        return [e['effort'] for e in models[name]['supported_reasoning_levels']]
    assert levels('Qwen/Qwen3.8-27B') == ['none', 'low', 'medium', 'xhigh']
    assert models['Qwen/Qwen3.8-27B']['default_reasoning_level'] == 'xhigh'
    assert levels('deepseek-ai/DeepSeek-V4-Flash') == ['none', 'low', 'high', 'max']
    assert levels('google/gemma-4-26B-A4B-it') == []
    assert levels('RiVault/Reasoning-Small') == []
    assert levels('ByteDance-Seed/Seed-OSS-36B-Instruct') == []
    assert levels('meta-llama/Llama-3.2-3B-Instruct') == []
    assert models['meta-llama/Llama-3.2-3B-Instruct']['default_reasoning_level'] is None


def test_default_non_thinking_chooses_none_when_supported():
    caps = {'x': capability(opts=['low', 'high'], default=False, effort_default='high')}
    m = build({'data': [{'id': 'x'}]}, caps=caps)[0]
    assert m['default_reasoning_level'] == 'none'
    assert [e['effort'] for e in m['supported_reasoning_levels']] == ['none', 'low', 'high']


def test_no_none_when_toggle_uses_budget_not_compatible():
    caps = {'x': capability(opts=['low', 'high'], switch='thinking_budget')}
    m = build({'data': [{'id': 'x'}]}, caps=caps)[0]
    assert [e['effort'] for e in m['supported_reasoning_levels']] == ['low', 'high']


def test_no_levels_when_reasoning_effort_parameter_is_custom():
    caps = {'x': capability(opts=['high'], param='chat_template_kwargs.thinking_level')}
    m = build({'data': [{'id': 'x'}]}, caps=caps)[0]
    assert m['supported_reasoning_levels'] == []


def test_context_and_non_capability_override():
    m = build({'data': [{'id': 'Qwen/Qwen3.8-27B', 'context_length': 32768}]},
              {'Qwen/Qwen3.8-27B': {'display_name': 'Qwen customized'}})[0]
    assert m['display_name'] == 'Qwen customized'
    assert m['context_window'] == 32768
    assert m['max_context_window'] == 32768


@pytest.mark.parametrize('key', ['slug', 'priority', 'default_reasoning_level',
                                 'supported_reasoning_levels', 'visibility'])
def test_protected_override(key):
    with pytest.raises(ValueError, match='cannot override'):
        build({'data': [{'id': 'Qwen/Qwen3.8-27B'}]},
              {'Qwen/Qwen3.8-27B': {key: 'invalid'}})


def test_no_false_reasoning_default_if_unverified():
    caps = {'x': capability(opts=['low', 'high'], effort_default=None)}
    m = build({'data': [{'id': 'x'}]}, caps=caps)[0]
    assert m['default_reasoning_level'] is None


def test_capabilities_loader(tmp_path, monkeypatch):
    p = tmp_path / 'caps.json'
    p.write_text(json.dumps({'models': CAPABILITIES}), encoding='utf-8')
    monkeypatch.setattr(proxy, 'MODEL_CAPABILITIES_FILE', str(p))
    assert proxy.get_capabilities()['Qwen/Qwen3.8-27B']['chat_supported'] is True
    p.write_text(json.dumps({'models': {'x': {'chat_supported': True}}}), encoding='utf-8')
    with pytest.raises(ValueError, match='missing thinking object'):
        proxy.get_capabilities()


def test_endpoint(monkeypatch, tmp_path):
    from httpx import Response, Request

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None
        async def get(self, url, headers=None):
            return Response(200, json=UPSTREAM, request=Request('GET', url))

    monkeypatch.setattr(proxy.httpx, 'AsyncClient', FakeAsyncClient)
    monkeypatch.setattr(proxy, 'get_debug_models', lambda: DEBUG)
    monkeypatch.setattr(proxy, 'get_capabilities', lambda: CAPABILITIES)
    monkeypatch.setattr(proxy, 'get_overrides', lambda: {})
    monkeypatch.setattr(proxy, 'MODEL_CATALOG_MODE', 'capabilities')
    resp = TestClient(proxy.app).get('/models.json')
    assert resp.status_code == 200
    assert len(resp.json()['models']) == 6


def test_endpoint_fails_closed_when_capabilities_unavailable(monkeypatch):
    monkeypatch.setattr(proxy, 'get_debug_models', lambda: DEBUG)
    monkeypatch.setattr(proxy, 'get_capabilities', lambda: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(proxy, 'get_overrides', lambda: {})
    monkeypatch.setattr(proxy, 'MODEL_CATALOG_MODE', 'capabilities')
    resp = TestClient(proxy.app).get('/models')
    assert resp.status_code == 502


# Legacy mode must be the default and must not require capabilities metadata.
def test_legacy_mode_is_default():
    assert proxy.MODEL_CATALOG_MODE == 'legacy'


def test_legacy_lists_all_upstream_without_loading_capabilities(monkeypatch):
    def reject_loading():
        raise AssertionError('get_capabilities must not run in legacy mode')
    monkeypatch.setattr(proxy, 'get_capabilities', reject_loading)
    data = proxy.convert_models(UPSTREAM, DEBUG, {}, mode='legacy')['models']
    assert [m['slug'] for m in data] == [r['id'] for r in UPSTREAM['data']]
    assert [m['priority'] for m in data] == list(range(1, len(data) + 1))
    for m in data:
        assert [x['effort'] for x in m['supported_reasoning_levels']] == [
            'none', 'minimal', 'low', 'medium', 'high', 'xhigh'
        ]
        assert m['default_reasoning_level'] == 'medium'


def test_legacy_default_effort_priority_and_overrides():
    upstream = {'data': [
        {'id': 'a', 'default_reasoning_level': 'high',
         'supported_reasoning_levels': [{'effort': 'high', 'description': 'Upstream high'}]},
        {'id': 'b', 'defaultReasoningEffort': 'low'},
        {'id': 'c', 'default_reasoning_level': 'invalid'},
    ]}
    debug = [{'slug': 'c', 'default_reasoning_level': 'xhigh'}]
    overrides = {'b': {'default_reasoning_level': 'none', 'visibility': 'hide'},
                 'a': {'slug': 'should-not-apply'}}
    models = proxy.convert_models(upstream, debug, overrides, mode='legacy')['models']
    models = {m['slug']: m for m in models}
    assert models['a']['default_reasoning_level'] == 'high'
    assert models['a']['supported_reasoning_levels'][4]['description'] == 'Upstream high'
    assert models['b']['default_reasoning_level'] == 'none'
    assert models['b']['visibility'] == 'hide'
    assert models['c']['default_reasoning_level'] == 'xhigh'
    assert models['a']['slug'] == 'a'


def test_legacy_rejects_overrides_of_six_efforts():
    with pytest.raises(ValueError, match='six levels required'):
        proxy.convert_models(
            {'data': [{'id': 'x'}]}, DEBUG,
            {'x': {'supported_reasoning_levels': []}}, mode='legacy',
        )


def test_legacy_endpoint_does_not_load_capabilities(monkeypatch):
    from httpx import Response, Request

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None
        async def get(self, url, headers=None):
            return Response(200, json=UPSTREAM, request=Request('GET', url))

    def reject_loading():
        raise AssertionError('get_capabilities must not be called')

    monkeypatch.setattr(proxy.httpx, 'AsyncClient', FakeAsyncClient)
    monkeypatch.setattr(proxy, 'MODEL_CATALOG_MODE', 'legacy')
    monkeypatch.setattr(proxy, 'get_debug_models', lambda: DEBUG)
    monkeypatch.setattr(proxy, 'get_capabilities', reject_loading)
    monkeypatch.setattr(proxy, 'get_overrides', lambda: {})
    response = TestClient(proxy.app).get('/models.json')
    assert response.status_code == 200
    assert len(response.json()['models']) == len(UPSTREAM['data'])
    assert response.json()['models'][-1]['slug'] == 'not-in-json'


def test_legacy_no_file_required_when_capabilities_missing(monkeypatch):
    monkeypatch.setattr(proxy, 'MODEL_CAPABILITIES_FILE', '/nonexistent/file.json')
    models = proxy.convert_models({'data': [{'id': 'test'}]}, DEBUG, {}, mode='legacy')
    assert models['models'][0]['slug'] == 'test'


def test_invalid_mode_rejected():
    with pytest.raises(ValueError, match='mode must be'):
        proxy.convert_models(UPSTREAM, DEBUG, {}, mode='invalid')


def deployment(name='public', **info):
    return {'model_name': name, 'model_info': info,
            'litellm_params': {'model': 'internal/provider', 'reasoning_effort': 'high'}}


def litellm_build(*entries, overrides=None):
    return proxy.convert_models({'data': list(entries)}, DEBUG, overrides or {},
                                mode='litellm')['models']


@pytest.mark.parametrize('info, included', [
    ({'mode': 'chat'}, True),
    ({'mode': 'responses'}, True),
    ({'mode': 'chat', 'supports_vision': True}, True),
    ({'supported_endpoints': ['/v1/responses']}, True),
    ({'supported_endpoints': ['/chat/completions']}, True),
    ({'mode': 'responses', 'supported_endpoints': ['/responses']}, True),
    ({}, False),
    ({'supports_vision': True, 'supports_function_calling': True}, False),
    ({'mode': 'completion'}, False),
    ({'mode': 'embedding', 'supported_endpoints': ['/v1/responses']}, False),
    ({'mode': 'chat', 'supported_endpoints': ['/v1/embeddings']}, False),
    ({'mode': 'chat', 'supported_endpoints': []}, False),
    ({'mode': 'chat', 'supported_endpoints': 'responses'}, False),
    ({'mode': 'chat', 'supported_endpoints': [None]}, False),
    *[({'mode': mode}, False) for mode in (
        'embedding', 'rerank', 'image_generation', 'audio_speech',
        'audio_transcription', 'realtime', 'unknown')],
])
def test_litellm_conversational_filter(info, included):
    assert bool(litellm_build(deployment(**info))) is included


@pytest.mark.parametrize('info, expected, default', [
    ({'reasoning_effort_levels': ['high', 'low', 'low', 'max', 'bad', None, {}],
      'default_reasoning_effort': 'low'}, ['low', 'high', 'max'], 'low'),
    ({'reasoning_effort_levels': []}, [], None),
    ({'reasoning_effort_levels': ['invalid', 123]}, [], None),
    ({'reasoning_effort_levels': 'high'}, [], None),
    ({'reasoning_effort_levels': None}, [], None),
    ({'supports_reasoning': True}, [], None),
    ({'supported_openai_params': ['reasoning_effort']}, [], None),
    ({}, [], None),  # litellm_params.reasoning_effort is not an enumeration
    ({'supports_low_reasoning_effort': True, 'supports_xhigh_reasoning_effort': True}, [], None),
    ({'reasoning_effort_levels': ['low'], 'supports_reasoning': False}, [], None),
    ({'reasoning_effort_levels': ['high'], 'default_reasoning_effort': 'low'}, ['high'], None),
    ({'reasoning_effort_levels': ['high'], 'default_reasoning_effort': []}, ['high'], None),
    ({'reasoning_effort_levels': ['high']}, ['high'], None),
    ({'reasoning_effort_levels': ['none', 'minimal', 'xhigh']}, ['none', 'minimal', 'xhigh'], None),
])
def test_litellm_reasoning(info, expected, default):
    model = litellm_build(deployment(mode='chat', **info))[0]
    assert [level['effort'] for level in model['supported_reasoning_levels']] == expected
    assert model['default_reasoning_level'] == default


def test_litellm_public_names_duplicates_order_and_exclusions(monkeypatch):
    monkeypatch.setattr(proxy, 'EXCLUDE_MODEL_IDS', {'hidden'})
    entries = [deployment('z', mode='chat', reasoning_effort_levels=['low', 'high'],
                          default_reasoning_effort='low'),
               deployment('a', mode='responses'), deployment('hidden', mode='chat'),
               deployment('z', mode='chat', reasoning_effort_levels=['low'],
                          default_reasoning_effort='low')]
    entries[0]['model_info']['id'] = 'internal-deployment-id'
    models = litellm_build(*entries)
    assert [m['slug'] for m in models] == ['z', 'a']
    assert [m['priority'] for m in models] == [1, 2]
    assert [level['effort'] for level in models[0]['supported_reasoning_levels']] == ['low']
    assert models[0]['default_reasoning_level'] == 'low'


@pytest.mark.parametrize('second', [
    {'mode': 'chat'}, {'mode': 'chat', 'reasoning_effort_levels': []},
    {'mode': 'chat', 'reasoning_effort_levels': ['low'], 'supports_reasoning': False},
    {'mode': 'chat', 'reasoning_effort_levels': ['high']},
])
def test_litellm_duplicate_reasoning_intersection(second):
    models = litellm_build(deployment(mode='chat', reasoning_effort_levels=['low']),
                           deployment(**second))
    assert models[0]['supported_reasoning_levels'] == []
    assert models[0]['default_reasoning_level'] is None


def test_litellm_duplicate_unknown_chat_excluded():
    assert litellm_build(deployment(mode='chat'), deployment()) == []


def test_litellm_duplicate_conflicting_defaults():
    models = litellm_build(*[deployment(mode='chat', reasoning_effort_levels=['low', 'high'],
                                        default_reasoning_effort=value)
                            for value in ('low', 'high')])
    assert models[0]['default_reasoning_level'] is None


def test_litellm_override_cannot_advertise_unverified_levels():
    with pytest.raises(ValueError, match='cannot override'):
        litellm_build(deployment(mode='chat'), overrides={'public': {'default_reasoning_level': 'high'}})


@pytest.mark.parametrize('payload', [None, {}, {'data': {}}, {'data': [None]},
                                      {'data': [{}]}, {'data': [{'model_name': 1}]},
                                      {'data': [{'model_name': 'public', 'model_info': []}]}])
def test_litellm_malformed_payload(payload):
    with pytest.raises(ValueError):
        proxy.convert_models(payload, DEBUG, {}, mode='litellm')


@pytest.mark.parametrize('path', ['/models', '/models.json'])
@pytest.mark.parametrize('failure', [None, 'status', 'json', 'shape', 'timeout'])
def test_litellm_endpoint(monkeypatch, path, failure):
    from httpx import Response, Request, ReadTimeout

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            assert kwargs == {'timeout': 4.0, 'follow_redirects': False}
        async def __aenter__(self): return self
        async def __aexit__(self, *args): return None
        async def get(self, url, headers):
            assert url == 'https://litellm.example/custom/model/info'
            assert headers == {'Authorization': 'Bearer test-key'}
            request = Request('GET', url)
            if failure == 'timeout':
                raise ReadTimeout('timeout', request=request)
            if failure == 'json':
                return Response(200, text='invalid json', request=request)
            return Response(401 if failure == 'status' else 200,
                            json={} if failure == 'shape' else {
                                'data': [deployment(mode='chat')]}, request=request)

    def reject_capabilities():
        raise AssertionError('LiteLLM must not load a capabilities file')

    monkeypatch.setattr(proxy.httpx, 'AsyncClient', FakeAsyncClient)
    monkeypatch.setattr(proxy, 'MODEL_CATALOG_MODE', 'litellm')
    monkeypatch.setattr(proxy, 'UPSTREAM_MODEL_INFO_URL', 'https://litellm.example/custom/model/info')
    monkeypatch.setattr(proxy, 'UPSTREAM_API_KEY', 'test-key')
    monkeypatch.setattr(proxy, 'get_debug_models', lambda: DEBUG)
    monkeypatch.setattr(proxy, 'get_overrides', lambda: {})
    monkeypatch.setattr(proxy, 'get_capabilities', reject_capabilities)
    response = TestClient(proxy.app).get(path)
    assert response.status_code == (502 if failure else 200)
    if not failure:
        assert response.json()['models'][0]['slug'] == 'public'
        assert response.json()['models'][0]['supported_reasoning_levels'] == []


def test_litellm_environment_selection(monkeypatch):
    monkeypatch.setenv('MODEL_CATALOG_MODE', ' LITELLM ')
    monkeypatch.setenv('UPSTREAM_MODEL_INFO_URL', 'https://example/model/info')
    selected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selected)
    assert selected.MODEL_CATALOG_MODE == 'litellm'
    assert selected.UPSTREAM_MODEL_INFO_URL == 'https://example/model/info'
