import json

import httpx

import diagnose_litellm as diagnostic


def entry(name='public', **info):
    return {'model_name': name, 'model_info': {'mode': 'chat', **info},
            'litellm_params': {'api_key': 'secret', 'model': 'private-deployment'}}


def test_diagnostic_reasons_and_output_fields(monkeypatch):
    monkeypatch.setattr(diagnostic.proxy, 'get_overrides', lambda: {})
    monkeypatch.setattr(diagnostic.proxy, 'EXCLUDE_MODEL_IDS', set())
    payload = {'data': [
        entry('missing'), entry('unsupported', supports_reasoning=False),
        entry('duplicate', reasoning_effort_levels=['low']),
        entry('duplicate', reasoning_effort_levels=['high']),
        entry('empty', reasoning_effort_levels=[]),
        entry('invalid', reasoning_effort_levels=['unknown']),
        entry('supported', reasoning_effort_levels=['low', 'high'], default_reasoning_effort='low'),
    ]}
    rows = diagnostic.diagnose(payload, [])
    by_name = {row['model_name']: row for row in rows}
    assert 'missing' in by_name['missing']['explanation'][0]
    assert 'explicitly unsupported' in by_name['unsupported']['explanation'][0]
    assert 'intersection is empty' in by_name['duplicate']['explanation'][0]
    assert 'is empty' in by_name['empty']['explanation'][0]
    assert 'no valid' in by_name['invalid']['explanation'][0]
    assert by_name['supported']['codex_reasoning_choices'] == ['low', 'high']
    assert by_name['supported']['codex_default_reasoning_level'] == 'low'
    for row in rows:
        assert set(row) == {'model_name', 'deployments', 'codex_reasoning_choices',
                            'codex_default_reasoning_level', 'explanation'}
        for info in row['deployments']:
            assert set(info) == {'mode', 'supported_endpoints', 'reasoning_effort_levels',
                                 'default_reasoning_effort'}
    assert 'secret' not in json.dumps(rows)
    assert 'private-deployment' not in json.dumps(rows)


def test_diagnostic_intended_names_and_exclusions(monkeypatch):
    monkeypatch.setattr(diagnostic.proxy, 'get_overrides', lambda: {})
    monkeypatch.setattr(diagnostic.proxy, 'EXCLUDE_MODEL_IDS', {'hidden'})
    rows = diagnostic.diagnose({'data': [entry('hidden'), entry('other'),
                                        entry('unknown', mode='embedding')]},
                               ['absent', 'hidden', 'unknown'])
    assert [row['model_name'] for row in rows] == ['absent', 'hidden', 'unknown']
    assert 'absent' in rows[0]['explanation'][0]
    assert 'EXCLUDE_MODEL_IDS' in rows[1]['explanation'][0]
    assert 'conversational metadata' in rows[2]['explanation'][0]


def test_diagnostic_http_error_redacts_details(monkeypatch, capsys):
    monkeypatch.setenv('UPSTREAM_MODEL_INFO_URL', 'https://example/model/info')
    monkeypatch.setattr(diagnostic.proxy, 'UPSTREAM_MODEL_INFO_URL', 'https://example/model/info')
    monkeypatch.setattr(diagnostic.proxy, 'UPSTREAM_API_KEY', 'secret')

    class FakeClient:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, headers):
            assert headers == {'Authorization': 'Bearer secret'}
            raise httpx.ConnectError('secret URL and response details')

    monkeypatch.setattr(diagnostic.httpx, 'Client', FakeClient)
    assert diagnostic.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ''
    assert captured.err == 'Diagnostic failed: ConnectError\n'
