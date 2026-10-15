import json
import shutil
import asyncio
import threading
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import httpx

from demo.backend.main import MAX_UPLOAD, create_app
from demo.prepare import GRID, MODELS, PARAMS, SCHEMA, frozen_config, prepare_artifacts
from src.data.raw_to_interim import INTERIM_SCHEMA
from src.evaluation.manifest import file_sha256
from src.models.pipeline import AnomalyPipeline
from src.models.registry import create_pipeline


def test_prepare_artifacts_from_processed_matrix(tmp_path):
    _, features = frozen_config()
    rng = np.random.default_rng(7)
    rows = []
    for day in range(1, 43):
        for user in ('UserA', 'UserB', 'UserC'):
            row = dict(DomainName='dom', UserName=user, day=day, entity_type='User')
            row.update(zip(features, rng.normal(size=len(features)).tolist()))
            rows.append(row)
    processed = tmp_path / 'processed.parquet'
    pl.DataFrame(rows).write_parquet(processed)
    interim = tmp_path / 'interim'
    for day in range(1, 43):
        for eid in (4624, 4625):
            f = interim / f'event_{eid}' / f'event_{eid}_day-{day:02d}.parquet'
            f.parent.mkdir(parents=True, exist_ok=True); f.touch()
    output = prepare_artifacts(processed, interim, tmp_path / 'artifacts')
    metadata = json.loads((output / 'metadata.json').read_text())
    assert metadata['feature_names'] == features
    assert metadata['train_start'] == 1 and metadata['train_end'] == 42
    assert metadata['context_archive_end'] == 42
    assert all((output / f'{name}.joblib').is_file() for name in MODELS)


def event(day, user, eid=4624):
    values = {field.name: None for field in INTERIM_SCHEMA}
    values.update(Time=(day - 1) * 86400 + 36000, EventID=eid, UserName=user,
                  DomainName='DOM', LogHost='Host1', LogonType=3, Source='Host2',
                  LogonID='1', AuthenticationPackage='NTLM')
    return values


def write_events(path, events):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(events, schema=INTERIM_SCHEMA)
    pq.write_table(table, path)


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    folder = tmp_path_factory.mktemp('demo-artifacts')
    _, features = frozen_config()
    rng = np.random.default_rng(42)
    train = rng.normal(size=(80, len(features)))
    metadata = {'format': 'ueba-demo-artifacts/1', 'feature_names': features,
                'feature_schema_sha256': file_sha256(SCHEMA), 'train_start': 1, 'train_end': 42,
                'segment': 'User', 'seed': 42,
                'grid_sha256': file_sha256(GRID), 'params_sha256': file_sha256(PARAMS),
                'context_archive_end': 42, 'feature_schema_version': 'v4.1',
                'model_version': 'test', 'models': {}}
    scores_before = {}
    for name in MODELS:
        params = {'preprocessing': {'scaler': 'standard' if name == 'one_class_svm' else 'robust'},
                  'isolation_forest': {'n_estimators': 20, 'max_samples': 32},
                  'local_outlier_factor': {'n_neighbors': 5},
                  'one_class_svm': {'n_components': 20, 'nu': .05}}
        pipeline = create_pipeline(name, params=params, feature_names=features, contamination=.05, random_state=42)
        pipeline.fit(train)
        scores_before[name] = pipeline.score(train[:3])
        pipeline.save(folder / f'{name}.joblib')
        np.testing.assert_allclose(AnomalyPipeline.load(folder / f'{name}.joblib').score(train[:3]), scores_before[name])
        metadata['models'][name] = {'threshold': pipeline.model.threshold_}
    (folder / 'metadata.json').write_text(json.dumps(metadata))
    pl.DataFrame({'DomainName':['dom'], 'UserName':['UserB'], 'day':[30]}).write_parquet(folder / 'historical_activity.parquet')
    return folder


@pytest.fixture
def client(tmp_path, trained):
    interim = tmp_path / 'interim'
    for day in range(36, 43):
        write_events(interim / 'event_4624' / f'event_4624_day-{day:02d}.parquet', [event(day, 'UserA')])
        write_events(interim / 'event_4625' / f'event_4625_day-{day:02d}.parquet', [])
    app = create_app(trained, interim)
    app.state.demo_service.load()
    return app


def request(app, method, path, **kwargs):
    async def send():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            return await client.request(method, path, **kwargs)
    return asyncio.run(send())


def upload(events):
    import io
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(events, schema=INTERIM_SCHEMA), buf)
    return buf.getvalue()


def test_health_and_upload(client):
    page = request(client, 'GET', '/')
    assert page.status_code == 200 and 'UEBA Log Analyzer' in page.text
    assert request(client, 'GET', '/static/app.js').status_code == 200
    health = request(client, 'GET', '/api/health')
    assert health.status_code == 200
    assert all(health.json()['models'].values())
    response = request(client, 'POST', '/api/analyze', files={'file': ('batch.parquet', upload([event(43, 'UserA'), event(43, 'UserB'), event(43, 'Machine$'), event(43, 'administrator')]), 'application/octet-stream')})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['totals']['events'] == 4
    assert body['totals']['scored_user_account_days'] == 2
    assert body['totals']['unscored_by_type']['Machine'] == 1
    assert body['totals']['unscored_by_type']['Admin'] == 1
    assert body['totals']['account_days'] == 4
    assert sum(body['agreement'].values()) == 2
    assert all(row['day'] == 43 for row in body['top_50'])
    assert all(len([k for k in body['feature_names'] if k in row]) == 22 for row in body['top_50'])
    userb = next(r for r in body['top_50'] if r['UserName'] == 'UserB')
    assert userb['days_since_last_activity'] == 13
    for name in MODELS:
        m = body['models'][name]
        assert m['alert_rate'] == m['alerts'] / 2
        assert all(row[name]['alert'] == (row[name]['score'] >= m['threshold']) for row in body['top_50'])
    assert all(row['agreement_count'] == sum(row[name]['alert'] for name in MODELS) for row in body['top_50'])
    assert not list(client.state.demo_service.interim_dir.parent.glob('ueba-*'))


def test_invalid_inputs(client, tmp_path, monkeypatch):
    response = request(client, 'POST', '/api/analyze', files={'file': ('bad.csv', b'wrong', 'text/csv')})
    assert response.status_code == 400 and response.json()['error'] == 'invalid_file'
    response = request(client, 'POST', '/api/analyze', files={'file': ('bad.parquet', b'wrong', 'application/octet-stream')})
    assert response.status_code == 400 and response.json()['error'] == 'invalid_file'
    bad = pa.Table.from_pydict({'Time':[42*86400]}, schema=pa.schema([('Time', pa.int64())]))
    import io
    buf = io.BytesIO(); pq.write_table(bad, buf)
    response = request(client, 'POST', '/api/analyze', files={'file': ('bad.parquet', buf.getvalue(), 'application/octet-stream')})
    assert response.status_code == 400 and response.json()['error'] == 'invalid_schema'
    good = pa.Table.from_pylist([event(43, 'UserA')], schema=INTERIM_SCHEMA)
    bad_time = good.set_column(0, 'Time', pa.array(['not-a-time']))
    buf = io.BytesIO(); pq.write_table(bad_time, buf)
    response = request(client, 'POST', '/api/analyze', files={'file': ('wrong-type.parquet', buf.getvalue(), 'application/octet-stream')})
    assert response.status_code == 400 and response.json()['error'] == 'invalid_schema'
    response = request(client, 'POST', '/api/analyze', files={'file': ('train.parquet', upload([event(42, 'UserA')]), 'application/octet-stream')})
    assert response.status_code == 400 and response.json()['error'] == 'invalid_window'
    response = request(client, 'POST', '/api/analyze', files={'file': ('no-context.parquet', upload([event(44, 'UserA')]), 'application/octet-stream')})
    assert response.status_code == 400 and response.json()['error'] == 'missing_context'
    assert 'Day 43' in response.json()['message']
    monkeypatch.setattr('demo.backend.main.MAX_UPLOAD', 1)
    response = request(client, 'POST', '/api/analyze', files={'file': ('big.parquet', b'ab', 'application/octet-stream')})
    assert response.status_code == 413 and response.json()['error'] == 'file_too_large'
    assert not list(client.state.demo_service.interim_dir.parent.glob('ueba-*'))


def test_unavailable(tmp_path, trained):
    app = create_app(tmp_path / 'absent', tmp_path / 'interim')
    assert request(app, 'GET', '/api/health').status_code == 503
    response = request(app, 'POST', '/api/analyze', files={'file': ('x.parquet', b'x', 'application/octet-stream')})
    assert response.status_code == 503 and response.json()['error'] == 'artifact_unavailable'
    partial = tmp_path / 'partial'; partial.mkdir()
    for name in ('metadata.json', 'historical_activity.parquet', 'isolation_forest.joblib'):
        shutil.copyfile(trained / name, partial / name)
    app = create_app(partial, tmp_path / 'interim')
    with pytest.raises(FileNotFoundError):
        app.state.demo_service.load()
    health = request(app, 'GET', '/api/health')
    assert health.status_code == 503
    assert health.json()['models'] == {'isolation_forest': True, 'local_outlier_factor': False, 'one_class_svm': False}
    wrong = tmp_path / 'wrong'; wrong.mkdir()
    metadata = json.loads((trained / 'metadata.json').read_text())
    metadata['feature_names'].reverse()
    (wrong / 'metadata.json').write_text(json.dumps(metadata))
    app = create_app(wrong, tmp_path / 'interim')
    with pytest.raises(ValueError, match='22 đặc trưng'):
        app.state.demo_service.load()
    assert request(app, 'GET', '/api/health').status_code == 503


def test_lifespan_loads_artifacts_once(tmp_path, trained, monkeypatch):
    app = create_app(trained, tmp_path / 'interim')
    service = app.state.demo_service
    calls = 0
    original = service.load
    def counted():
        nonlocal calls
        calls += 1
        original()
    monkeypatch.setattr(service, 'load', counted)
    async def check():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as http:
                assert (await http.get('/api/health')).status_code == 200
                assert (await http.get('/api/health')).status_code == 200
    asyncio.run(check())
    assert calls == 1


def test_inference_never_fits(client, monkeypatch):
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import RobustScaler, StandardScaler
    from src.models.base import BaseAnomalyModel
    def fail(*args, **kwargs):
        raise AssertionError('fit called during inference')
    for cls in [SimpleImputer, RobustScaler, StandardScaler, BaseAnomalyModel]:
        monkeypatch.setattr(cls, 'fit', fail)
        if cls is not BaseAnomalyModel:
            monkeypatch.setattr(cls, 'fit_transform', fail)
    response = request(client, 'POST', '/api/analyze', files={'file': ('batch.parquet', upload([event(43, 'UserA')]), 'application/octet-stream')})
    assert response.status_code == 200, response.text


def test_future_day_does_not_change_earlier_features(client):
    one = request(client, 'POST', '/api/analyze', files={'file': ('one.parquet', upload([event(43, 'UserA')]), 'application/octet-stream')})
    two = request(client, 'POST', '/api/analyze', files={'file': ('two.parquet', upload([event(43, 'UserA'), event(44, 'UserA')]), 'application/octet-stream')})
    assert one.status_code == two.status_code == 200
    left = one.json()['top_50'][0]
    right = next(row for row in two.json()['top_50'] if row['day'] == 43)
    assert {name: left[name] for name in one.json()['feature_names']} == {name: right[name] for name in two.json()['feature_names']}


def test_machine_only_batch_has_no_scores(client):
    response = request(client, 'POST', '/api/analyze', files={'file': ('machine.parquet', upload([event(43, 'Machine$')]), 'application/octet-stream')})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['totals']['scored_user_account_days'] == 0
    assert body['totals']['unscored_by_type']['Machine'] == 1
    assert body['top_50'] == []
    assert all(model['alert_rate'] == 0 for model in body['models'].values())


def test_busy_response(client, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = client.state.demo_service.analyze
    def slow(*args):
        started.set()
        release.wait(5)
        return original(*args)
    monkeypatch.setattr(client.state.demo_service, 'analyze', slow)
    async def two_requests():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=client), base_url='http://test') as http:
            first = asyncio.create_task(http.post('/api/analyze', files={'file': ('one.parquet', upload([event(43, 'UserA')]))}))
            while not started.is_set():
                await asyncio.sleep(.01)
            second = await http.post('/api/analyze', files={'file': ('two.parquet', upload([event(43, 'UserA')]))})
            release.set()
            return second, await first
    second, first = asyncio.run(two_requests())
    assert second.status_code == 409 and second.json()['error'] == 'server_busy'
    assert first.status_code == 200


def test_internal_error_does_not_leak_traceback(client, monkeypatch):
    def fail(*args):
        raise RuntimeError('sensitive internal detail')
    monkeypatch.setattr(client.state.demo_service, 'analyze', fail)
    response = request(client, 'POST', '/api/analyze', files={'file': ('one.parquet', upload([event(43, 'UserA')]))})
    assert response.status_code == 500
    assert response.json() == {'status': 'error', 'error': 'internal_error', 'message': 'Không thể phân tích batch này.'}
    assert 'sensitive internal detail' not in response.text
    assert not list(client.state.demo_service.interim_dir.parent.glob('ueba-*'))
