"""Check plotted numbers, row alignment, scope and real exported artifacts."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.evaluation.plots import compute_plot_tables, load_plot_run, plot_benchmark_results


def make_run(root, offset=0, days=(43, 44), with_labels=True):
    root.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame({
        'DomainName': ['D'] * 8 + ['M'] * 8, 'UserName': [f'u{i}' for i in range(8)] * 2,
        'day': [days[0]] * 4 + [days[1]] * 4 + [days[0]] * 4 + [days[1]] * 4,
        'segment': ['User'] * 8 + ['Machine'] * 8,
        'label': [0, 1, 0, 0, 1, 0, 0, 0] * 2,
        'isolation_forest_score': (np.arange(16, 0, -1) / 16 + offset).tolist(),
    })
    if not with_labels:
        frame = frame.drop('label')
    frame.write_parquet(root / 'anomaly_scores.parquet')
    pl.DataFrame({'segment': ['User', 'Machine'], 'model': ['isolation_forest'] * 2,
                  'seed': [42, 42], 'fit_seconds': [1.25, 2.5], 'score_seconds': [.25, .5]}).write_csv(root / 'benchmark_summary.csv')
    metadata = {'split': {'split_day': 42}, 'extra': {'eval_days': list(days)},
                'config': {'feature_names': ['f1'], 'feature_set': 'core'},
                'models': [{'segment': s, 'model': {'model_name': 'isolation_forest',
                            'native_params': {'n_estimators': 10}, 'random_state': 42}} for s in ('User', 'Machine')]}
    (root / 'run_manifest.json').write_text(json.dumps(metadata))
    return frame


def test_numbers_and_daily_budget_are_not_global_topk(temp_artifact_dir):
    make_run(temp_artifact_dir)
    tables = compute_plot_tables([load_plot_run(temp_artifact_dir)], [1, 2, 10], [1, 2])
    for row in tables['metrics'].to_dict('records'):
        assert row['n_positive'] == 2 and row['n_eval'] == 8
        assert row['precision_at_1'] == 0 and row['precision_at_2'] == .5
        assert np.isnan(row['precision_at_10'])
        assert row['ap'] == pytest.approx((1 / 2 + 2 / 5) / 2)
    daily = tables['daily_recall']
    assert daily[daily.alerts_per_day == 1].recall.tolist() == [.5, .5]
    assert daily[daily.alerts_per_day == 2].recall.tolist() == [1, 1]
    assert set(tables['metrics'].segment) == {'User', 'Machine'}


def test_baselines_align_by_keys_and_exclude_train(temp_artifact_dir):
    frame = make_run(temp_artifact_dir)
    baseline = frame.select(['DomainName', 'UserName', 'day', 'segment']).with_columns(
        pl.Series('random', np.arange(16) / 16), pl.lit('eval').alias('split'))
    training = baseline.head(1).with_columns(pl.lit(1, pl.Int64).alias('day'), pl.lit('train').alias('split'))
    dest = temp_artifact_dir / 'baselines'
    dest.mkdir()
    pl.concat([baseline.reverse(), training]).write_parquet(dest / 'baseline_scores.parquet')
    run = load_plot_run(temp_artifact_dir)
    assert run['frame'].height == 16
    assert run['frame']['baseline:random_score'].to_list() == (np.arange(16) / 16).tolist()
    baseline.head(15).write_parquet(dest / 'baseline_scores.parquet')
    with pytest.raises(ValueError, match='exactly the same'):
        load_plot_run(temp_artifact_dir)


def test_external_labels_exclusions_and_disagreement(temp_artifact_dir):
    frame = make_run(temp_artifact_dir)
    labels = frame.select(['DomainName', 'UserName', 'day', 'label']).rename({'label': 'is_anomaly'})
    labels = labels.with_columns((pl.col('UserName') == 'u0').alias('eval_exclude'),
                                pl.when(pl.col('is_anomaly') == 1).then(pl.lit('brute_force')).otherwise(None).alias('scenario'))
    path = temp_artifact_dir / 'labels.parquet'
    labels.write_parquet(path)
    run = load_plot_run(temp_artifact_dir, path)
    assert run['frame'].height == 14
    assert run['frame']['scenario'].drop_nulls().unique().to_list() == ['brute_force']
    labels.with_columns(pl.lit(0).alias('is_anomaly')).write_parquet(path)
    with pytest.raises(ValueError, match='disagree'):
        load_plot_run(temp_artifact_dir, path)


@pytest.mark.parametrize('problem', ['duplicates', 'nan', 'train', 'unlabeled'])
def test_reject_invalid_artifacts(temp_artifact_dir, problem):
    frame = make_run(temp_artifact_dir)
    if problem == 'duplicates':
        frame = pl.concat([frame, frame.head(1)])
    elif problem == 'nan':
        frame = frame.with_columns(pl.lit(float('nan')).alias('isolation_forest_score'))
    elif problem == 'train':
        frame = frame.with_columns(pl.lit(1).alias('day'))
    else:
        frame = frame.drop('label')
    frame.write_parquet(temp_artifact_dir / 'anomaly_scores.parquet')
    with pytest.raises(ValueError):
        load_plot_run(temp_artifact_dir)


def test_single_class_is_undefined_for_ap_and_pr(temp_artifact_dir):
    frame = make_run(temp_artifact_dir).with_columns(pl.lit(0).alias('label'))
    frame.write_parquet(temp_artifact_dir / 'anomaly_scores.parquet')
    tables = compute_plot_tables([load_plot_run(temp_artifact_dir)], [1], [1])
    assert tables['metrics'].ap.isna().all()
    assert tables['pr_curves'].empty
    assert tables['daily_recall'].recall.isna().all()


def test_render_multiple_runs_and_export_uncertainty(temp_artifact_dir):
    first, second = temp_artifact_dir / 'seed42', temp_artifact_dir / 'seed7'
    make_run(first)
    make_run(second, offset=.001)
    result = plot_benchmark_results([first, second], output_dir=temp_artifact_dir / 'figures',
                                    precision_ks=[1, 2], daily_budgets=[1, 2], formats=['png', 'pdf'], dpi=60)
    assert len(result['figures']) == 20  # 5 figures x 2 segments x 2 formats
    for path in result['figures']:
        assert Path(path).stat().st_size > 1000
    summary = result['tables']['metrics_summary']
    assert summary.ap_count.tolist() == [2, 2]
    assert summary.ap_std.tolist() == [0, 0]
    assert result['manifest']['reference_pr_run'] == str(first)
    assert (temp_artifact_dir / 'figures' / 'plot_manifest.json').is_file()
    # Counts use full result directories, not the benchmark's stability sample seeds.
    assert result['manifest']['n_runs'] == 2


def test_reject_duplicate_runs_and_mixed_blocks_or_parameters(temp_artifact_dir):
    first, second = temp_artifact_dir / 'a', temp_artifact_dir / 'b'
    make_run(first)
    make_run(second)
    with pytest.raises(ValueError, match='Duplicate score'):
        plot_benchmark_results([first, second])
    make_run(second, offset=.001, days=(52, 53))
    with pytest.raises(ValueError, match='same evaluation days'):
        plot_benchmark_results([first, second])
    make_run(second, offset=.001)
    metadata_path = second / 'run_manifest.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['models'][0]['model']['native_params']['n_estimators'] = 20
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='different model parameters'):
        plot_benchmark_results([first, second])


def test_benchmark_exports_exact_scores_and_plots_without_refitting(temp_artifact_dir):
    from src.models.benchmark import run_model_benchmark
    matrix = pl.DataFrame({'DomainName': ['D'] * 24, 'UserName': [f'u{i}' for i in range(6)] * 4,
                          'day': np.repeat([1, 2, 3, 4], 6), 'f1': np.linspace(.01, .98, 24)})
    matrix_path = temp_artifact_dir / 'matrix.parquet'
    matrix.write_parquet(matrix_path)
    labels_path = temp_artifact_dir / 'labels.parquet'
    matrix.filter(pl.col('day') == 3).head(2).select(['DomainName', 'UserName', 'day']).with_columns(
        pl.lit(1).alias('is_anomaly')).write_parquet(labels_path)
    root = temp_artifact_dir / 'results'
    result = run_model_benchmark(data_path=matrix_path, model_names=['isolation_forest'],
        output_results_dir=root, save_models=False, split_day=2, feature_names=['f1'],
        stability_seeds=[], experiment_log_path=None, system_config_path=None,
        use_segments=False, labels_path=labels_path, k=2)
    original = result['models']['all']['isolation_forest']['scores']
    np.testing.assert_array_equal(result['scores']['isolation_forest_score'].to_numpy(), original)
    tables = compute_plot_tables([load_plot_run(root)], [1, 2], [1, 2])
    assert tables['metrics'].iloc[0].ap == pytest.approx(result['summary'].iloc[0].pr_auc)


def test_cli_main_stage_and_scenario_figures(temp_artifact_dir):
    import subprocess
    import sys
    import yaml
    root = temp_artifact_dir / 'results'
    frame = make_run(root)
    frame = frame.with_columns(pl.when(pl.col('label') == 1).then(pl.lit('brute_force')).otherwise(None).alias('scenario'))
    frame.write_parquet(root / 'anomaly_scores.parquet')
    cfg = temp_artifact_dir / 'system.yaml'
    cfg.write_text(yaml.safe_dump({'paths': {'results_dir': str(root)},
                                 'evaluation': {'precision_ks': [1, 2], 'daily_budgets': [1, 2]}}))
    repo = Path(__file__).resolve().parents[1]
    output = temp_artifact_dir / 'main_plots'
    completed = subprocess.run([sys.executable, 'main.py', '--stage', 'plots', '--config', str(cfg),
                                '--plots-output-dir', str(output), '--plot-formats', 'svg'],
                               cwd=repo, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert len(list(output.glob('*.svg'))) == 12
    assert (output / 'User_scenario_recall.svg').is_file()
    assert (output / 'daily_recall_summary.csv').is_file()
    # Standalone CLI validates cardinality instead of quietly reusing the wrong label file.
    completed = subprocess.run([sys.executable, 'scripts/evaluation/plot_benchmark_results.py',
                                '--results-dir', str(root), '--labels', 'a.parquet', 'b.parquet'],
                               cwd=repo, capture_output=True, text=True)
    assert completed.returncode == 2
    assert 'one label path' in completed.stderr


def test_identical_deterministic_scores_are_valid_across_distinct_seeds(temp_artifact_dir, monkeypatch):
    from src.evaluation import plots
    first, second = temp_artifact_dir / 'a', temp_artifact_dir / 'b'
    make_run(first)
    make_run(second)
    path = second / 'run_manifest.json'
    metadata = json.loads(path.read_text())
    metadata['config']['seed'] = 7
    path.write_text(json.dumps(metadata))
    monkeypatch.setattr(plots, '_render', lambda *args: [])
    result = plot_benchmark_results([first, second], output_dir=temp_artifact_dir / 'plots')
    assert result['tables']['metrics_summary'].ap_std.tolist() == [0, 0]
    assert result['manifest']['n_runs'] == 2
