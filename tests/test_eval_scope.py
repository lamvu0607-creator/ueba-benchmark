"""Regression checks for dev/test scope, including days with no injections."""
import json

import polars as pl
import pytest
import yaml

from src.evaluation.split import restrict_eval_days, time_split
from src.injection.layout import InjectionLayoutError, RunLayout, load_run_eval_days


def test_scope_preserves_training_and_rejects_missing_days():
    matrix = pl.DataFrame({'day': [1, 2, 3, 4, 5]})
    train, evaluation, info = time_split(matrix, split_day=2)
    scoped, updated = restrict_eval_days(evaluation, info, [4, 5])
    assert train['day'].to_list() == [1, 2]
    assert scoped['day'].to_list() == [4, 5]
    assert updated.n_train == info.n_train and updated.n_eval == 2
    for days in ([], [2], [6]):
        with pytest.raises(ValueError):
            restrict_eval_days(evaluation, info, days)


def test_older_run_uses_full_config_block_not_only_injected_days(temp_artifact_dir):
    layout = RunLayout(temp_artifact_dir)
    pl.DataFrame({'split': ['dev'], 'day': [44]}).write_csv(layout.manifest_path)
    cfg = {'common': {'split_day': 42}, 'blocks': {'dev': {'days': list(range(43, 52))}}}
    path = temp_artifact_dir / 'injection.yaml'
    path.write_text(yaml.safe_dump(cfg))
    assert load_run_eval_days(layout, path) == list(range(43, 52))
    # Frozen metadata takes precedence even when configuration is changed later.
    (layout.root / 'run_config.json').write_text(json.dumps({'eval_days': [43, 44], 'split_day': 42}))
    assert load_run_eval_days(layout, path) == [43, 44]
    (layout.root / 'run_config.json').write_text(json.dumps({'eval_days': [43], 'split_day': 42}))
    with pytest.raises(InjectionLayoutError, match='outside'):
        load_run_eval_days(layout, path)
