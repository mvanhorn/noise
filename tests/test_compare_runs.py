"""CPU tests for utils/compare_runs.py. No torch, no GPU."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.compare_runs import (
    comparability_notes,
    compare_runs,
    config_diff,
    default_name,
    flatten,
    format_comparison,
    last_common_step,
    load_run,
    main,
    rank_runs,
    recipe_draws,
    snapshot_at,
)


def write_run(path, *, config=None, metrics=None, result=None):
    path.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (path / 'config.json').write_text(json.dumps(config) + '\n')
    if metrics is not None:
        (path / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metrics))
    if result is not None:
        (path / 'result.json').write_text(json.dumps(result) + '\n')
    return path


def recipe(scale=1.0):
    def count(k):
        return max(64, round(k * scale / 64) * 64)

    return dict(
        writers=[count(256)] * 8,
        hidden=[count(512)] * 8,
        hubs=[count(512)] * 8,
        embedding=count(2048),
        local=dict(q=count(384), v=count(384), gate=count(128), k=count(1664), ve=count(4608)),
        nrep=16,
        lr=0.003,
    )


def config(script='train/val/run.py', steps=4, batch=8, seed=0, scale=1.0, **extra):
    cfg = dict(
        script=f'/abs/{script}',
        world=8,
        batch=batch,
        steps=steps,
        seed=seed,
        recipe=recipe(scale),
        torch_version='2.10.0+cu128',
    )
    cfg.update(extra)
    return cfg


def metrics(n=4, loss_key='train_loss', start=8.0, end=4.0, dt=7.0, val=None):
    rows = []
    for step in range(1, n + 1):
        frac = 0 if n == 1 else (step - 1) / (n - 1)
        row = dict(step=step, train_seconds=round(step * dt, 2))
        row[loss_key] = start + (end - start) * frac
        if val is not None and step == n:
            row['val_loss'] = val
        rows.append(row)
    return rows


def result_of(rows, batch=8, val_loss=3.99, **extra):
    last = rows[-1]
    return dict(
        val_loss=val_loss,
        train_seconds=last['train_seconds'],
        steps=last['step'],
        tokens=last['step'] * batch * 2048,
        **extra,
    )


def pair(tmp_path, **b_over):
    a_metrics = metrics(4, end=4.0, dt=7.0)
    b_kwargs = dict(end=4.5, dt=2.0)
    b_kwargs.update({k: b_over.pop(k) for k in list(b_over) if k in ('end', 'dt', 'n')})
    n = b_kwargs.pop('n', 4)
    b_metrics = metrics(n, **b_kwargs)
    a = write_run(
        tmp_path / 'a',
        config=config(script='train/val/run.py', steps=4, scale=1.0),
        metrics=a_metrics,
        result=result_of(a_metrics, val_loss=3.99),
    )
    b_cfg = config(script='train/speedrun/run.py', steps=n, scale=0.3125, target=4.2)
    b = write_run(
        tmp_path / 'b',
        config=b_cfg,
        metrics=b_metrics,
        result=result_of(b_metrics, val_loss=b_over.get('val_loss', 4.18)),
    )
    return a, b


def test_flatten_keeps_scalar_lists_and_nests_dicts():
    flat = flatten({'recipe': {'writers': [256, 256], 'local': {'q': 384}}, 'seed': 0})
    assert flat['recipe.writers'] == [256, 256]
    assert flat['recipe.local.q'] == 384
    assert flat['seed'] == 0


def test_flatten_indexes_nested_lists():
    flat = flatten({'groups': [{'lr': 1}, {'lr': 2}]})
    assert flat['groups[0].lr'] == 1
    assert flat['groups[1].lr'] == 2


def test_config_diff_normalizes_script_and_finds_recipe_gap():
    left = config(script='train/val/run.py', scale=1.0)
    right = config(script='train/val/run.py', scale=0.3125)
    # scripts have different absolute prefixes but the same train/... tail
    left['script'] = '/other/host/train/val/run.py'
    diffs = {d['path']: d['values'] for d in config_diff([left, right])}
    assert 'script' not in diffs
    assert diffs['recipe.embedding'] == [2048, 640]
    assert 'recipe.writers' in diffs


def test_config_diff_identical_is_empty():
    cfg = config()
    assert config_diff([cfg, dict(cfg)]) == []


def test_config_diff_marks_missing_key():
    left = {'seed': 0, 'batch': 8}
    right = {'seed': 0}
    diffs = config_diff([left, right])
    assert any(d['path'] == 'batch' and d['values'] == [8, None] for d in diffs)


def test_default_name():
    assert default_name(Path('logs/val')) == 'val'
    assert default_name(Path('train/val/log')) == 'val/log'


def test_recipe_draws_speedrun_scale():
    assert recipe_draws(recipe(1.0)) == 19456
    assert recipe_draws(recipe(0.3125)) == 5440


def test_load_run_and_last_common_step(tmp_path):
    a, b = pair(tmp_path)
    ra, rb = load_run(a), load_run(b)
    assert ra.config['batch'] == 8
    assert last_common_step([ra, rb]) == 4
    write_run(tmp_path / 'short', config=config(steps=2), metrics=metrics(2))
    assert last_common_step([ra, load_run(tmp_path / 'short')]) == 2


def test_last_common_step_none_when_empty(tmp_path):
    write_run(tmp_path / 'empty', config=config(), metrics=[])
    (tmp_path / 'empty' / 'metrics.jsonl').write_text('')
    assert last_common_step([load_run(tmp_path / 'empty'), load_run(tmp_path / 'empty')]) is None


def test_snapshot_deltas_vs_baseline(tmp_path):
    a, b = pair(tmp_path)
    snap = snapshot_at([load_run(a, 'a'), load_run(b, 'b')], 4)
    assert snap['step'] == 4
    assert snap['rows'][0]['train_loss'] == pytest.approx(4.0)
    assert snap['rows'][1]['train_loss'] == pytest.approx(4.5)
    delta = snap['deltas_vs_baseline'][0]
    assert delta['train_loss'] == pytest.approx(0.5)
    assert delta['train_seconds'] == pytest.approx(8.0 - 28.0)
    assert delta['time_ratio'] == pytest.approx(28.0 / 8.0)


def test_snapshot_missing_step(tmp_path):
    a, b = pair(tmp_path)
    snap = snapshot_at([load_run(a, 'a'), load_run(b, 'b')], 99)
    assert snap['rows'][0]['present'] is False
    assert snap['deltas_vs_baseline'][0]['train_loss'] is None


def test_rank_runs_prefers_val_loss_then_time():
    ranking = rank_runs([
        {'name': 'slow', 'val_loss': 3.99, 'last_loss': 4.0, 'train_seconds': 4000},
        {'name': 'fast', 'val_loss': 4.18, 'last_loss': 4.1, 'train_seconds': 1000},
        {'name': 'tie', 'val_loss': 3.99, 'last_loss': 4.0, 'train_seconds': 3000},
    ])
    assert [r['name'] for r in ranking] == ['tie', 'slow', 'fast']
    assert ranking[0]['key'] == 'val_loss'


def test_rank_runs_falls_back_to_last_loss():
    ranking = rank_runs([
        {'name': 'a', 'val_loss': None, 'last_loss': 5.0, 'train_seconds': 10},
        {'name': 'b', 'val_loss': None, 'last_loss': 4.0, 'train_seconds': 20},
    ])
    assert [r['name'] for r in ranking] == ['b', 'a']
    assert ranking[0]['key'] == 'last_loss'


def test_compare_runs_table(tmp_path):
    a, b = pair(tmp_path)
    out = compare_runs([a, b], names=['baseline', 'candidate'])
    assert out['baseline'] == 'baseline'
    assert out['snapshot']['step'] == 4
    assert out['snapshot_source'] == 'last_common'
    paths = {d['path'] for d in out['config_diff']}
    assert 'recipe.embedding' in paths
    assert 'target' in paths
    assert out['ranking'][0]['name'] == 'baseline'
    assert out['runs'][0]['recipe_draws'] == 19456
    assert out['runs'][1]['recipe_draws'] == 5440


def test_compare_runs_at_step(tmp_path):
    a, b = pair(tmp_path)
    out = compare_runs([a, b], at_step=2)
    assert out['snapshot_source'] == 'at_step'
    assert out['snapshot']['step'] == 2
    assert all(row['present'] for row in out['snapshot']['rows'])


def test_compare_three_runs(tmp_path):
    a, b = pair(tmp_path)
    extra = metrics(4, end=3.5, dt=10.0)
    c = write_run(
        tmp_path / 'c',
        config=config(script='train/val/run.py', seed=1),
        metrics=extra,
        result=result_of(extra, val_loss=3.80),
    )
    out = compare_runs([a, b, c])
    assert len(out['runs']) == 3
    assert out['ranking'][0]['name'] == 'c'
    notes = ' '.join(out['comparability'])
    assert 'seed differs' in notes


def test_comparability_world_and_missing_result(tmp_path):
    write_run(tmp_path / 'a', config=config(world=8), metrics=metrics(2))
    write_run(tmp_path / 'b', config=config(world=1), metrics=metrics(2))
    notes = comparability_notes([load_run(tmp_path / 'a'), load_run(tmp_path / 'b')])
    joined = ' '.join(notes)
    assert 'world differs' in joined
    assert 'wall-clock' in joined
    assert 'result.json' in joined


def test_names_must_match_count(tmp_path):
    a, b = pair(tmp_path)
    with pytest.raises(ValueError):
        compare_runs([a, b], names=['only-one'])


def test_duplicate_default_names_get_suffix(tmp_path):
    a, b = pair(tmp_path)
    # both named via --names collision
    out = compare_runs([a, b], names=['run', 'run'])
    assert out['runs'][0]['name'] == 'run'
    assert out['runs'][1]['name'] == 'run#2'


def test_format_text_json_markdown(tmp_path):
    a, b = pair(tmp_path)
    comparison = compare_runs([a, b], names=['val', 'speed'])
    text = format_comparison(comparison, 'text')
    assert 'baseline: val' in text
    assert 'recipe.embedding' in text
    assert 'Ranking' in text
    md = format_comparison(comparison, 'markdown')
    assert md.startswith('# Compare runs')
    assert '| name |' in md
    payload = json.loads(format_comparison(comparison, 'json'))
    assert payload['baseline'] == 'val'
    with pytest.raises(ValueError):
        format_comparison(comparison, 'html')


def test_cli_success(tmp_path, capsys):
    a, b = pair(tmp_path)
    assert main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert 'Comparing 2 runs' in out
    assert 'recipe.embedding' in out


def test_cli_json_and_names_and_at_step(tmp_path, capsys):
    a, b = pair(tmp_path)
    assert main(['--format', 'json', '--names', 'left', 'right', '--at-step', '1', str(a), str(b)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['baseline'] == 'left'
    assert payload['snapshot']['step'] == 1


def test_cli_requires_two_dirs(tmp_path):
    with pytest.raises(SystemExit):
        main([str(tmp_path)])


def test_cli_names_mismatch(tmp_path):
    a, b = pair(tmp_path)
    with pytest.raises(SystemExit):
        main(['--names', 'only', str(a), str(b)])


def test_cli_missing_and_not_dir(tmp_path, capsys):
    a, b = pair(tmp_path)
    assert main([str(a), str(tmp_path / 'gone')]) == 2
    file_path = tmp_path / 'file.txt'
    file_path.write_text('x')
    assert main([str(a), str(file_path)]) == 2
    err = capsys.readouterr().err
    assert 'does not exist' in err
    assert 'not a directory' in err


def test_cli_broken_run_exits_1(tmp_path):
    a, _ = pair(tmp_path)
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert main([str(a), str(empty)]) == 1


def test_cli_at_step_missing(tmp_path, capsys):
    a, b = pair(tmp_path)
    assert main(['--at-step', '99', str(a), str(b)]) == 1
    assert 'no run has step 99' in capsys.readouterr().err


def test_cli_rejects_bad_at_step(tmp_path):
    a, b = pair(tmp_path)
    with pytest.raises(SystemExit):
        main(['--at-step', '0', str(a), str(b)])


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0


def test_compare_checked_in_records():
    comparison = compare_runs([
        ROOT / 'train/val/log',
        ROOT / 'train/speedrun/log',
    ])
    assert comparison['snapshot']['step'] == 610
    assert comparison['runs'][0]['val_loss'] == pytest.approx(3.991304921753266)
    assert comparison['runs'][1]['val_loss'] == pytest.approx(4.180039276094997)
    assert comparison['ranking'][0]['name'] == 'val/log'
    paths = {d['path'] for d in comparison['config_diff']}
    assert 'recipe.embedding' in paths
    assert 'target' in paths
    assert comparison['runs'][1]['reached_target'] is True
    delta = comparison['snapshot']['deltas_vs_baseline'][0]
    assert delta['train_seconds'] < 0
    assert delta['time_ratio'] > 1


def test_cli_on_checked_in_records(capsys):
    assert main([str(ROOT / 'train/val/log'), str(ROOT / 'train/speedrun/log')]) == 0
    out = capsys.readouterr().out
    assert 'val/log' in out
    assert 'speedrun/log' in out
    assert '3.3x faster' in out or 'faster' in out
