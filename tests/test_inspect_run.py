"""CPU tests for utils/inspect_run.py. No torch, no GPU."""
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.inspect_run import (
    Issue,
    default_name,
    format_duration,
    format_report,
    inspect_path,
    is_finite,
    load_run,
    main,
    recipe_draws,
    row_loss,
    sparkline,
    step_seconds,
    summarize_run,
    validate_run,
)


def write_run(path, *, config=None, metrics=None, result=None, skip=()):
    path.mkdir(parents=True, exist_ok=True)
    if config is not None and 'config.json' not in skip:
        (path / 'config.json').write_text(json.dumps(config) + '\n')
    if metrics is not None and 'metrics.jsonl' not in skip:
        (path / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in metrics))
    if result is not None and 'result.json' not in skip:
        (path / 'result.json').write_text(json.dumps(result) + '\n')
    return path


def sample_config(**overrides):
    cfg = dict(
        script='/home/ubuntu/noise/train/val/run.py',
        world=8,
        batch=8,
        steps=4,
        seed=0,
        recipe=dict(
            writers=[256] * 8,
            hidden=[512] * 8,
            hubs=[512] * 8,
            embedding=2048,
            local=dict(q=384, v=384, gate=128, k=1664, ve=4608),
            nrep=16,
            lr=0.003,
        ),
        torch_version='2.10.0+cu128',
    )
    cfg.update(overrides)
    return cfg


def sample_metrics(n=4, loss_key='train_loss', start_loss=8.0, end_loss=4.0, dt=7.0, val_every=0):
    rows = []
    for step in range(1, n + 1):
        frac = 0 if n == 1 else (step - 1) / (n - 1)
        row = dict(step=step, train_seconds=round(step * dt, 2))
        row[loss_key] = start_loss + (end_loss - start_loss) * frac
        if val_every and step % val_every == 0:
            row['val_loss'] = row[loss_key] + 0.1
        rows.append(row)
    return rows


def sample_result(metrics, batch=8, **extra):
    last = metrics[-1]
    return dict(
        val_loss=3.99,
        train_seconds=last['train_seconds'],
        steps=last['step'],
        tokens=last['step'] * batch * 2048,
        **extra,
    )


def complete_run(path, n=4):
    metrics = sample_metrics(n)
    return write_run(
        path,
        config=sample_config(steps=n),
        metrics=metrics,
        result=sample_result(metrics),
    )


def issue_codes(issues, level=None):
    return [i.code for i in issues if level is None or i.level == level]


def test_is_finite():
    assert is_finite(0)
    assert is_finite(1.5)
    assert not is_finite(None)
    assert not is_finite(float('nan'))
    assert not is_finite(float('inf'))
    assert not is_finite('1.0')


def test_row_loss_prefers_train_then_population():
    assert row_loss({'train_loss': 1.2}) == 1.2
    assert row_loss({'population_loss': 2.3}) == 2.3
    assert row_loss({'train_loss': 1.2, 'population_loss': 9}) == 1.2
    assert row_loss({'step': 1}) is None


def test_recipe_draws_matches_16k_cell():
    draws = recipe_draws(sample_config()['recipe'])
    assert draws == 8 * 256 + 8 * 512 + 8 * 512 + 2048 + 384 + 384 + 128 + 1664 + 4608
    assert draws == 19456
    assert recipe_draws(None) is None
    assert recipe_draws({}) is None
    assert recipe_draws({'embedding': 64}) == 64


def test_default_name_disambiguates_log_dirs():
    assert default_name(Path('logs/val')) == 'val'
    assert default_name(Path('train/val/log')) == 'val/log'
    assert default_name(Path('train/speedrun/log')) == 'speedrun/log'


def test_sparkline_empty_constant_and_width():
    assert sparkline([]) == ''
    assert sparkline([3, 3, 3]) == '▁' * 3
    rising = sparkline([1, 2, 3, 4, 5, 6, 7, 8], width=8)
    assert rising[0] == '▁'
    assert rising[-1] == '█'
    assert len(sparkline(range(100), width=10)) == 10
    assert sparkline([1, float('nan'), 2], width=2)


def test_format_duration():
    assert format_duration(12.3) == '12.30 s'
    assert 'min' in format_duration(125)
    assert 'h' in format_duration(4303.45)
    assert format_duration(None) is None


def test_load_run_reads_all_files(tmp_path):
    complete_run(tmp_path / 'r', n=3)
    run = load_run(tmp_path / 'r')
    assert run.config['batch'] == 8
    assert len(run.metrics) == 3
    assert run.result['val_loss'] == 3.99
    assert run.missing == []
    assert run.parse_errors == []


def test_load_run_skips_blank_metrics_lines(tmp_path):
    path = tmp_path / 'r'
    path.mkdir()
    (path / 'config.json').write_text(json.dumps(sample_config()) + '\n')
    (path / 'metrics.jsonl').write_text(
        json.dumps(sample_metrics(1)[0]) + '\n\n' + json.dumps(sample_metrics(2)[1]) + '\n'
    )
    run = load_run(path)
    assert [row['step'] for row in run.metrics] == [1, 2]


def test_load_run_records_bad_jsonl_and_missing_result(tmp_path):
    path = tmp_path / 'r'
    write_run(path, config=sample_config(), metrics=sample_metrics(2))
    with (path / 'metrics.jsonl').open('a') as fh:
        fh.write('not-json\n')
    run = load_run(path)
    assert 'result.json' in run.missing
    assert any('metrics.jsonl:3' in err for err in run.parse_errors)
    assert len(run.metrics) == 2


def test_load_run_missing_directory(tmp_path):
    run = load_run(tmp_path / 'gone')
    assert run.parse_errors
    assert run.metrics == []


def test_validate_clean_complete_run(tmp_path):
    complete_run(tmp_path / 'r')
    issues = validate_run(load_run(tmp_path / 'r'))
    assert issues == []


def test_validate_empty_metrics(tmp_path):
    path = tmp_path / 'r'
    write_run(path, config=sample_config(), metrics=[])
    (path / 'metrics.jsonl').write_text('')
    issues = validate_run(load_run(path))
    assert 'empty_metrics' in issue_codes(issues, 'error')


def test_validate_missing_required_files(tmp_path):
    path = tmp_path / 'r'
    path.mkdir()
    issues = validate_run(load_run(path))
    assert issue_codes(issues, 'error').count('missing') == 2
    assert 'missing' in issue_codes(issues, 'warning')


def test_validate_step_gaps_and_order(tmp_path):
    metrics = sample_metrics(3)
    metrics[1]['step'] = 3
    write_run(tmp_path / 'gap', config=sample_config(steps=3), metrics=metrics)
    codes = issue_codes(validate_run(load_run(tmp_path / 'gap')))
    assert 'step' in codes

    metrics = sample_metrics(3)
    metrics[0]['step'] = 2
    write_run(tmp_path / 'start', config=sample_config(steps=3), metrics=metrics)
    warnings = issue_codes(validate_run(load_run(tmp_path / 'start')), 'warning')
    assert 'step' in warnings


def test_validate_duplicate_step(tmp_path):
    metrics = sample_metrics(3)
    metrics[2]['step'] = 2
    write_run(tmp_path / 'dup', config=sample_config(), metrics=metrics)
    assert 'step' in issue_codes(validate_run(load_run(tmp_path / 'dup')), 'error')


def test_validate_nonfinite_and_decreasing_time(tmp_path):
    metrics = sample_metrics(3)
    metrics[1]['train_loss'] = float('nan')
    write_run(tmp_path / 'nan', config=sample_config(), metrics=metrics)
    assert 'loss' in issue_codes(validate_run(load_run(tmp_path / 'nan')), 'error')

    metrics = sample_metrics(3)
    metrics[2]['train_seconds'] = 1.0
    write_run(tmp_path / 'time', config=sample_config(), metrics=metrics)
    assert 'time' in issue_codes(validate_run(load_run(tmp_path / 'time')), 'error')


def test_validate_result_mismatch_and_tokens(tmp_path):
    metrics = sample_metrics(3)
    result = sample_result(metrics)
    result['steps'] = 99
    result['train_seconds'] = 0.01
    result['tokens'] = 1
    write_run(tmp_path / 'r', config=sample_config(steps=3), metrics=metrics, result=result)
    issues = validate_run(load_run(tmp_path / 'r'))
    assert issue_codes(issues, 'error').count('result') >= 2
    assert 'tokens' in issue_codes(issues, 'warning')


def test_validate_epoch_budget_and_seed(tmp_path):
    write_run(
        tmp_path / 'r',
        config=sample_config(steps=700, batch=8, seed=7),
        metrics=sample_metrics(2),
    )
    issues = validate_run(load_run(tmp_path / 'r'))
    assert 'epoch' in issue_codes(issues, 'warning')
    assert 'seed' in issue_codes(issues, 'warning')


def test_validate_nonfinite_val_loss(tmp_path):
    metrics = sample_metrics(2)
    result = sample_result(metrics)
    result['val_loss'] = float('inf')
    write_run(tmp_path / 'r', config=sample_config(steps=2), metrics=metrics, result=result)
    assert 'result' in issue_codes(validate_run(load_run(tmp_path / 'r')), 'error')


def test_summarize_complete_run(tmp_path):
    complete_run(tmp_path / 'r', n=4)
    summary = summarize_run(load_run(tmp_path / 'r'))
    assert summary['status'] == 'complete'
    assert summary['script'] == 'train/val/run.py'
    assert summary['logged_steps'] == 4
    assert summary['planned_steps'] == 4
    assert summary['progress'] == 1.0
    assert summary['first_loss'] == pytest.approx(8.0)
    assert summary['last_loss'] == pytest.approx(4.0)
    assert summary['min_loss'] == pytest.approx(4.0)
    assert summary['min_loss_step'] == 4
    assert summary['val_loss'] == pytest.approx(3.99)
    assert summary['tokens'] == 4 * 8 * 2048
    assert summary['recipe_draws'] == 19456
    assert summary['tokens_per_second'] == pytest.approx(summary['tokens'] / summary['train_seconds'])
    assert summary['sparkline']
    assert summary['eta_seconds'] is None


def test_summarize_in_progress_eta(tmp_path):
    write_run(
        tmp_path / 'r',
        config=sample_config(steps=10),
        metrics=sample_metrics(4, dt=5.0),
    )
    summary = summarize_run(load_run(tmp_path / 'r'))
    assert summary['status'] == 'in_progress'
    assert summary['remaining_steps'] == 6
    assert summary['seconds_per_step'] == pytest.approx(5.0)
    assert summary['eta_seconds'] == pytest.approx(30.0)


def test_summarize_stopped_early_and_population_loss(tmp_path):
    metrics = sample_metrics(3, loss_key='population_loss')
    write_run(
        tmp_path / 'r',
        config=sample_config(steps=10, script='train/val/eggroll.py'),
        metrics=metrics,
        result=sample_result(metrics),
    )
    summary = summarize_run(load_run(tmp_path / 'r'))
    assert summary['status'] == 'stopped_early'
    assert summary['loss_key'] == 'population_loss'
    assert summary['last_loss'] == pytest.approx(4.0)


def test_summarize_val_loss_from_metrics_when_no_result(tmp_path):
    write_run(
        tmp_path / 'r',
        config=sample_config(steps=6),
        metrics=sample_metrics(4, val_every=2),
    )
    summary = summarize_run(load_run(tmp_path / 'r'))
    assert summary['val_loss'] == pytest.approx(4.0 + 0.1)


def test_summarize_missing_result_after_planned_steps(tmp_path):
    write_run(tmp_path / 'r', config=sample_config(steps=3), metrics=sample_metrics(3))
    assert summarize_run(load_run(tmp_path / 'r'))['status'] == 'missing_result'


def test_step_seconds_median_and_short_series():
    assert step_seconds([]) is None
    assert step_seconds([{'step': 1, 'train_seconds': 1}]) is None
    rows = [
        {'step': 1, 'train_seconds': 0},
        {'step': 2, 'train_seconds': 10},
        {'step': 3, 'train_seconds': 12},
        {'step': 4, 'train_seconds': 22},
    ]
    assert step_seconds(rows, window=10) == 10


def test_format_text_and_markdown_and_json(tmp_path):
    complete_run(tmp_path / 'r')
    run = load_run(tmp_path / 'r')
    summary = summarize_run(run)
    text = format_report(summary, [], 'text')
    assert 'Status: complete' in text
    assert 'train_loss:' in text
    assert 'Issues: none' in text
    md = format_report(summary, [Issue('warning', 'seed', 'seed is 1')], 'markdown')
    assert md.startswith('# Run')
    assert '**warning**' in md
    payload = json.loads(format_report(summary, [], 'json'))
    assert payload['status'] == 'complete'
    assert payload['issues'] == []
    with pytest.raises(ValueError):
        format_report(summary, [], 'html')


def test_inspect_path_roundtrip(tmp_path):
    complete_run(tmp_path / 'val/log')
    report, issues = inspect_path(tmp_path / 'val/log', fmt='text')
    assert issues == []
    assert 'val/log' in report
    assert '19,456' in report


def test_cli_success(tmp_path, capsys):
    complete_run(tmp_path / 'r')
    assert main([str(tmp_path / 'r')]) == 0
    out = capsys.readouterr().out
    assert 'Status: complete' in out


def test_cli_json_and_strict_warning(tmp_path, capsys):
    write_run(
        tmp_path / 'r',
        config=sample_config(seed=3, steps=2),
        metrics=sample_metrics(2),
        result=sample_result(sample_metrics(2)),
    )
    assert main(['--format', 'json', str(tmp_path / 'r')]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload['seed'] == 3
    assert any(i['code'] == 'seed' for i in payload['issues'])
    assert main(['--strict', str(tmp_path / 'r')]) == 1


def test_cli_errors_on_broken_run_and_missing_path(tmp_path, capsys):
    (tmp_path / 'r').mkdir()
    assert main([str(tmp_path / 'r')]) == 1
    assert main([str(tmp_path / 'missing')]) == 2
    err = capsys.readouterr().err
    assert 'does not exist' in err
    file_path = tmp_path / 'file.txt'
    file_path.write_text('x')
    assert main([str(file_path)]) == 2


def test_cli_rejects_bad_spark_width(tmp_path):
    complete_run(tmp_path / 'r')
    with pytest.raises(SystemExit):
        main(['--spark-width', '0', str(tmp_path / 'r')])


def test_cli_help():
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0


def test_checked_in_val_log():
    report, issues = inspect_path(ROOT / 'train/val/log', fmt='json')
    assert [i.level for i in issues if i.level == 'error'] == []
    payload = json.loads(report)
    assert payload['status'] == 'complete'
    assert payload['logged_steps'] == 610
    assert payload['val_loss'] == pytest.approx(3.991304921753266)
    assert payload['tokens'] == 9_994_240
    assert payload['recipe_draws'] == 19456


def test_checked_in_speedrun_log():
    report, issues = inspect_path(ROOT / 'train/speedrun/log', fmt='text')
    assert issues == []
    assert 'reached target' in report
    assert '4.1800' in report
    assert '5,440' in report
