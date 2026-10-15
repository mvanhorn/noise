"""Inspect a training run directory written by a Noise training script.

    uv run utils/inspect_run.py logs/val
    uv run utils/inspect_run.py logs/val --format json
    uv run utils/inspect_run.py logs/val --format markdown --strict

Reads ``config.json``, ``metrics.jsonl`` and ``result.json`` (when present).
Prints progress, throughput, an ETA for unfinished runs, a loss sparkline,
and integrity checks. Does not load weights or run the model.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys

# Same one-epoch budget as utils/prepare_data.py (avoid importing torch).
TRAIN_ROWS = 4880
PRED_TOKENS = 2048
LOSS_KEYS = ('train_loss', 'population_loss')
BLOCKS = '▁▂▃▄▅▆▇█'


@dataclass
class Issue:
    """One integrity finding. ``error`` fails the default exit; ``warning`` fails only with --strict."""

    level: str
    code: str
    message: str


@dataclass
class RunLog:
    path: Path
    config: dict | None = None
    metrics: list[dict] = field(default_factory=list)
    result: dict | None = None
    missing: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


def is_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def row_loss(row):
    """Training or population loss on a metrics line, whichever the script wrote."""
    for key in LOSS_KEYS:
        if key in row:
            return row[key]
    return None


def loss_key(rows):
    for row in rows:
        for key in LOSS_KEYS:
            if key in row:
                return key
    return None


def short_script(script):
    if not script:
        return None
    parts = Path(str(script)).parts
    if 'train' in parts:
        return '/'.join(parts[parts.index('train'):])
    return Path(str(script)).name


def default_name(path):
    path = Path(path)
    if path.name in {'log', 'logs'}:
        return f'{path.parent.name}/{path.name}'
    return path.name


def recipe_draws(recipe):
    """Draws (or antithetic pairs are not counted) summed over sites for one update."""
    if not isinstance(recipe, dict):
        return None
    total = 0
    found = False
    for key in ('writers', 'hidden', 'hubs'):
        value = recipe.get(key)
        if isinstance(value, list) and value and all(isinstance(x, (int, float)) for x in value):
            total += sum(value)
            found = True
        elif isinstance(value, (int, float)):
            total += value
            found = True
    if isinstance(recipe.get('embedding'), (int, float)):
        total += recipe['embedding']
        found = True
    local = recipe.get('local')
    if isinstance(local, dict):
        total += sum(v for v in local.values() if isinstance(v, (int, float)))
        found = True
    return int(total) if found else None


def load_json_file(path):
    text = path.read_text()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f'{path.name}: expected a JSON object')
    return data


def load_metrics(path):
    """Parse metrics.jsonl. Returns (rows, parse_errors). Bad lines are skipped."""
    rows, errors = [], []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f'{path.name}:{lineno}: {exc}')
            continue
        if not isinstance(row, dict):
            errors.append(f'{path.name}:{lineno}: expected a JSON object')
            continue
        rows.append(row)
    return rows, errors


def load_run(path):
    """Load a run directory. Missing files are listed on the result, not raised."""
    path = Path(path)
    run = RunLog(path=path)
    if not path.is_dir():
        run.parse_errors.append(f'not a directory: {path}')
        return run
    mapping = {
        'config.json': 'config',
        'result.json': 'result',
    }
    for name, attr in mapping.items():
        file = path / name
        if not file.is_file():
            run.missing.append(name)
            continue
        try:
            setattr(run, attr, load_json_file(file))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            run.parse_errors.append(f'{name}: {exc}')
    metrics_path = path / 'metrics.jsonl'
    if not metrics_path.is_file():
        run.missing.append('metrics.jsonl')
    else:
        run.metrics, metric_errors = load_metrics(metrics_path)
        run.parse_errors.extend(metric_errors)
    return run


def sparkline(values, width=48):
    """Unicode sparkline of finite numbers, downsampled to ``width`` buckets."""
    nums = [float(v) for v in values if is_finite(v)]
    if not nums or width <= 0:
        return ''
    if len(nums) > width:
        step = len(nums) / width
        bucketed = []
        for i in range(width):
            start = int(i * step)
            end = max(start + 1, int((i + 1) * step))
            chunk = nums[start:end]
            bucketed.append(sum(chunk) / len(chunk))
        nums = bucketed
    lo, hi = min(nums), max(nums)
    span = hi - lo
    last = len(BLOCKS) - 1
    if span == 0:
        return BLOCKS[0] * len(nums)
    chars = []
    for value in nums:
        idx = int((value - lo) / span * last)
        chars.append(BLOCKS[min(last, max(0, idx))])
    return ''.join(chars)


def median(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def step_seconds(metrics, window=20):
    """Median seconds per step over the last ``window`` intervals."""
    if len(metrics) < 2:
        return None
    recent = metrics[-window:]
    dts = []
    for prev, cur in zip(recent, recent[1:]):
        t0, t1 = prev.get('train_seconds'), cur.get('train_seconds')
        s0, s1 = prev.get('step'), cur.get('step')
        if not all(is_finite(v) for v in (t0, t1, s0, s1)):
            continue
        if s1 <= s0:
            continue
        dt = (t1 - t0) / (s1 - s0)
        if dt >= 0:
            dts.append(dt)
    return median(dts)


def format_duration(seconds):
    if not is_finite(seconds):
        return None
    seconds = float(seconds)
    if seconds < 60:
        return f'{seconds:.2f} s'
    if seconds < 3600:
        return f'{seconds:.2f} s ({seconds / 60:.2f} min)'
    return f'{seconds:.2f} s ({seconds / 3600:.2f} h)'


def validate_run(run):
    """Integrity checks against the files a training script writes to its output directory."""
    issues = []
    for message in run.parse_errors:
        issues.append(Issue('error', 'parse', message))
    for name in run.missing:
        level = 'warning' if name == 'result.json' else 'error'
        issues.append(Issue(level, 'missing', f'{name} is missing'))

    metrics = run.metrics
    if 'metrics.jsonl' not in run.missing and not metrics and not any(
        i.code == 'parse' for i in issues
    ):
        issues.append(Issue('error', 'empty_metrics', 'metrics.jsonl has no records'))

    steps = []
    for index, row in enumerate(metrics):
        step = row.get('step')
        if not isinstance(step, int) or isinstance(step, bool):
            issues.append(Issue('error', 'step', f'metrics[{index}] has no integer step'))
            continue
        steps.append(step)
        if row_loss(row) is None:
            issues.append(Issue('error', 'loss', f'step {step}: missing {LOSS_KEYS[0]} or {LOSS_KEYS[1]}'))
        elif not is_finite(row_loss(row)):
            issues.append(Issue('error', 'loss', f'step {step}: non-finite loss'))
        seconds = row.get('train_seconds')
        if not is_finite(seconds):
            issues.append(Issue('error', 'time', f'step {step}: missing or non-finite train_seconds'))
        elif seconds < 0:
            issues.append(Issue('error', 'time', f'step {step}: negative train_seconds'))

    if steps:
        if steps[0] != 1:
            issues.append(Issue('warning', 'step', f'metrics start at step {steps[0]}, expected 1'))
        expected = steps[0]
        for step in steps:
            if step < expected:
                issues.append(Issue('error', 'step', f'step {step} is out of order (expected {expected})'))
                break
            if step > expected:
                issues.append(Issue('error', 'step', f'missing steps {expected}..{step - 1}'))
                break
            expected = step + 1
        seen = {}
        for step in steps:
            if step in seen:
                issues.append(Issue('error', 'step', f'duplicate step {step}'))
                break
            seen[step] = True
        times = [row.get('train_seconds') for row in metrics if is_finite(row.get('train_seconds'))]
        for prev, cur in zip(times, times[1:]):
            if cur < prev:
                issues.append(Issue('error', 'time', 'train_seconds decreased between steps'))
                break

    config = run.config or {}
    planned = config.get('steps')
    batch = config.get('batch')
    last_step = steps[-1] if steps else None
    if isinstance(planned, int) and isinstance(batch, int) and planned * batch > TRAIN_ROWS:
        issues.append(Issue(
            'warning', 'epoch',
            f'config plans {planned} steps × batch {batch} = {planned * batch} sequences, '
            f'over the {TRAIN_ROWS}-sequence epoch',
        ))
    if isinstance(last_step, int) and isinstance(batch, int) and last_step * batch > TRAIN_ROWS:
        issues.append(Issue(
            'warning', 'epoch',
            f'logged {last_step} steps × batch {batch} = {last_step * batch} sequences, '
            f'over the {TRAIN_ROWS}-sequence epoch',
        ))
    seed = config.get('seed')
    if seed is not None and seed != 0:
        issues.append(Issue('warning', 'seed', f'config seed is {seed}, runs normally use seed 0'))

    result = run.result
    if result is not None and last_step is not None:
        if result.get('steps') != last_step:
            issues.append(Issue(
                'error', 'result',
                f'result.json steps={result.get("steps")!r} != last metrics step {last_step}',
            ))
        last_time = metrics[-1].get('train_seconds')
        result_time = result.get('train_seconds')
        if is_finite(last_time) and is_finite(result_time) and abs(float(result_time) - float(last_time)) > 0.011:
            issues.append(Issue(
                'error', 'result',
                f'result.json train_seconds={result_time} != last metrics train_seconds {last_time}',
            ))
        val_loss = result.get('val_loss')
        if val_loss is not None and not is_finite(val_loss):
            issues.append(Issue('error', 'result', 'result.json val_loss is not finite'))
        tokens = result.get('tokens')
        if is_finite(tokens) and isinstance(last_step, int) and isinstance(batch, int):
            expected_tokens = last_step * batch * PRED_TOKENS
            if int(tokens) != expected_tokens:
                issues.append(Issue(
                    'warning', 'tokens',
                    f'result.json tokens={int(tokens)} != steps×batch×{PRED_TOKENS}={expected_tokens}',
                ))
    return issues


def _status(run, last_step, planned):
    has_result = run.result is not None
    if has_result and planned is not None and last_step is not None and last_step >= planned:
        return 'complete'
    if has_result and planned is not None and last_step is not None and last_step < planned:
        return 'stopped_early'
    if has_result:
        return 'complete'
    if last_step is not None and planned is not None and last_step < planned:
        return 'in_progress'
    if last_step is not None and planned is not None and last_step >= planned and not has_result:
        return 'missing_result'
    if last_step is None:
        return 'empty'
    return 'in_progress'


def summarize_run(run, spark_width=48):
    """Numbers and labels for the report. Safe on incomplete or broken logs."""
    config = run.config or {}
    metrics = run.metrics
    result = run.result or {}
    key = loss_key(metrics)
    losses = [row_loss(row) for row in metrics]
    finite_losses = [(row.get('step'), row_loss(row)) for row in metrics if is_finite(row_loss(row))]
    steps = [row.get('step') for row in metrics if isinstance(row.get('step'), int) and not isinstance(row.get('step'), bool)]
    last_step = steps[-1] if steps else None
    planned = config.get('steps') if isinstance(config.get('steps'), int) else None
    batch = config.get('batch') if isinstance(config.get('batch'), int) else None
    last_time = None
    for row in reversed(metrics):
        if is_finite(row.get('train_seconds')):
            last_time = float(row['train_seconds'])
            break
    if last_time is None and is_finite(result.get('train_seconds')):
        last_time = float(result['train_seconds'])

    tokens = result.get('tokens') if is_finite(result.get('tokens')) else None
    if tokens is None and last_step is not None and batch is not None:
        tokens = last_step * batch * PRED_TOKENS

    per_step = step_seconds(metrics)
    remaining = None
    if planned is not None and last_step is not None and last_step < planned:
        remaining = planned - last_step
    eta = remaining * per_step if remaining is not None and per_step is not None else None

    min_pair = min(finite_losses, key=lambda p: p[1]) if finite_losses else (None, None)
    max_pair = max(finite_losses, key=lambda p: p[1]) if finite_losses else (None, None)
    first_loss = finite_losses[0][1] if finite_losses else None
    last_loss = finite_losses[-1][1] if finite_losses else None

    val_loss = result.get('val_loss') if is_finite(result.get('val_loss')) else None
    if val_loss is None:
        for row in reversed(metrics):
            if is_finite(row.get('val_loss')):
                val_loss = float(row['val_loss'])
                break

    draws = recipe_draws(config.get('recipe'))
    throughput = (tokens / last_time) if tokens and last_time and last_time > 0 else None
    progress = None
    if planned and last_step is not None:
        progress = min(1.0, last_step / planned)

    return {
        'path': str(run.path),
        'name': default_name(run.path),
        'status': _status(run, last_step, planned),
        'script': short_script(config.get('script')),
        'world': config.get('world'),
        'batch': batch,
        'seed': config.get('seed'),
        'target': config.get('target'),
        'planned_steps': planned,
        'logged_steps': last_step,
        'progress': progress,
        'loss_key': key,
        'first_loss': first_loss,
        'last_loss': last_loss,
        'min_loss': min_pair[1],
        'min_loss_step': min_pair[0],
        'max_loss': max_pair[1],
        'max_loss_step': max_pair[0],
        'val_loss': val_loss,
        'reached_target': result.get('reached_target') if 'reached_target' in result else None,
        'train_seconds': last_time,
        'seconds_per_step': per_step,
        'remaining_steps': remaining,
        'eta_seconds': eta,
        'tokens': int(tokens) if tokens is not None else None,
        'tokens_per_second': throughput,
        'recipe_draws': draws,
        'nrep': (config.get('recipe') or {}).get('nrep') if isinstance(config.get('recipe'), dict) else None,
        'torch_version': config.get('torch_version'),
        'sparkline': sparkline([loss for _, loss in finite_losses], spark_width),
        'missing': list(run.missing),
        'parse_errors': list(run.parse_errors),
    }


def _fmt(value, digits=4):
    if value is None:
        return '—'
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if isinstance(value, int) and not isinstance(value, bool):
        return f'{value:,}'
    if isinstance(value, float):
        if abs(value) >= 100:
            return f'{value:,.2f}'
        return f'{value:.{digits}f}'
    return str(value)


def format_text(summary, issues):
    lines = [
        f"Run: {summary['path']}",
        f"Name: {summary['name']}",
        f"Status: {summary['status']}",
    ]
    if summary['script']:
        lines.append(f"Script: {summary['script']}")
    hardware = []
    if summary['world'] is not None:
        hardware.append(f"world {summary['world']}")
    if summary['batch'] is not None:
        hardware.append(f"batch {summary['batch']}")
    if summary['seed'] is not None:
        hardware.append(f"seed {summary['seed']}")
    if hardware:
        lines.append('Setup: ' + ', '.join(hardware))
    if summary['target'] is not None:
        lines.append(f"Target: {summary['target']}")
    if summary['planned_steps'] is not None or summary['logged_steps'] is not None:
        planned = _fmt(summary['planned_steps'])
        logged = _fmt(summary['logged_steps'])
        pct = f" ({100 * summary['progress']:.1f}%)" if summary['progress'] is not None else ''
        lines.append(f'Steps: {logged} / {planned}{pct}')
    if summary['last_loss'] is not None:
        key = summary['loss_key'] or 'loss'
        trend = f"{_fmt(summary['first_loss'])} → {_fmt(summary['last_loss'])}"
        extra = ''
        if summary['min_loss'] is not None:
            extra = f"  min {_fmt(summary['min_loss'])} @ step {summary['min_loss_step']}"
        lines.append(f'{key}: {trend}{extra}')
    if summary['val_loss'] is not None:
        reached = ''
        if summary['reached_target'] is not None:
            reached = '  reached target' if summary['reached_target'] else '  missed target'
        lines.append(f"val_loss: {_fmt(summary['val_loss'])}{reached}")
    if summary['train_seconds'] is not None:
        bits = [format_duration(summary['train_seconds'])]
        if summary['seconds_per_step'] is not None:
            bits.append(f"{summary['seconds_per_step']:.2f} s/step")
        if summary['tokens_per_second'] is not None:
            bits.append(f"{summary['tokens_per_second']:,.0f} tok/s")
        lines.append('Time: ' + ', '.join(bits))
    if summary['tokens'] is not None:
        lines.append(f"Tokens: {_fmt(summary['tokens'])}")
    if summary['recipe_draws'] is not None:
        nrep = f", nrep {summary['nrep']}" if summary['nrep'] is not None else ''
        lines.append(f"Draws / update: {_fmt(summary['recipe_draws'])}{nrep}")
    if summary['eta_seconds'] is not None:
        lines.append(
            f"ETA: {format_duration(summary['eta_seconds'])} "
            f"({_fmt(summary['remaining_steps'])} steps left)"
        )
    elif summary['remaining_steps']:
        lines.append(f"Remaining: {_fmt(summary['remaining_steps'])} steps (not enough timing to estimate ETA)")
    if summary['sparkline']:
        lines.append(f"Loss: {summary['sparkline']}")
    if summary['torch_version']:
        lines.append(f"torch: {summary['torch_version']}")
    if issues:
        lines.append('Issues:')
        for issue in issues:
            lines.append(f"  [{issue.level}] {issue.code}: {issue.message}")
    else:
        lines.append('Issues: none')
    return '\n'.join(lines) + '\n'


def format_markdown(summary, issues):
    rows = [
        ('Name', summary['name']),
        ('Status', summary['status']),
        ('Script', summary['script']),
        ('World', summary['world']),
        ('Batch', summary['batch']),
        ('Seed', summary['seed']),
        ('Steps', f"{_fmt(summary['logged_steps'])} / {_fmt(summary['planned_steps'])}"),
        ('Train loss', f"{_fmt(summary['first_loss'])} → {_fmt(summary['last_loss'])}"),
        ('Min train loss', f"{_fmt(summary['min_loss'])} @ step {summary['min_loss_step']}"),
        ('Val loss', summary['val_loss']),
        ('Time', format_duration(summary['train_seconds'])),
        ('Throughput', f"{summary['tokens_per_second']:,.0f} tok/s" if summary['tokens_per_second'] else None),
        ('Tokens', summary['tokens']),
        ('Draws / update', summary['recipe_draws']),
        ('ETA', format_duration(summary['eta_seconds'])),
    ]
    lines = [f"# Run {summary['name']}", '', '| Field | Value |', '| --- | --- |']
    for key, value in rows:
        if value is None:
            continue
        lines.append(f'| {key} | {_fmt(value) if not isinstance(value, str) else value} |')
    if summary['sparkline']:
        lines += ['', f"Loss: `{summary['sparkline']}`"]
    lines += ['', '## Issues']
    if not issues:
        lines.append('None.')
    else:
        for issue in issues:
            lines.append(f"- **{issue.level}** `{issue.code}`: {issue.message}")
    return '\n'.join(lines) + '\n'


def format_json(summary, issues):
    payload = dict(summary)
    payload['issues'] = [asdict(issue) for issue in issues]
    return json.dumps(payload, indent=2) + '\n'


def format_report(summary, issues, fmt='text'):
    if fmt == 'json':
        return format_json(summary, issues)
    if fmt == 'markdown':
        return format_markdown(summary, issues)
    if fmt == 'text':
        return format_text(summary, issues)
    raise ValueError(f'unknown format: {fmt}')


def inspect_path(path, fmt='text', spark_width=48):
    run = load_run(path)
    issues = validate_run(run)
    summary = summarize_run(run, spark_width=spark_width)
    return format_report(summary, issues, fmt), issues


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dir', type=Path, help='Directory with config.json, metrics.jsonl, result.json.')
    parser.add_argument('--format', choices=('text', 'json', 'markdown'), default='text')
    parser.add_argument('--strict', action='store_true', help='Also fail the exit code on warnings.')
    parser.add_argument('--spark-width', type=int, default=48, help='Sparkline columns (default 48).')
    args = parser.parse_args(argv)
    if args.spark_width < 1:
        parser.error('--spark-width must be >= 1')
    if not args.run_dir.exists():
        print(f'error: {args.run_dir} does not exist', file=sys.stderr)
        return 2
    if not args.run_dir.is_dir():
        print(f'error: {args.run_dir} is not a directory', file=sys.stderr)
        return 2
    report, issues = inspect_path(args.run_dir, fmt=args.format, spark_width=args.spark_width)
    sys.stdout.write(report)
    errors = [i for i in issues if i.level == 'error']
    if errors or (args.strict and issues):
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
