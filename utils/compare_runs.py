"""Compare two or more training run directories written by Noise training scripts.

    uv run utils/compare_runs.py logs/val logs/speedrun
    uv run utils/compare_runs.py logs/a logs/b --format markdown
    uv run utils/compare_runs.py logs/a logs/b logs/c --at-step 100
    uv run utils/compare_runs.py logs/a logs/b --names baseline candidate

The first directory is the baseline. Later directories are scored against it
at the last step present in every log (or at ``--at-step``). Prints a table of
losses and timings, a config diff, and a ranking. Does not load weights or
run the model.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import sys

PRED_TOKENS = 2048
LOSS_KEYS = ('train_loss', 'population_loss')
MISSING = object()


@dataclass
class RunLog:
    path: Path
    name: str
    config: dict = field(default_factory=dict)
    metrics: list[dict] = field(default_factory=list)
    result: dict | None = None
    missing: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


def is_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def row_loss(row):
    if not row:
        return None
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
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f'{path.name}: expected a JSON object')
    return data


def load_metrics(path):
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


def load_run(path, name=None):
    path = Path(path)
    run = RunLog(path=path, name=name or default_name(path))
    if not path.is_dir():
        run.parse_errors.append(f'not a directory: {path}')
        return run
    for name_, attr in (('config.json', 'config'), ('result.json', 'result')):
        file = path / name_
        if not file.is_file():
            run.missing.append(name_)
            continue
        try:
            setattr(run, attr, load_json_file(file))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            run.parse_errors.append(f'{name_}: {exc}')
    metrics_path = path / 'metrics.jsonl'
    if not metrics_path.is_file():
        run.missing.append('metrics.jsonl')
    else:
        run.metrics, metric_errors = load_metrics(metrics_path)
        run.parse_errors.extend(metric_errors)
    if run.config is None:
        run.config = {}
    return run


def flatten(obj, prefix=''):
    """Flatten nested dicts; keep lists of scalars as a single value."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            child = f'{prefix}.{key}' if prefix else str(key)
            out.update(flatten(value, child))
        return out
    if isinstance(obj, list) and any(isinstance(x, (dict, list)) for x in obj):
        out = {}
        for i, value in enumerate(obj):
            out.update(flatten(value, f'{prefix}[{i}]'))
        return out
    return {prefix: obj} if prefix else {}


def normalize_config(config):
    cfg = dict(config or {})
    if 'script' in cfg:
        cfg['script'] = short_script(cfg['script'])
    return cfg


def config_diff(configs):
    """Keys whose flattened values are not identical across every config."""
    maps = [flatten(normalize_config(cfg)) for cfg in configs]
    keys = sorted(set().union(*(m.keys() for m in maps)))
    diffs = []
    for key in keys:
        values = [m[key] if key in m else MISSING for m in maps]
        present = [v for v in values if v is not MISSING]
        if len(present) != len(values) or any(_neq(v, present[0]) for v in present):
            diffs.append({
                'path': key,
                'values': [None if v is MISSING else v for v in values],
            })
    return diffs


def _neq(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isfinite(a) and math.isfinite(b):
        return abs(a - b) > 1e-12
    return a != b


def metrics_by_step(run):
    out = {}
    for row in run.metrics:
        step = row.get('step')
        if isinstance(step, int) and not isinstance(step, bool):
            out[step] = row
    return out


def last_common_step(runs):
    sets = [set(metrics_by_step(run)) for run in runs]
    if not sets or any(not steps for steps in sets):
        return None
    common = set.intersection(*sets)
    return max(common) if common else None


def snapshot_at(runs, step):
    rows = []
    for run in runs:
        row = metrics_by_step(run).get(step)
        rows.append({
            'name': run.name,
            'path': str(run.path),
            'step': step,
            'present': row is not None,
            'loss_key': loss_key([row]) if row else None,
            'train_loss': row_loss(row) if row and is_finite(row_loss(row)) else None,
            'train_seconds': float(row['train_seconds']) if row and is_finite(row.get('train_seconds')) else None,
            'val_loss': float(row['val_loss']) if row and is_finite(row.get('val_loss')) else None,
        })
    baseline = rows[0]
    deltas = []
    for item in rows[1:]:
        delta = {'name': item['name']}
        for field in ('train_loss', 'train_seconds', 'val_loss'):
            left, right = baseline.get(field), item.get(field)
            delta[field] = (right - left) if is_finite(left) and is_finite(right) else None
        if is_finite(baseline.get('train_seconds')) and is_finite(item.get('train_seconds')) and item['train_seconds'] > 0:
            delta['time_ratio'] = baseline['train_seconds'] / item['train_seconds']
        else:
            delta['time_ratio'] = None
        deltas.append(delta)
    return {'step': step, 'rows': rows, 'deltas_vs_baseline': deltas}


def run_stats(run):
    config = run.config or {}
    metrics = run.metrics
    result = run.result or {}
    steps = [row.get('step') for row in metrics if isinstance(row.get('step'), int) and not isinstance(row.get('step'), bool)]
    last_step = steps[-1] if steps else None
    planned = config.get('steps') if isinstance(config.get('steps'), int) else None
    batch = config.get('batch') if isinstance(config.get('batch'), int) else None
    finite = [row_loss(row) for row in metrics if is_finite(row_loss(row))]
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
    val_loss = result.get('val_loss') if is_finite(result.get('val_loss')) else None
    if val_loss is None:
        for row in reversed(metrics):
            if is_finite(row.get('val_loss')):
                val_loss = float(row['val_loss'])
                break
    complete = bool(run.result) and (planned is None or last_step is None or last_step >= planned)
    return {
        'name': run.name,
        'path': str(run.path),
        'script': short_script(config.get('script')),
        'world': config.get('world'),
        'batch': batch,
        'seed': config.get('seed'),
        'target': config.get('target'),
        'planned_steps': planned,
        'logged_steps': last_step,
        'complete': complete,
        'loss_key': loss_key(metrics),
        'first_loss': finite[0] if finite else None,
        'last_loss': finite[-1] if finite else None,
        'val_loss': val_loss,
        'reached_target': result.get('reached_target') if 'reached_target' in result else None,
        'train_seconds': last_time,
        'tokens': int(tokens) if tokens is not None else None,
        'tokens_per_second': (tokens / last_time) if tokens and last_time and last_time > 0 else None,
        'recipe_draws': recipe_draws(config.get('recipe')),
        'missing': list(run.missing),
        'parse_errors': list(run.parse_errors),
        'ok': not run.parse_errors and 'metrics.jsonl' not in run.missing and bool(metrics),
    }


def rank_runs(stats):
    def key(stat):
        val, loss, seconds = stat.get('val_loss'), stat.get('last_loss'), stat.get('train_seconds')
        return (
            0 if is_finite(val) else 1,
            float(val) if is_finite(val) else (float(loss) if is_finite(loss) else math.inf),
            float(seconds) if is_finite(seconds) else math.inf,
            stat.get('name') or '',
        )

    ordered = sorted(stats, key=key)
    ranking = []
    for index, stat in enumerate(ordered, 1):
        ranking.append({
            'rank': index,
            'name': stat['name'],
            'val_loss': stat.get('val_loss'),
            'last_loss': stat.get('last_loss'),
            'train_seconds': stat.get('train_seconds'),
            'key': 'val_loss' if is_finite(stat.get('val_loss')) else 'last_loss',
        })
    return ranking


def comparability_notes(runs):
    notes = []

    def collect(label, getter):
        values = [getter(run) for run in runs]
        shown = [repr(v) if v is not None else '—' for v in values]
        if len(set(shown)) > 1:
            paired = ', '.join(f'{run.name}={value}' for run, value in zip(runs, shown))
            notes.append(f'{label} differs ({paired})')

    collect('seed', lambda r: r.config.get('seed'))
    collect('batch', lambda r: r.config.get('batch'))
    collect('world', lambda r: r.config.get('world'))
    collect('planned steps', lambda r: r.config.get('steps'))
    collect('loss key', lambda r: loss_key(r.metrics))
    worlds = {r.config.get('world') for r in runs}
    if len(worlds) > 1:
        notes.append('wall-clock times are not comparable across different world sizes')
    if any('result.json' in run.missing for run in runs):
        unfinished = [run.name for run in runs if 'result.json' in run.missing]
        notes.append('no result.json yet: ' + ', '.join(unfinished))
    return notes


def compare_runs(paths, names=None, at_step=None):
    paths = [Path(p) for p in paths]
    if names is None:
        names = [default_name(p) for p in paths]
    if len(names) != len(paths):
        raise ValueError('names must match the number of run directories')
    # Keep labels unique so a table of two "log" dirs stays readable.
    seen = {}
    unique = []
    for name, path in zip(names, paths):
        count = seen.get(name, 0) + 1
        seen[name] = count
        unique.append(name if count == 1 else f'{name}#{count}')
    runs = [load_run(path, name) for path, name in zip(paths, unique)]
    stats = [run_stats(run) for run in runs]
    chosen = at_step if at_step is not None else last_common_step(runs)
    snapshot = snapshot_at(runs, chosen) if chosen is not None else None
    return {
        'baseline': unique[0],
        'runs': stats,
        'config_diff': config_diff([run.config for run in runs]),
        'comparability': comparability_notes(runs),
        'snapshot': snapshot,
        'snapshot_source': 'at_step' if at_step is not None else 'last_common',
        'ranking': rank_runs(stats),
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
    if isinstance(value, list):
        return json.dumps(value, separators=(',', ':'))
    return str(value)


def _signed(value, digits=4):
    if not is_finite(value):
        return '—'
    sign = '+' if value > 0 else ''
    if abs(value) >= 100:
        return f'{sign}{value:,.2f}'
    return f'{sign}{value:.{digits}f}'


def _cell_table(headers, rows):
    str_rows = [[('—' if c is None else str(c)) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def line(cells):
        parts = []
        for i, (cell, width) in enumerate(zip(cells, widths)):
            parts.append(cell.ljust(width) if i == 0 else cell.rjust(width))
        return '  '.join(parts)

    out = [line(headers), line(['-' * w for w in widths])]
    out.extend(line(row) for row in str_rows)
    return '\n'.join(out)


def format_text(comparison):
    lines = [f"Comparing {len(comparison['runs'])} runs  (baseline: {comparison['baseline']})", '']
    table_rows = []
    for stat in comparison['runs']:
        table_rows.append([
            stat['name'],
            'done' if stat['complete'] else 'open',
            _fmt(stat['logged_steps']),
            _fmt(stat['last_loss']),
            _fmt(stat['val_loss']),
            _fmt(stat['train_seconds']),
            f"{stat['tokens_per_second']:,.0f}" if stat['tokens_per_second'] else '—',
            _fmt(stat['recipe_draws']),
        ])
    lines.append(_cell_table(
        ['name', 'status', 'steps', 'last_loss', 'val_loss', 'seconds', 'tok/s', 'draws'],
        table_rows,
    ))
    if comparison['comparability']:
        lines += ['', 'Notes']
        for note in comparison['comparability']:
            lines.append(f'  {note}')
    diffs = comparison['config_diff']
    lines += ['', f'Config differences ({len(diffs)})']
    if not diffs:
        lines.append('  none')
    else:
        headers = ['key'] + [stat['name'] for stat in comparison['runs']]
        rows = [[diff['path'], *(_fmt(v) for v in diff['values'])] for diff in diffs]
        lines.append(_cell_table(headers, rows))
    snap = comparison['snapshot']
    if snap is None:
        lines += ['', 'No common step to compare.']
    else:
        source = 'requested step' if comparison['snapshot_source'] == 'at_step' else 'last common step'
        lines += ['', f'At {source} {snap["step"]}']
        snap_rows = []
        for row in snap['rows']:
            snap_rows.append([
                row['name'],
                'yes' if row['present'] else 'no',
                _fmt(row['train_loss']),
                _fmt(row['train_seconds']),
                _fmt(row['val_loss']),
            ])
        lines.append(_cell_table(['name', 'present', 'loss', 'seconds', 'val_loss'], snap_rows))
        if snap['deltas_vs_baseline']:
            lines.append(f"  vs {comparison['baseline']}:")
            for delta in snap['deltas_vs_baseline']:
                time_note = ''
                if is_finite(delta.get('time_ratio')):
                    ratio = delta['time_ratio']
                    if ratio > 1:
                        time_note = f'  ({ratio:.2f}x faster)'
                    elif ratio < 1 and ratio > 0:
                        time_note = f'  ({1 / ratio:.2f}x slower)'
                lines.append(
                    f"    {delta['name']}:  loss {_signed(delta['train_loss'])}  "
                    f"time {_signed(delta['train_seconds'])} s{time_note}"
                )
    lines += ['', 'Ranking (lower val_loss, then lower last_loss, then less time)']
    for item in comparison['ranking']:
        score = item['val_loss'] if item['key'] == 'val_loss' else item['last_loss']
        lines.append(f"  {item['rank']}. {item['name']}  {_fmt(score)}  {_fmt(item['train_seconds'])} s")
    return '\n'.join(lines) + '\n'


def format_markdown(comparison):
    lines = [f"# Compare runs (baseline: {comparison['baseline']})", '',
             '| name | status | steps | last_loss | val_loss | seconds | tok/s | draws |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for stat in comparison['runs']:
        lines.append('| ' + ' | '.join([
            stat['name'],
            'done' if stat['complete'] else 'open',
            _fmt(stat['logged_steps']),
            _fmt(stat['last_loss']),
            _fmt(stat['val_loss']),
            _fmt(stat['train_seconds']),
            f"{stat['tokens_per_second']:,.0f}" if stat['tokens_per_second'] else '—',
            _fmt(stat['recipe_draws']),
        ]) + ' |')
    if comparison['comparability']:
        lines += ['', '## Notes']
        for note in comparison['comparability']:
            lines.append(f'- {note}')
    lines += ['', '## Config differences']
    if not comparison['config_diff']:
        lines.append('None.')
    else:
        headers = ['key'] + [stat['name'] for stat in comparison['runs']]
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        for diff in comparison['config_diff']:
            lines.append('| ' + ' | '.join([diff['path'], *(_fmt(v) for v in diff['values'])]) + ' |')
    snap = comparison['snapshot']
    lines += ['', '## Snapshot']
    if snap is None:
        lines.append('No common step to compare.')
    else:
        lines.append(f"Step {snap['step']} ({comparison['snapshot_source']}).")
        lines.append('')
        lines.append('| name | loss | seconds | val_loss |')
        lines.append('| --- | ---: | ---: | ---: |')
        for row in snap['rows']:
            lines.append('| ' + ' | '.join([
                row['name'], _fmt(row['train_loss']), _fmt(row['train_seconds']), _fmt(row['val_loss']),
            ]) + ' |')
    lines += ['', '## Ranking']
    for item in comparison['ranking']:
        score = item['val_loss'] if item['key'] == 'val_loss' else item['last_loss']
        lines.append(f"{item['rank']}. **{item['name']}** — {_fmt(score)}, {_fmt(item['train_seconds'])} s")
    return '\n'.join(lines) + '\n'


def format_json(comparison):
    return json.dumps(comparison, indent=2) + '\n'


def format_comparison(comparison, fmt='text'):
    if fmt == 'json':
        return format_json(comparison)
    if fmt == 'markdown':
        return format_markdown(comparison)
    if fmt == 'text':
        return format_text(comparison)
    raise ValueError(f'unknown format: {fmt}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run_dirs', nargs='+', type=Path, help='Run directories; the first is the baseline.')
    parser.add_argument('--format', choices=('text', 'json', 'markdown'), default='text')
    parser.add_argument('--at-step', type=int, help='Compare metrics at this step instead of the last common step.')
    parser.add_argument('--names', nargs='+', help='Labels in the same order as the run directories.')
    args = parser.parse_args(argv)
    if len(args.run_dirs) < 2:
        parser.error('need at least two run directories')
    if args.names is not None and len(args.names) != len(args.run_dirs):
        parser.error('--names must have one label per run directory')
    if args.at_step is not None and args.at_step < 1:
        parser.error('--at-step must be >= 1')
    missing = [p for p in args.run_dirs if not p.exists()]
    if missing:
        for path in missing:
            print(f'error: {path} does not exist', file=sys.stderr)
        return 2
    not_dirs = [p for p in args.run_dirs if not p.is_dir()]
    if not_dirs:
        for path in not_dirs:
            print(f'error: {path} is not a directory', file=sys.stderr)
        return 2
    comparison = compare_runs(args.run_dirs, names=args.names, at_step=args.at_step)
    sys.stdout.write(format_comparison(comparison, args.format))
    if any(not stat['ok'] for stat in comparison['runs']):
        return 1
    if args.at_step is not None:
        snap = comparison['snapshot']
        if snap is None or not any(row['present'] for row in snap['rows']):
            print(f'error: no run has step {args.at_step}', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
