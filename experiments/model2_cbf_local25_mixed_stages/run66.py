"""Rerun saved scenes with configurable mixed-stage CBF radii; preserve other settings."""

import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from statistics import mean
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ('data/eval/NNDM_dim24_model2v63_e500_r52_local25_1step_'
                 'k12_route_memory_rand56_hard200_stop10_20261026')
ENTRY = 'experiments/model2_cbf_local25_mixed_stages/eval.py'
FILES = (
    'eval.py', 'diffusion_policy/policy/mind.py',
    'diffusion_policy/policy/diffusion_unet_lowdim_policy.py',
    'experiments/contact_gap_5s/policy_contact_gap.py',
    'experiments/model2_cbf_diverse_hard/policy.py',
    'experiments/model2_cbf_diverse_hard_infeasible_regrasp/policy.py',
    'experiments/model2_cbf_full_page/filter.py',
    'experiments/model2_cbf_full_page/policy.py',
    'experiments/model2_cbf_full_page_1step_p047/policy.py',
    'experiments/model2_cbf_local25_dedup/policy.py',
    'experiments/model2_cbf_local25_route_memory/policy.py',
    'experiments/model2_cbf_local25_route_memory/routes.py',
    'experiments/nndm_dim24_model2_v63_eval/runner_core_stop10.py',
    ENTRY, 'experiments/model2_cbf_local25_mixed_stages/policy.py',
    'experiments/model2_cbf_local25_mixed_stages/runner.py',
    'experiments/model2_cbf_local25_mixed_stages/run66.py',
)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def set_arg(command, flag, value):
    if flag in command:
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend((flag, str(value)))


def setup(args):
    source = args.source_dir.resolve()
    mixed_source = (source / 'experiment_config.json').is_file()
    source_config = json.loads((source / 'experiment_config.json').read_text()) if mixed_source else None
    if mixed_source:
        all_seeds = list(map(int, source_config['seeds']))
        parts = {int(seed): dict(command=command) for seed, command in source_config['commands'].items()}
    else:
        original = json.loads((source / 'aggregate_partial.json').read_text())
        prefix = 'test/sim_max_reward_'
        all_seeds = [int(key[len(prefix):]) for key in original if key.startswith(prefix)]
        parts = {}
        for group in range(1, 5):
            group_config = json.loads((source / f'group_{group:02d}/results/experiment_config.json').read_text())
            for part in group_config['parts']:
                for seed in part['seeds']:
                    parts[int(seed)] = part
    if len(set(all_seeds)) != len(all_seeds):
        raise RuntimeError('Source seeds must be distinct')
    seeds = args.seeds if args.seeds is not None else all_seeds
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset(all_seeds):
        raise ValueError('Requested seeds must be distinct members of the source experiment')
    if args.smoke_seed is not None:
        if args.smoke_seed not in seeds:
            raise ValueError('Smoke seed must belong to the requested cohort')
        seeds = [args.smoke_seed]
    output = args.output_dir.resolve()
    if output == source:
        raise ValueError('New results must not overwrite the source experiment')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifests').mkdir(exist_ok=True)
    commands, original_metrics, expected_profiles, manifest_hashes = {}, {}, {}, {}
    for seed in seeds:
        command = list(parts[seed]['command'])
        # The source folder may have been renamed after its original run.
        local_manifest = source / 'manifests' / f'seed_{seed}.json'
        manifest = local_manifest if local_manifest.exists() else Path(command[command.index('--test_seed_file') + 1])
        scene = json.loads(manifest.read_text())
        if scene['seeds'] != [seed] or [row['seed'] for row in scene['scenarios']] != [seed]:
            raise RuntimeError(f'Source part for {seed} does not contain exactly that scene')
        manifest_out = output / 'manifests' / f'seed_{seed}.json'
        write_json(manifest_out, scene)
        if json.loads(manifest_out.read_text()) != scene:
            raise RuntimeError('Initial scene changed while copying')
        manifest_hashes[str(seed)] = dict(
            source=str(manifest), source_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
            copied_sha256=hashlib.sha256(manifest_out.read_bytes()).hexdigest())
        command[1] = ENTRY
        set_arg(command, '--output_dir', output / 'results' / f'seed_{seed}')
        set_arg(command, '--test_seed_file', manifest_out)
        for flag in ('--side_proxy_radius', '--lower_proxy_radius'):
            set_arg(command, flag, args.model1_radius)
        for flag in ('--regrasp_side_proxy_radius', '--regrasp_lower_proxy_radius'):
            set_arg(command, flag, args.model2_radius)
        set_arg(command, '--tracking_cost_multiplier', 1)
        if args.smoke_seed is not None:
            for flag in ('--score_extension_trigger_steps', '--score_extension_max_steps',
                         '--score_extension_threshold'):
                if flag in command:
                    index = command.index(flag)
                    del command[index:index + 2]
            set_arg(command, '--max_steps', args.smoke_steps)
        commands[str(seed)] = command
        if mixed_source:
            source_run = source / 'results' / f'seed_{seed}'
            metrics = json.loads((source_run / 'eval_log.json').read_text())
            original_metrics[str(seed)] = dict(
                score=metrics[f'test/sim_max_reward_{seed}'],
                collision=metrics[f'test/sim_collision_{seed}'],
                stop10=metrics[f'test/regrasp_infeasible_10_terminated_{seed}'])
            profiles = json.loads((source_run / 'effective_cbf_profiles.json').read_text())
            for stage, radius in (('model1', args.model1_radius), ('model2', args.model2_radius)):
                profiles[stage]['safety_proxy_radii'] = [radius] * 3
                profiles[stage]['block_clearance_radius'] = radius
            expected_profiles[str(seed)] = profiles
            if args.smoke_seed is None:
                previous = list(parts[seed]['command'])
                # Enforce an exact command match apart from paths and the four radii.
                for flag in ('--output_dir', '--test_seed_file', '--side_proxy_radius',
                             '--lower_proxy_radius', '--regrasp_side_proxy_radius',
                             '--regrasp_lower_proxy_radius'):
                    previous[previous.index(flag) + 1] = command[command.index(flag) + 1]
                if previous != command:
                    raise RuntimeError(f'Unexpected non-radius command change for seed {seed}')
        else:
            original_metrics[str(seed)] = dict(score=original[f'test/sim_max_reward_{seed}'],
                collision=original[f'test/sim_collision_{seed}'],
                stop10=original[f'test/regrasp_infeasible_10_terminated_{seed}'])
    hashes = {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in FILES}
    config = dict(
        created_at=datetime.now(timezone.utc).isoformat(), source=str(source),
        seeds=seeds, all_source_seeds=all_seeds, smoke_only=args.smoke_seed is not None,
        workers=args.workers, commands=commands, code_sha256=hashes,
        model1=dict(radii_px=[args.model1_radius] * 3, hard_cbf=True,
                    safety_slack_cap_px2=None, tracking_cost_multiplier=1),
        model2=dict(radii_px=[args.model2_radius] * 3, hard_cbf=False,
                    safety_slack_cap_px2_per_constraint=300, tracking_cost_multiplier=40),
        original_metrics=original_metrics, expected_profiles=expected_profiles,
        manifests=manifest_hashes)
    path = output / 'experiment_config.json'
    if path.exists():
        existing = json.loads(path.read_text())
        for key in ('seeds', 'code_sha256', 'commands', 'expected_profiles', 'model1', 'model2'):
            if existing.get(key) != config[key]:
                raise RuntimeError(f'Output directory has different {key}')
    else:
        write_json(path, config)
    return output, seeds, commands, original_metrics

def run_one(output, seed, command):
    run_dir = output / 'results' / f'seed_{seed}'
    metrics = run_dir / 'eval_log.json'
    if not metrics.exists():
        logs = output / 'logs'
        logs.mkdir(exist_ok=True)
        env = dict(os.environ, SDL_VIDEODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1')
        with (logs / f'seed_{seed}.log').open('w') as log:
            process = subprocess.run(command, cwd=ROOT, env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
        if process.returncode:
            raise RuntimeError(f'eval exit {process.returncode}; see logs/seed_{seed}.log')
    data = json.loads(metrics.read_text())
    if f'test/sim_max_reward_{seed}' not in data:
        raise RuntimeError('Incomplete eval JSON')
    profiles = json.loads((run_dir / 'effective_cbf_profiles.json').read_text())
    if (not profiles['model1']['hard_cbf_enabled']
            or profiles['model1']['safety_slack_cap_px2'] is not None
            or profiles['model1']['safety_proxy_radii'] != [float(command[command.index('--side_proxy_radius') + 1])] * 3
            or profiles['model2']['hard_cbf_enabled']
            or profiles['model2']['safety_proxy_radii'] != [float(command[command.index('--regrasp_side_proxy_radius') + 1])] * 3
            or profiles['model2']['safety_slack_cap_px2'] != 300.
            or profiles['model2']['tracking_cost_multiplier'] != 40.
            or data['mind/max_safety_slack'] > 1.e-6):
        raise RuntimeError('Effective CBF profiles differ from the requested experiment')
    expected = json.loads((output / 'experiment_config.json').read_text()).get('expected_profiles', {}).get(str(seed))
    if expected is not None and profiles != expected:
        raise RuntimeError('A non-radius effective profile value changed from the source experiment')
    write_json(run_dir / 'summary.json', data)


def compare(output, completed, original, final=False):
    rows = []
    for seed in sorted(completed):
        d = json.loads((output / 'results' / f'seed_{seed}' / 'eval_log.json').read_text())
        rows.append(dict(seed=seed, original=original[str(seed)], new=dict(
            score=d[f'test/sim_max_reward_{seed}'],
            collision=d[f'test/sim_collision_{seed}'],
            stop10=d[f'test/regrasp_infeasible_10_terminated_{seed}'],
            success=d['test/success_count'], safe_success=d['test/safe_success_count'],
            model1_no_solution=d['mind/no_solution_count'],
            model2_no_solution=d['model2_cbf/no_solution_count'])))
    if not rows:
        return
    aggregates = {}
    for version in ('original', 'new'):
        values = [row[version] for row in rows]
        aggregates[version] = dict(n=len(values), mean_score=mean(v['score'] for v in values),
            collisions=sum(bool(v['collision']) for v in values),
            stop10=sum(bool(v['stop10']) for v in values),
            zero_score=sum(v['score'] == 0 for v in values),
            score_ge_085=sum(v['score'] >= .85 for v in values),
            score_ge_095=sum(v['score'] >= .95 for v in values))
    for key in ('success', 'safe_success', 'model1_no_solution', 'model2_no_solution'):
        aggregates['new'][key] = sum(row['new'][key] for row in rows)
    result = dict(complete=final, aggregate=aggregates, by_seed=rows)
    write_json(output / 'comparison_vs_previous.json', result)
    lines = ['# Same-seed comparison: source experiment vs current mixed-stage radii', '',
             f'Completed scenes: {len(rows)}. Final: {final}.', '',
             '| Metric | Previous | Current |', '| --- | ---: | ---: |']
    for key in ('n', 'mean_score', 'collisions', 'stop10', 'zero_score', 'score_ge_085', 'score_ge_095'):
        lines.append(f"| {key} | {aggregates['original'][key]} | {aggregates['new'][key]} |")
    lines += ['', '| Seed | Previous score | New score | Previous collision | New collision |',
              '| ---: | ---: | ---: | :---: | :---: |']
    for row in rows:
        a, b = row['original'], row['new']
        lines.append(f"| {row['seed']} | {a['score']:.6f} | {b['score']:.6f} | {a['collision']} | {b['collision']} |")
    (output / 'comparison_vs_previous.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--source_dir', type=Path, default=SOURCE)
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--model1_radius', type=float, default=52.)
    parser.add_argument('--model2_radius', type=float, default=54.)
    parser.add_argument('--smoke_seed', type=int)
    parser.add_argument('--smoke_steps', type=int, default=40)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if not all(0 < radius < float('inf') for radius in (args.model1_radius, args.model2_radius)):
        parser.error('Stage radii must be finite and positive')
    output, seeds, commands, original = setup(args)
    state = dict(state='running', expected=len(seeds), completed=[], failed={})
    write_json(output / 'progress.json', state)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, output, seed, commands[str(seed)]): seed for seed in seeds}
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            try:
                future.result()
                state['completed'].append(seed)
            except Exception as error:
                state['failed'][str(seed)] = repr(error)
            state['updated_at'] = datetime.now(timezone.utc).isoformat()
            write_json(output / 'progress.json', state)
            compare(output, state['completed'], original)
            print(f"completed={len(state['completed'])}/{len(seeds)} failed={len(state['failed'])}", flush=True)
    state['state'] = 'failed' if state['failed'] else 'complete'
    write_json(output / 'progress.json', state)
    compare(output, state['completed'], original, final=not state['failed'])


if __name__ == '__main__':
    main()
