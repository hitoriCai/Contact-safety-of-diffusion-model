"""Portable wrapper around the original, byte-preserved mixed-CBF entry."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify():
    manifest = json.loads((ROOT / 'source_inventory/manifest.json').read_text())
    for record in manifest['files']:
        path = ROOT / record['repository_path']
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size < 256 and path.suffix == '.ckpt':
            if path.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise RuntimeError('Checkpoint is an LFS pointer. Run git lfs install && git lfs pull.')
        if sha256(path) != record['sha256']:
            raise RuntimeError('Source/artifact checksum mismatch: ' + str(path))
    print(f"Verified {len(manifest['files'])} original source/artifact files.", flush=True)


def set_arg(command, flag, value):
    command[command.index(flag) + 1] = str(value)


def make_command(config, seed, output, device, max_steps=None):
    command = list(config['commands'][str(seed)])
    command[0] = sys.executable
    command[1] = str(ROOT / command[1])
    replacements = {
        '--checkpoint': ROOT / 'checkpoints/model1_epoch0600.ckpt',
        '--regrasp_model2_checkpoint': ROOT / 'checkpoints/model2_v6_3_epoch0500.ckpt',
        '--mind_checkpoint': ROOT / 'checkpoints/nndm_dim24.pth',
        '--test_seed_file': ROOT / 'scenarios' / f'seed_{seed}.json',
        '--output_dir': output / 'results' / f'seed_{seed}',
        '--device': device,
    }
    for flag, value in replacements.items():
        set_arg(command, flag, value)
    if max_steps is not None:
        # An explicit smoke test, never silently used in a full reproduction.
        for flag in ('--score_extension_trigger_steps', '--score_extension_max_steps',
                     '--score_extension_threshold'):
            if flag in command:
                index = command.index(flag)
                del command[index:index + 2]
        if '--max_steps' in command:
            set_arg(command, '--max_steps', max_steps)
        else:
            command.extend(('--max_steps', str(max_steps)))
    return command


def run_one(seed, command, output):
    metrics = output / 'results' / f'seed_{seed}' / 'eval_log.json'
    if not metrics.exists():
        logs = output / 'logs'
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / f'seed_{seed}.log').open('w') as log:
            result = subprocess.run(command, cwd=ROOT,
                env=dict(os.environ, SDL_VIDEODRIVER='dummy', PYTHONDONTWRITEBYTECODE='1'),
                stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Seed {seed} exited {result.returncode}; see logs/seed_{seed}.log')
    data = json.loads(metrics.read_text())
    profiles = json.loads((metrics.parent / 'effective_cbf_profiles.json').read_text())
    if (not profiles['model1']['hard_cbf_enabled']
            or profiles['model1']['safety_proxy_radii'] != [57.] * 3
            or profiles['model1']['tracking_cost_multiplier'] != 1.
            or profiles['model2']['hard_cbf_enabled']
            or profiles['model2']['safety_proxy_radii'] != [59.] * 3
            or profiles['model2']['safety_slack_cap_px2'] != 300.
            or profiles['model2']['tracking_cost_multiplier'] != 40.):
        raise RuntimeError('Effective controller configuration differs from the published experiment')
    return dict(seed=seed, score=data[f'test/sim_max_reward_{seed}'],
                collision=data[f'test/sim_collision_{seed}'],
                success=data['test/success_count'], safe_success=data['test/safe_success_count'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--all', action='store_true', help='Run all fixed66 scenarios')
    selection.add_argument('--seed', type=int, nargs='+', help='Run selected fixed scenario seeds')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/r57_hard_r59_soft')
    parser.add_argument('--max-steps', type=int, help='Explicit short smoke test; overrides300/700 rollout budget')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without loading models or running evaluation')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    config = json.loads((ROOT / 'experiment/experiment_config.json').read_text())
    if args.verify_only:
        verify()
        return
    if not args.all and args.seed is None:
        parser.error('Choose --all, --seed, or --verify-only')
    seeds = config['seeds'] if args.all else args.seed
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset(config['seeds']):
        parser.error('Seeds must be distinct members of the published66-scene cohort')
    output = args.output_dir.resolve()
    commands = {seed: make_command(config, seed, output, args.device, args.max_steps) for seed in seeds}
    if args.dry_run:
        print(json.dumps({str(s): c for s, c in commands.items()}, indent=2))
        return
    verify()
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'portable_commands.json'
    command_record = {str(s): c for s, c in commands.items()}
    if snapshot.exists() and json.loads(snapshot.read_text()) != command_record:
        raise RuntimeError('Output directory belongs to a different set of commands')
    snapshot.write_text(json.dumps(command_record, indent=2) + '\n')
    rows, failures = [], {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, seed, cmd, output): seed for seed, cmd in commands.items()}
        for future in concurrent.futures.as_completed(futures):
            try:
                rows.append(future.result())
            except Exception as error:
                failures[str(futures[future])] = str(error)
            summary = dict(expected=len(seeds), completed=len(rows), failures=failures, by_seed=rows)
            if rows:
                summary.update(mean_score=sum(r['score'] for r in rows) / len(rows),
                               collisions=sum(bool(r['collision']) for r in rows),
                               successes=sum(r['success'] for r in rows),
                               safe_successes=sum(r['safe_success'] for r in rows))
            temporary = output / 'summary.json.tmp'
            temporary.write_text(json.dumps(summary, indent=2) + '\n')
            temporary.replace(output / 'summary.json')
            print(f'Completed {len(rows)}/{len(seeds)}, failed={len(failures)}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
