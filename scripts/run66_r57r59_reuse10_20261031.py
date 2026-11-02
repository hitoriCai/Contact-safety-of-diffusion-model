"""Run only missing scenes, using the unchanged mixed-stage evaluation entry."""
import concurrent.futures
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path('/home/cps/czt/2026_summer/0928_lowdim/eval_lowdim_0.8964')
sys.path.insert(0, str(ROOT))
from experiments.model2_cbf_local25_mixed_stages import run66 as common

OUTPUT = ROOT / 'data/eval/10.31_59px+有界软约束'
config = json.loads((OUTPUT / 'experiment_config.json').read_text())
for file, digest in config['code_sha256'].items():
    if common.hashlib.sha256((ROOT / file).read_bytes()).hexdigest() != digest:
        raise RuntimeError('Controller code changed after preparation: ' + file)
seeds = config['seeds']
completed = []
for seed in seeds:
    if (OUTPUT / 'results' / f'seed_{seed}' / 'eval_log.json').is_file():
        common.run_one(OUTPUT, seed, config['commands'][str(seed)])
        completed.append(seed)
pending = [s for s in seeds if s not in completed]
reused = config['reuse']['seeds']
state = dict(state='running', expected=len(seeds), completed=completed,
             reused=reused, newly_completed=[s for s in completed if s not in reused],
             failed={}, remaining=len(pending))
common.write_json(OUTPUT / 'progress.json', state)
common.compare(OUTPUT, completed, config['original_metrics'])
print(f'Reused={len(reused)}, already complete={len(completed)}, starting={len(pending)}', flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    futures = {pool.submit(common.run_one, OUTPUT, seed, config['commands'][str(seed)]): seed for seed in pending}
    for future in concurrent.futures.as_completed(futures):
        seed = futures[future]
        try:
            future.result()
            state['completed'].append(seed)
            state['newly_completed'].append(seed)
        except Exception as error:
            state['failed'][str(seed)] = repr(error)
        state['remaining'] = len(seeds) - len(state['completed']) - len(state['failed'])
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        common.write_json(OUTPUT / 'progress.json', state)
        common.compare(OUTPUT, state['completed'], config['original_metrics'])
        print(f"completed={len(state['completed'])}/{len(seeds)} reused={len(reused)} failed={len(state['failed'])}", flush=True)
state['state'] = 'failed' if state['failed'] else 'complete'
common.write_json(OUTPUT / 'progress.json', state)
common.compare(OUTPUT, state['completed'], config['original_metrics'], final=not state['failed'])
