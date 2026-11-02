"""Record every clustering/switch decision in addition to existing metrics."""
from pathlib import Path
from experiments.model2_cbf_diverse_hard_infeasible_regrasp.runner import (
    PushTKeypointsRunner as BaseRunner,
)


class PushTKeypointsRunner(BaseRunner):
    def run(self, policy):
        path = Path(self.output_dir) / 'model2_route_trace.jsonl'
        with path.open('w', buffering=1) as trace:
            policy._route_trace_file = trace
            try:
                result = super().run(policy)
            finally:
                policy._route_trace_file = None
        result.update({f'model2_routes/{key}': value
                       for key, value in policy._route_counters.items()})
        result['model2_routes/reused_certificates'] = policy._regrasp_mind_filter._dedup_reuse_count
        result['model2_routes/trace_path'] = str(path)
        result['model2_routes/config'] = dict(
            similarity_threshold_px=policy.route_similarity_px,
            progress_window_steps=policy.route_progress_window_steps,
            progress_px=policy.route_progress_px)
        return result
