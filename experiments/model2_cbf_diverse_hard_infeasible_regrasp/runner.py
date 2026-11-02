"""Expose the new trigger count alongside existing model-2 CBF metrics."""

from experiments.model2_cbf_diverse_v2.runner import (
    PushTKeypointsRunner as BaseRunner,
)


class PushTKeypointsRunner(BaseRunner):
    def run(self, policy):
        result = super().run(policy)
        result["regrasp/model1_infeasible_trigger_count"] = int(
            policy._model1_infeasible_regrasp_count_total)
        result["model2_candidates/sample_count"] = int(
            policy._candidate_sampled_total)
        result["model2_candidates/extra_batches"] = int(
            policy._candidate_extra_batches_total)
        result["model2_candidates/all_infeasible_replans"] = int(
            policy._candidate_all_infeasible_total)
        return result
