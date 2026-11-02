"""Two-step policy that keeps raw model-2 proposals for global CBF search."""

from experiments.model2_cbf_diverse_hard_infeasible_regrasp.policy import (
    InfeasibleRegraspPolicy,
)


class GlobalThenClipPolicy(InfeasibleRegraspPolicy):
    def _limit_regrasp_action_deltas(self, selected, agent):
        # The CBF filter searches globally and enforces the 25-px execution
        # limit after solving. Do not preclip the diffusion reference.
        if self._regrasp_mind_filter is None:
            raise RuntimeError("Model-2 hard CBF is required")
        return selected.clone()
