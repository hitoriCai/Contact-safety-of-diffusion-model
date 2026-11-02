from experiments.contact_gap_5s.runner_contact_gap import PushTKeypointsRunner as BaseRunner


class PushTKeypointsRunner(BaseRunner):
    def run(self, policy):
        result = super().run(policy)
        filter_ = getattr(policy, "_regrasp_mind_filter", None)
        if filter_ is not None:
            for key, value in filter_.diagnostics_summary().items():
                result[f"model2_cbf/{key}"] = value
        result["model2_candidates/replans"] = policy._candidate_replans
        result["model2_candidates/moving_selections"] = policy._moving_selections
        result["model2_candidates/stuck_switches"] = policy._stuck_switches
        return result
