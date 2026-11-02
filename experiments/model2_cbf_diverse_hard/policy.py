"""Hard-CBF variant of the v2 recovery policy.

The parent still samples four model-2 trajectories and previews each through
the model-2 filter. Both live filters must have zero safety slack enabled.
"""

from experiments.model2_cbf_diverse_v2.policy import DiverseCBFModel2Policy


class HardCBFModel2Policy(DiverseCBFModel2Policy):
    bounded_soft_cbf_enabled = False
    bounded_soft_cbf_cap_px2 = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_cbf_configs()

    def _validate_cbf_configs(self):
        """Allow an explicit child policy to validate separate stage profiles."""
        if self.mind_filter is None or self._regrasp_mind_filter is None:
            raise RuntimeError("Both model-1 and model-2 CBFs must be enabled")
        configs = (self.mind_config, self.mind_filter.config,
                   self._regrasp_mind_filter.config)
        if self.bounded_soft_cbf_enabled:
            if any(config.hard_cbf_enabled or
                   config.safety_slack_cap_px2 != self.bounded_soft_cbf_cap_px2
                   for config in configs):
                raise RuntimeError("Both CBFs must use the configured bounded allowance")
            print(f"[BOUNDED SOFT CBF] model 1 and model 2: "
                  f"each safety slack <= {self.bounded_soft_cbf_cap_px2:g} px^2", flush=True)
            return
        if not all(config.hard_cbf_enabled for config in configs):
            raise RuntimeError("Both model-1 and model-2 CBFs must be hard")
        print("[HARD CBF] model 1 and model 2: safety slack fixed to zero; "
              "no-incumbent fallback holds the full execution chunk",
              flush=True)
