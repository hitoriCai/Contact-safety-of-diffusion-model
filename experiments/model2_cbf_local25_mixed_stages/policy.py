"""Hard model-1 CBF and bounded-soft model-2 CBF with configurable radii."""

from dataclasses import replace

from experiments.model2_cbf_local25_route_memory.policy import RouteMemoryLocalPolicy


class MixedStageCBFPolicy(RouteMemoryLocalPolicy):
    # The route selector evaluates only model-2 candidates, so this flag
    # describes the bounded-soft recovery filter.
    bounded_soft_cbf_enabled = True
    bounded_soft_cbf_cap_px2 = 300.0
    require_matching_stage_radii = False

    def _make_regrasp_mind_config(self):
        return replace(
            super()._make_regrasp_mind_config(),
            hard_cbf_enabled=False,
            safety_slack_cap_px2=300.0,
            tracking_cost_multiplier=40.0,
        )

    def _validate_cbf_configs(self):
        if self.mind_filter is None or self._regrasp_mind_filter is None:
            raise RuntimeError("Both stage filters must be enabled")
        main_radii = tuple(float(r) for r in self.mind_config.safety_proxy_radii)
        recovery_radii = tuple(float(r) for r in self._regrasp_region_proxy_radii)
        for config in (self.mind_config, self.mind_filter.config):
            if (not config.hard_cbf_enabled
                    or config.safety_slack_cap_px2 is not None
                    or config.tracking_cost_multiplier != 1.0
                    or tuple(config.safety_proxy_radii) != main_radii):
                raise ValueError("Model 1 requires configured radii, hard CBF and tracking1")
        config = self._regrasp_mind_filter.config
        if (config.hard_cbf_enabled
                or config.safety_slack_cap_px2 != 300.0
                or config.tracking_cost_multiplier != 40.0
                or tuple(config.safety_proxy_radii) != recovery_radii):
            raise ValueError("Model 2 requires configured radii, bounded-soft300 CBF and tracking40")
        print(f"[MIXED STAGE CBF] model1: radii={main_radii}, hard, slack=0, tracking1; "
              f"model2: radii={recovery_radii}, bounded soft, each slack<=300 px^2, tracking40", flush=True)
