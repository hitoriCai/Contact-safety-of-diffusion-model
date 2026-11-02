"""Persist effective per-stage profiles alongside the normal route traces."""

from dataclasses import asdict
import json
from pathlib import Path
from omegaconf import OmegaConf

from experiments.model2_cbf_local25_route_memory.runner import PushTKeypointsRunner as BaseRunner


class PushTKeypointsRunner(BaseRunner):
    def run(self, policy):
        def plain(value):
            if OmegaConf.is_config(value):
                value = OmegaConf.to_container(value, resolve=True)
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value

        profiles = plain({
            'model1': asdict(policy.mind_filter.config),
            'model2': asdict(policy._regrasp_mind_filter.config),
        })
        output = Path(self.output_dir)
        (output / 'effective_cbf_profiles.json').write_text(
            json.dumps(profiles, indent=2) + '\n')
        readme = output / 'README.md'
        text = readme.read_text() if readme.exists() else ''
        # The shared eval README contains historical four-action descriptions.
        # Put the actual profiles first so the run's configuration is explicit.
        readme.write_text(
            '# Effective mixed-stage experiment configuration\n\n'
            f"- Model 1: radii {profiles['model1']['safety_proxy_radii']} px, hard CBF, zero safety slack, tracking x1.\n"
            f"- Model 2: radii {profiles['model2']['safety_proxy_radii']} px, soft CBF, each slack <=300 px^2, tracking x40.\n"
            '- Centers: mean([1,3]), mean([2,3]), mean([0,4,7]).\n'
            '- Both stages solve directly within +/-25 px per axis and execute one action.\n'
            '- NNDM_dim24; model2 v6.3 epoch500; K12, lazy Frechet50 complete-link/medoid grouping and route memory.\n'
            '- Model-1 infeasibility immediately triggers regrasp; 10 consecutive model-2 infeasible actions stop the scene.\n'
            '- These effective settings supersede historical descriptions below.\n\n'
            + text)
        result = super().run(policy)
        if result.get('mind/max_safety_slack', 0.) > 1.e-6:
            raise RuntimeError("Model-1 hard CBF produced positive safety slack")
        result['stage_cbf/model1'] = profiles['model1']
        result['stage_cbf/model2'] = profiles['model2']
        return result
