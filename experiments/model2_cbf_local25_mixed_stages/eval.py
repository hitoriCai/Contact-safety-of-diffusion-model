"""Evaluate hard model 1 and bounded-soft model 2 with independent radii."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import eval as base_eval
from omegaconf import open_dict
from experiments.contact_gap_5s import runner_contact_gap as core_runner
from experiments.nndm_dim24_model2_v63_eval import runner_core_stop10
from experiments.model2_cbf_full_page.filter import install
from experiments.model2_cbf_full_page import policy as full_page_policy
from experiments.model2_cbf_diverse_hard_infeasible_regrasp import runner as runner_module
from experiments.model2_cbf_local25_mixed_stages.policy import MixedStageCBFPolicy
from experiments.model2_cbf_local25_mixed_stages.runner import PushTKeypointsRunner

install(local_only=True)
core_runner.PushTKeypointsRunner.run = runner_core_stop10.PushTKeypointsRunner.run
full_page_policy.FullPagePolicy = MixedStageCBFPolicy
runner_module.PushTKeypointsRunner = PushTKeypointsRunner

TARGET = 'experiments.model2_cbf_local25_mixed_stages.policy.MixedStageCBFPolicy'
RUNNER_TARGET = 'experiments.model2_cbf_local25_mixed_stages.runner.PushTKeypointsRunner'
base_eval.SAFE_POLICY_TARGET = TARGET
base_eval.MODEL2_REGRASP_POLICY_TARGET = TARGET
base_eval.REGRASP_CORNER_MIND_ENABLED = True
base_eval.MIND_HARD_CBF_ENABLED = True
base_eval.SAFE_MIND_DEFAULTS['safety_slack_cap_px2'] = None
base_eval.SAFE_MIND_DEFAULTS['tracking_cost_multiplier'] = 1.0
original_instantiate = base_eval.hydra.utils.instantiate


def instantiate_with_runner(config, *args, **kwargs):
    if str(config.get('_target_', '')).endswith('PushTKeypointsRunner'):
        with open_dict(config):
            config._target_ = RUNNER_TARGET
    return original_instantiate(config, *args, **kwargs)


base_eval.hydra.utils.instantiate = instantiate_with_runner

if __name__ == '__main__':
    base_eval.main()
