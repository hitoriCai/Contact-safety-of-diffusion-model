"""
Usage:
python eval.py --checkpoint data/image/pusht/diffusion_policy_cnn/train_0/checkpoints/latest.ckpt -o data/pusht_eval_output
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from omegaconf import open_dict
from diffusion_policy.workspace.base_workspace import BaseWorkspace


SAFE_MIND_DEFAULTS = {
    'enabled': True,
    'checkpoint_path': 'diffusion_policy/data/dynamics_model_3layer_64.pth',
    'output_mode': 'delta',
    'clearance_scale': 1.0,
    # Asymmetric proxies: side-bar midpoints (1,3)/(2,3), and the lower
    # region centroid (0,4,3).
    'block_clearance_radius': 50.0,
    'center_keypoint_indices': [1, 2, 4],
    'safety_keypoint_groups': [[1, 3], [2, 3], [0, 4, 3]],
    'safety_proxy_radii': [35.0, 35.0, 50.0],
    'execution_action_steps': 4,
    'joint_mpc_enabled': False,
    'max_action_delta_px': 25.0,
    'parallel_workers': 8,
    'gamma': 0.001,
    'safe_alpha': 1.0,
    'recovery_alpha': 0.01,
    # Unsafe T states are nonworsening for every MIND stage. No terminal
    # action is forced to predict an outward T displacement.
    'recovery_terminal_step_only': False,
    'safe_slack_weight': 60.0,
    'recovery_slack_weight': 30.0,
    'terminal_slack_weight': 300.0,
    'tracking_cost_multiplier': 1.0,
    'control_dt_sec': 0.1,
    'action_search_radius': 1.0,
    # Compatibility-only: the active controller selects its solve path above.
    'mip_horizon': 1,
    'mip_gap': 0.0,
    'time_limit_sec': 5.0,
    'node_limit': 20000,
    'objective_norm': 'l1',
    'fail_safe': 'agent_hold',
    'print_status': True,
    'verbose': False,
}

DEFAULT_EVAL_N_ACTION_STEPS = 4
DEFAULT_SIDE_PROXY_RADIUS = 35.0
DEFAULT_LOWER_PROXY_RADIUS = 50.0
ORIGIN_POLICY_TARGET = (
    'diffusion_policy.policy.diffusion_unet_lowdim_policy_origin.'
    'OriginDiffusionUnetLowdimPolicy'
)
SAFE_POLICY_TARGET = (
    'diffusion_policy.policy.diffusion_unet_lowdim_policy.'
    'DiffusionUnetLowdimPolicy'
)
MODEL2_REGRASP_POLICY_TARGET = (
    'diffusion_policy.policy.diffusion_unet_lowdim_policy_regrasp_model2.'
    'Model2RegraspDiffusionPolicy'
)
# Specialized entry points may override this before invoking ``main``.
REGRASP_CORNER_MIND_ENABLED = True
MIND_JOINT_MPC_ENABLED = False
MIND_HARD_CBF_ENABLED = False

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option(
    '--mind_checkpoint',
    type=click.Path(exists=True, dir_okay=False),
    default=SAFE_MIND_DEFAULTS['checkpoint_path'], show_default=True,
    help=('Standalone MIND dynamics checkpoint: legacy ReLU NNDM or the '
          'v4.0 control-affine checkpoint.'),
)
@click.option('-o', '--output_dir', required=True)
@click.option(
    '--n_action_steps',
    default=DEFAULT_EVAL_N_ACTION_STEPS,
    show_default=True,
    type=click.IntRange(min=2),
    help='Number of diffusion action candidates generated per observation.',
)
@click.option(
    '--execution_action_steps',
    default=4,
    show_default=True,
    type=click.IntRange(min=1, max=16),
    help=('Number of leading candidates to project with MIND and execute '
          'before replanning.'),
)
@click.option(
    '--proxy_mode',
    type=click.Choice(['single_center', 'three_region']),
    default='three_region',
    show_default=True,
    help=('MIND safety geometry: the original single T-center circle or the '
          'newer three-region approximation.'),
)
@click.option(
    '--side_proxy_radius',
    default=DEFAULT_SIDE_PROXY_RADIUS,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help='Radius (px) for midpoint proxies [1,3] and [2,3].',
)
@click.option(
    '--lower_proxy_radius',
    default=DEFAULT_LOWER_PROXY_RADIUS,
    show_default=True,
    type=click.FloatRange(min=0.0),
    help='Radius (px) for lower-region centroid proxy [0,4,3].',
)
@click.option(
    '--block_clearance_radius',
    default=None,
    type=click.FloatRange(min=0.0),
    help='Legacy: override all three proxy radii with one value.',
)
@click.option(
    '--test_seed_file',
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    default=None,
    help='JSON list/object containing an explicit shared test seed set.',
)
@click.option(
    '--safety_stress_test',
    is_flag=True,
    help='Place one deterministic obstacle in each seed task corridor.',
)
@click.option(
    '--n_test_vis',
    type=click.IntRange(min=0),
    default=None,
    help='Number of test videos; defaults to the runner configuration.',
)
@click.option(
    '--max_steps', default=None, type=click.IntRange(min=1),
    help=('Optional total task rollout limit. Increase this when a late '
          '80--160-step model-2 recovery must remain fully recorded.'),
)
@click.option(
    '--score_extension_trigger_steps', default=None,
    type=click.IntRange(min=1),
    help=('At this task rollout step, stop if every seed has score >= '
          '--score_extension_threshold; otherwise continue to the extension cap.'),
)
@click.option(
    '--score_extension_max_steps', default=None,
    type=click.IntRange(min=1),
    help='Hard rollout cap used with --score_extension_trigger_steps.',
)
@click.option(
    '--score_extension_threshold', default=0.95, show_default=True,
    type=click.FloatRange(min=0.0),
    help='Accumulated score required for every seed to avoid extending rollout.',
)
@click.option(
    '--regrasp_model2_checkpoint',
    type=click.Path(exists=True, dir_okay=False),
    default=(
        'data/outputs/2026.09.06/diffusion_model_2_reannotated22_e500/'
        'checkpoints/last/epoch=0500.ckpt'),
    show_default=True,
    help='Escape diffusion checkpoint used only while regrasp is active.',
)
@click.option(
    '--regrasp_min_steps', default=0, show_default=True,
    type=click.IntRange(min=0),
    help='Minimum model-2 environment actions before returning to task diffusion.',
)
@click.option(
    '--regrasp_max_steps', default=160, show_default=True,
    type=click.IntRange(min=1),
    help='Maximum model-2 environment actions before forced task-policy return.',
)
@click.option(
    '--oscillation_regrasp/--no-oscillation_regrasp',
    default=False, show_default=True,
    help=('Experimental trigger for repeated large pusher-action reversals '
          'while the T remains stalled.'),
)
@click.option(
    '--timeout_rearm/--no-timeout_rearm',
    default=False, show_default=True,
    help=('Experimental one-time recovery retry after the ordinary rearm '
          'sequence remains incomplete for 120 actions.'),
)
@click.option(
    '--regrasp_corner_mind_recovery/--no-regrasp_corner_mind_recovery',
    default=True, show_default=True,
    help=('Enable nonworsening MIND recovery constraints for regrasp '
          'corner points already inside their safety margins.'),
)
@click.option(
    '--regrasp_mind/--no-regrasp_mind',
    default=REGRASP_CORNER_MIND_ENABLED, show_default=True,
    help=('Apply MIND during model-2 regrasp. Disabling this leaves the '
          'main-task MIND active but executes model-2 recovery actions raw.'),
)
@click.option(
    '--regrasp_side_proxy_radius', default=25.0, show_default=True,
    type=click.FloatRange(min=0.0),
    help='Radius (px) for each of the two side-region MIND proxies during regrasp.',
)
@click.option(
    '--regrasp_lower_proxy_radius', default=40.0, show_default=True,
    type=click.FloatRange(min=0.0),
    help='Radius (px) for the lower-region MIND proxy during regrasp.',
)
@click.option('-d', '--device', default='cuda:0')
@click.option(
    '--tracking_cost_multiplier', type=click.FloatRange(min=0.0, min_open=True),
    default=None,
    help='Multiply one-step diffusion tracking cost; default uses the experiment profile.',
)
def main(checkpoint, mind_checkpoint, output_dir, n_action_steps, execution_action_steps,
         proxy_mode, side_proxy_radius, lower_proxy_radius,
         block_clearance_radius,
         test_seed_file, safety_stress_test, n_test_vis,
         max_steps, score_extension_trigger_steps,
         score_extension_max_steps, score_extension_threshold,
         regrasp_model2_checkpoint, regrasp_min_steps,
         regrasp_max_steps, oscillation_regrasp, timeout_rearm,
         regrasp_corner_mind_recovery,
         regrasp_mind, regrasp_side_proxy_radius,
         regrasp_lower_proxy_radius, device, tracking_cost_multiplier):
    if MIND_JOINT_MPC_ENABLED and (
            n_action_steps != 4 or execution_action_steps != 4):
        raise click.UsageError(
            'The joint-MPC evaluation requires --n_action_steps 4 and '
            '--execution_action_steps 4.')
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if execution_action_steps > n_action_steps:
        raise click.UsageError(
            '--execution_action_steps must not exceed --n_action_steps.')
    if regrasp_min_steps > regrasp_max_steps:
        raise click.UsageError(
            '--regrasp_min_steps must not exceed --regrasp_max_steps.')
    if ((score_extension_trigger_steps is None)
            != (score_extension_max_steps is None)):
        raise click.UsageError(
            '--score_extension_trigger_steps and '
            '--score_extension_max_steps must be provided together.')
    if (score_extension_trigger_steps is not None
            and score_extension_max_steps < score_extension_trigger_steps):
        raise click.UsageError(
            '--score_extension_max_steps must be at least '
            '--score_extension_trigger_steps.')

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    # Keep the requested diffusion action window. MIND independently projects
    # the configured leading actions; the environment then executes them
    # before replanning.
    with open_dict(cfg):
        cfg.n_action_steps = n_action_steps
    with open_dict(cfg.policy):
        # Pure-origin training checkpoints use a policy adapter that forces
        # MIND off. Safe eval explicitly swaps back to the safety-capable
        # policy; both classes have the same registered model parameters.
        if cfg.policy._target_ == ORIGIN_POLICY_TARGET:
            cfg.policy._target_ = SAFE_POLICY_TARGET
        cfg.policy.n_action_steps = n_action_steps
        if SAFE_POLICY_TARGET == MODEL2_REGRASP_POLICY_TARGET:
            if regrasp_model2_checkpoint is None:
                raise click.UsageError(
                    '--regrasp_model2_checkpoint is required by '
                    'eval_regrasp_diffusion.py.')
            cfg.policy.regrasp_model2 = {
                'checkpoint_path': str(regrasp_model2_checkpoint),
                'min_steps': int(regrasp_min_steps),
                'max_steps': int(regrasp_max_steps),
                'escape_clearance_px': 100.0,
                'escape_displacement_px': 45.0,
                'escape_min_corner_distance_px': 150.0,
                'checkpoint_state': 'ema_model',
                'corner_mind_enabled': bool(regrasp_mind),
                'corner_mind_radius_px': 10.0,
                'region_mind_proxy_radii': [
                    float(regrasp_side_proxy_radius),
                    float(regrasp_side_proxy_radius),
                    float(regrasp_lower_proxy_radius),
                ],
                'corner_mind_recovery_enabled': bool(
                    regrasp_corner_mind_recovery),
                'oscillation_enabled': bool(oscillation_regrasp),
                'timeout_rearm_enabled': bool(timeout_rearm),
            }
    with open_dict(cfg.task.env_runner):
        cfg.task.env_runner.n_action_steps = execution_action_steps
        # A checkpoint trained through the origin adapter records this as false.
        # Safe evaluation must always pass actual obstacle data to MIND.
        cfg.task.env_runner.pass_obstacles_to_policy = True
        if test_seed_file is not None:
            seed_payload = json.loads(test_seed_file.read_text(encoding='utf-8'))
            test_seeds = (seed_payload['seeds'] if isinstance(seed_payload, dict)
                          else seed_payload)
            test_seeds = [int(seed) for seed in test_seeds]
            if len(test_seeds) == 0 or len(set(test_seeds)) != len(test_seeds):
                raise ValueError('Test seed file must contain unique seeds.')
            cfg.task.env_runner.n_train = 0
            cfg.task.env_runner.n_train_vis = 0
            cfg.task.env_runner.n_test = len(test_seeds)
            cfg.task.env_runner.test_seed_list = test_seeds
            # Version-2 stress manifests contain explicit obstacle geometry.
            # Propagate it to the runner so safe and origin evaluations use
            # exactly the same physical scenes rather than regenerating the
            # legacy seed-derived corridor obstacle.
            if isinstance(seed_payload, dict) and 'scenarios' in seed_payload:
                scenarios = seed_payload['scenarios']
                by_seed = {int(item['seed']): item for item in scenarios}
                if set(by_seed) != set(test_seeds):
                    raise ValueError(
                        'Scenario entries must match the explicit seed list.')
                cfg.task.env_runner.test_scenario_specs = []
                for seed in test_seeds:
                    item = by_seed[seed]
                    spec = {
                        'position': item['obstacle_position'],
                        'radius': item['obstacle_radius'],
                    }
                    # Goal pose is optional for backward compatibility with
                    # all existing fixed-goal seed manifests.
                    if 'goal_pose' in item:
                        spec['goal_pose'] = item['goal_pose']
                    cfg.task.env_runner.test_scenario_specs.append(spec)
        elif safety_stress_test:
            raise click.UsageError(
                '--safety_stress_test requires --test_seed_file.')
        cfg.task.env_runner.safety_stress_test = bool(safety_stress_test)
        if n_test_vis is not None:
            cfg.task.env_runner.n_test_vis = n_test_vis
        if max_steps is not None:
            cfg.task.env_runner.max_steps = int(max_steps)
        if score_extension_trigger_steps is not None:
            # The runner treats this as the normal budget, then raises its
            # physical wrapper cap to score_extension_max_steps only when a
            # trajectory remains below score_extension_threshold.
            cfg.task.env_runner.max_steps = int(score_extension_trigger_steps)
            cfg.task.env_runner.score_extension_trigger_steps = int(
                score_extension_trigger_steps)
            cfg.task.env_runner.score_extension_max_steps = int(
                score_extension_max_steps)
            cfg.task.env_runner.score_extension_threshold = float(
                score_extension_threshold)
    # Materialize one complete profile instead of inheriting stale safety
    # fields from a training checkpoint.
    with open_dict(cfg.policy):
        mind_config = dict(SAFE_MIND_DEFAULTS)
        mind_config['checkpoint_path'] = str(mind_checkpoint)
        if tracking_cost_multiplier is not None:
            mind_config['tracking_cost_multiplier'] = float(tracking_cost_multiplier)
        if proxy_mode == 'single_center':
            single_radius = (float(lower_proxy_radius)
                             if block_clearance_radius is None
                             else float(block_clearance_radius))
            mind_config['center_keypoint_indices'] = [0, 3]
            mind_config['safety_keypoint_groups'] = [[0, 3]]
            proxy_radii = [single_radius]
        elif block_clearance_radius is None:
            proxy_radii = [
                float(side_proxy_radius), float(side_proxy_radius),
                float(lower_proxy_radius)]
        else:
            proxy_radii = [float(block_clearance_radius)] * 3
        # This legacy scalar is only used by dormant horizon diagnostics;
        # active DCBF constraints use the three values above.
        mind_config['block_clearance_radius'] = max(proxy_radii)
        mind_config['safety_proxy_radii'] = proxy_radii
        mind_config['execution_action_steps'] = int(execution_action_steps)
        mind_config['joint_mpc_enabled'] = bool(MIND_JOINT_MPC_ENABLED)
        mind_config['hard_cbf_enabled'] = bool(MIND_HARD_CBF_ENABLED)
        mind_config['mip_horizon'] = (4 if MIND_JOINT_MPC_ENABLED else 1)
        cfg.policy.mind = mind_config
    if proxy_mode == 'single_center':
        proxy_description = (
            '- One T proxy center: midpoint of keypoints `[0,3]`.\n'
            f'- Proxy radius: `{proxy_radii[0]:g}`.\n')
    else:
        proxy_description = (
            '- Three T proxy centers: midpoint `[1,3]` (left bar), midpoint '
            '`[2,3]` (right bar), centroid `[0,4,3]` (lower region).\n'
            f'- Proxy radii: side `{proxy_radii[0]:g}`, '
            f'`{proxy_radii[1]:g}`; lower `{proxy_radii[2]:g}`.\n')
    if SAFE_POLICY_TARGET == MODEL2_REGRASP_POLICY_TARGET:
        regrasp_safety_description = (
            '- During model-2 recovery, MIND uses three-region '
            'geometry as the main task: `[1,3]`, `[2,3]`, and `[0,4,3]`, '
            'with recovery proxy radii `['
            f'{regrasp_side_proxy_radius:g}, {regrasp_side_proxy_radius:g}, '
            f'{regrasp_lower_proxy_radius:g}]`. ' + (
                'Its nonworsening recovery branch is enabled for regions '
                'inside their safety margins. '
                if regrasp_corner_mind_recovery else
                'Its recovery branch is disabled: regions already inside '
                'their safety margins are not modified. '
            ) + f'The main task retains proxy radii `{proxy_radii}` and '
            'recovery logic.\n'
            if regrasp_mind else
            '- During model-2 recovery, MIND is disabled; model-2 actions '
            'are executed raw, and the main-task MIND resumes only after '
            'recovery exits.\n')
        regrasp_description = (
            '- Regrasp controller: obstacle-aware diffusion model 2.\n'
            f'- Model 2 checkpoint: `{regrasp_model2_checkpoint}`.\n'
            '- Model 2 executes four actions per observation and '
            'returns at the next replan when the escape condition succeeds. '
            f'It runs at most `{regrasp_max_steps}` actions and can trigger '
            'again after the normal leave/recontact/push rearm sequence.\n'
            f'- Experimental oscillation trigger: `{bool(oscillation_regrasp)}`.\n'
            f'- Experimental one-time timeout retry: `{bool(timeout_rearm)}`.\n'
            f'{regrasp_safety_description}'
            '- Successful return requires exact T-obstacle '
            'clearance >= 100 px, T-center displacement >= 45 px, and '
            'T-center distance to every page corner > 150 px. '
            'The oscillation trigger has no pusher-to-T distance gate.\n')
    else:
        regrasp_description = ''
    rollout_extension_description = (
        '- Rollout duration: stop at `'
        f'{score_extension_trigger_steps}` steps only when every seed has '
        f'accumulated score >= `{score_extension_threshold:g}`; otherwise '
        f'continue to at most `{score_extension_max_steps}` steps.\n'
        if score_extension_trigger_steps is not None else
        f'- Rollout duration cap: `{cfg.task.env_runner.max_steps}` steps.\n')
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    if policy.mind_filter is None:
        raise RuntimeError('Safe evaluation unexpectedly has no MIND filter.')

    if MIND_JOINT_MPC_ENABLED:
        mind_solve_description = (
            f'- Joint MIND/MPC solve over `{execution_action_steps}` diffusion '
            'actions; safety slack is optimized before diffusion tracking.\n')
        mind_prediction_description = (
            '- The four actions and NNDM-predicted T states are optimized '
            'jointly.\n')
    else:
        mind_solve_description = (
            '- Independent one-step MIND solves per diffusion sample: '
            f'`{execution_action_steps}`\n')
        mind_prediction_description = (
            '- Each solve uses the preceding projected action and '
            'NNDM-predicted T state as its starting point.\n')
    safety_description = (
        '- Every main-task DCBF slack has hard upper bound zero, both '
        'outside and inside the proxy safety boundary.\n'
        if MIND_HARD_CBF_ENABLED else
        '- Safe and already-unsafe states both use soft DCBF constraints.\n')
    fallback_description = (
        '- If any one-step MIND solve has no incumbent, all four actions '
        'target the measured pusher position (zero displacement); the video '
        'shows `INFEASIBLE`. This also covers time limits without an '
        'incumbent, not only proven mathematical infeasibility.\n'
        if MIND_HARD_CBF_ENABLED else
        '- If no incumbent exists, the pusher targets its current position '
        'and a `NO_SOLUTION` diagnostic line is printed.\n')

    pathlib.Path(output_dir, 'README.md').write_text(
        '# MIND safe evaluation\n\n'
        f'- Checkpoint: `{checkpoint}`\n'
        f'- Diffusion action candidates per observation: `{n_action_steps}`\n'
        f'{mind_solve_description}'
        f'- MIND dynamics checkpoint: `{mind_checkpoint}`.\n'
        f'{mind_prediction_description}'
        f'- Projected actions executed before replanning: `{execution_action_steps}`\n'
        '- Hard per-axis action-rate constraint: `|u[k]-anchor[k]| <= 25 px`; '
        'action 1 is anchored at the measured pusher position and actions '
        '2 onward at the preceding projected action.\n'
        '- Policy receives obstacle positions/radii.\n'
        f'{safety_description}'
        f'{proxy_description}'
        f'{regrasp_description}'
        f'{rollout_extension_description}'
        f'- DCBF uses `safe_alpha={mind_config["safe_alpha"]:g}`, '
        f'`recovery_alpha={mind_config["recovery_alpha"]:g}`.\n'
        '- Multi-action recovery timing: all stages are nonworsening by '
        'default; strict terminal recovery is optional '
        f'(`recovery_terminal_step_only='
        f'{mind_config.get("recovery_terminal_step_only", False)}).\n'
        '- Early recovery margins: '
        f'enter `{mind_config.get("recovery_trigger_margin_px", 0.0):g}px`, '
        f'exit `{mind_config.get("recovery_release_margin_px", 0.0):g}px`.\n'
        '- Normalized slack weights: '
        f'safe `{mind_config["safe_slack_weight"]:g}`, '
        f'recovery `{mind_config["recovery_slack_weight"]:g}`.\n'
        f'- Diffusion tracking cost multiplier: '
        f'`{mind_config["tracking_cost_multiplier"]:g}`.\n'
        f'{fallback_description}'
        '- MIND parameters: `clearance_scale=1.0`.\n'
        f'- Explicit test seed file: `{test_seed_file}`.\n'
        f'- Safety-stress corridor obstacles: `{safety_stress_test}`.\n',
        encoding='utf-8')
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log = env_runner.run(policy)
    for key, value in policy.mind_filter.diagnostics_summary().items():
        runner_log[f'mind/{key}'] = value
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
