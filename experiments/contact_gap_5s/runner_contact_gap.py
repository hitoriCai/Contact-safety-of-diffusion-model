import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import json
import os
from experiments.contact_gap_5s.contact_env import ContactPushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from experiments.contact_gap_5s.contact_wrapper import ContactMultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
import matplotlib.pyplot as plt

class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_seed_list=None,
            test_scenario_specs=None,
            safety_stress_test=False,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            score_extension_trigger_steps=None,
            score_extension_max_steps=None,
            score_extension_threshold=0.95,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            pass_obstacles_to_policy=True,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        base_max_steps = int(max_steps)
        if base_max_steps < 1:
            raise ValueError('max_steps must be positive.')
        if score_extension_trigger_steps is None:
            if score_extension_max_steps is not None:
                raise ValueError(
                    'score_extension_max_steps requires '
                    'score_extension_trigger_steps.')
            self.score_extension_trigger_steps = None
            self.score_extension_max_steps = None
            self.score_extension_threshold = None
            self.max_steps = base_max_steps
        else:
            trigger_steps = int(score_extension_trigger_steps)
            if score_extension_max_steps is None:
                raise ValueError(
                    'score_extension_trigger_steps requires '
                    'score_extension_max_steps.')
            extension_max_steps = int(score_extension_max_steps)
            if not 1 <= trigger_steps <= extension_max_steps:
                raise ValueError(
                    'score extension requires '
                    '1 <= trigger_steps <= extension_max_steps.')
            self.score_extension_trigger_steps = trigger_steps
            self.score_extension_max_steps = extension_max_steps
            self.score_extension_threshold = float(score_extension_threshold)
            # The wrapper must permit the extended interval. The runner
            # explicitly ends at trigger_steps when every rollout has already
            # reached the requested score threshold.
            self.max_steps = extension_max_steps

        if test_seed_list is not None:
            test_seed_list = [int(seed) for seed in test_seed_list]
            if len(test_seed_list) == 0:
                raise ValueError('test_seed_list cannot be empty.')
            n_test = len(test_seed_list)
        if test_scenario_specs is not None:
            test_scenario_specs = list(test_scenario_specs)
            if test_seed_list is None:
                raise ValueError(
                    'test_scenario_specs requires an explicit test_seed_list.')
            if len(test_scenario_specs) != len(test_seed_list):
                raise ValueError(
                    'test_scenario_specs must align one-to-one with test seeds.')
        if n_envs is None:
            n_envs = n_train + n_test
        self.unsafe_count = 0

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = ContactPushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return ContactMultiStepWrapper(
                VideoRecordingWrapper(
                    ContactPushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=self.max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                env.env.env.safety_stress_obstacles = False
                env.env.env.safety_stress_obstacle_spec = None
                env.env.env.safety_stress_goal_pose = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', f'train_{seed}.mp4')
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = (test_seed_list[i] if test_seed_list is not None
                    else test_start_seed + i)
            enable_render = i < n_test_vis
            scenario_spec = (
                None if test_scenario_specs is None
                else test_scenario_specs[i])

            def init_fn(env, seed=seed, enable_render=enable_render,
                        safety_stress_test=safety_stress_test,
                        scenario_spec=scenario_spec):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                env.env.env.safety_stress_obstacles = bool(
                    safety_stress_test)
                env.env.env.safety_stress_obstacle_spec = scenario_spec
                env.env.env.safety_stress_goal_pose = (
                    None if scenario_spec is None
                    else scenario_spec.get('goal_pose'))
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', f'test_{seed}.mp4')
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.pass_obstacles_to_policy = pass_obstacles_to_policy
        self.test_seed_list = test_seed_list
        self.test_scenario_specs = test_scenario_specs
        self.safety_stress_test = bool(safety_stress_test)
        self.tqdm_interval_sec = tqdm_interval_sec
    
    @staticmethod
    def _atomic_json_dump(payload, path):
        """Write a JSON snapshot atomically so interruption cannot corrupt it."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _write_partial_results(
            self, all_rewards, all_infos, completed_count, total_count,
            chunk_idx, n_chunks, policy):
        """Persist cumulative metrics after every completed rollout chunk."""
        payload = {
            'partial/completed_count': int(completed_count),
            'partial/total_count': int(total_count),
            'partial/completed_chunks': int(chunk_idx + 1),
            'partial/total_chunks': int(n_chunks),
            'partial/is_final': bool(completed_count == total_count),
        }
        scores = collections.defaultdict(list)
        collision_counts = collections.defaultdict(int)
        success_counts = collections.defaultdict(int)
        safe_success_counts = collections.defaultdict(int)
        chunk_payload = dict(payload)

        for index in range(completed_count):
            if all_rewards[index] is None or all_infos[index] is None:
                continue
            seed = int(self.env_seeds[index])
            prefix = self.env_prefixs[index]
            max_reward = float(np.max(all_rewards[index]))
            collision = bool(all_infos[index]['is_collision'][-1])
            success = bool(max_reward >= 1.0)
            payload[prefix + f'sim_max_reward_{seed}'] = max_reward
            payload[prefix + f'sim_collision_{seed}'] = collision
            scores[prefix].append(max_reward)
            collision_counts[prefix] += int(collision)
            success_counts[prefix] += int(success)
            safe_success_counts[prefix] += int(success and not collision)

        for prefix, values in scores.items():
            count = len(values)
            payload[prefix + 'mean_score'] = float(np.mean(values))
            payload[prefix + 'success_count'] = int(success_counts[prefix])
            payload[prefix + 'success_rate'] = success_counts[prefix] / count
            payload[prefix + 'collision_count'] = int(collision_counts[prefix])
            payload[prefix + 'collision_rate'] = collision_counts[prefix] / count
            payload[prefix + 'safe_success_count'] = int(
                safe_success_counts[prefix])
            payload[prefix + 'safe_success_rate'] = (
                safe_success_counts[prefix] / count)

        if getattr(policy, 'mind_filter', None) is not None:
            for key, value in policy.mind_filter.diagnostics_summary().items():
                payload[f'mind/{key}'] = value

        chunk_start = chunk_idx * len(self.env_fns)
        for index in range(chunk_start, completed_count):
            seed = int(self.env_seeds[index])
            prefix = self.env_prefixs[index]
            chunk_payload[prefix + f'sim_max_reward_{seed}'] = float(
                np.max(all_rewards[index]))
            chunk_payload[prefix + f'sim_collision_{seed}'] = bool(
                all_infos[index]['is_collision'][-1])

        output_dir = pathlib.Path(self.output_dir)
        self._atomic_json_dump(
            payload, output_dir / 'partial_eval_log.json')
        self._atomic_json_dump(
            chunk_payload,
            output_dir / 'partial_chunks' / f'chunk_{chunk_idx:03d}.json')

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_infos = [None] * n_inits
        all_actions = [None] * n_inits
        mind_step_diagnostics = []

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            actions = []
            executed_steps = 0


            obstacles_dict = None
            if self.pass_obstacles_to_policy:
                # Keep obstacle data aligned with every vectorized environment.
                # MIND solves an independent MILP for each batch member.
                obstacles = env.call('get_obstacle_message')[this_local_slice]
                obstacles_dict = {
                    'pos': [item[0] for item in obstacles],
                    'r': [item[1] for item in obstacles],
                }





            while not done:
                # Regrasp should never replace an already high-scoring task
                # rollout.  Supply the current per-environment maximum reward
                # before each policy replan.
                if hasattr(policy, 'set_task_scores'):
                    reward_histories = env.call('get_attr', 'reward')
                    policy.set_task_scores([
                        float(np.max(history)) if len(history) else 0.0
                        for history in reward_histories[this_local_slice]
                    ])
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # Exact number of executed 0.1-second control steps since
                # physical pusher-to-T contact, measured inside each env.
                if hasattr(policy, 'set_no_contact_steps'):
                    policy.set_no_contact_steps(
                        env.call('get_no_contact_steps'))

                # run policy
                with torch.no_grad():
                    if self.pass_obstacles_to_policy:
                        action_dict = policy.predict_action(obs_dict, obstacles_dict)
                    else:
                        action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # Mirror-regrasp policies expose one boolean per environment.
                # VideoRecordingWrapper renders all four actions in this chunk
                # with the same visible state marker.
                regrasp_active = np_action_dict.get('regrasp_active')
                regrasp_force_continue = np_action_dict.get(
                    'regrasp_force_continue')
                regrasp_model2_active = np_action_dict.get(
                    'regrasp_model2_active')
                regrasp_target = np_action_dict.get('regrasp_target')
                env.call_each(
                    'set_regrasp_active',
                    args_list=[(
                        False if regrasp_active is None
                        else bool(regrasp_active[env_index]),
                    ) for env_index in range(n_envs)])
                env.call_each(
                    'set_regrasp_label',
                    args_list=[(
                        'REGRASP MODEL 2: ON'
                        if regrasp_model2_active is not None
                        and bool(regrasp_model2_active[env_index])
                        else None,
                    ) for env_index in range(n_envs)])
                env.call_each(
                    'set_regrasp_target',
                    args_list=[(
                        None if regrasp_target is None
                        else regrasp_target[env_index, :2],
                    ) for env_index in range(n_envs)])

                # Show the direct diffusion sample, before MIND changes the
                # executed first action, in the recorded diagnostic video.
                raw_plan = np_action_dict.get('diffusion_action_pred')
                if raw_plan is not None:
                    raw_plan_start = (
                        self.n_obs_steps - 1 if getattr(
                            policy, 'oa_step_convention', False)
                        else self.n_obs_steps)
                    raw_plan = raw_plan[:, raw_plan_start:, :2]
                    env.call_each(
                        'set_diffusion_trajectory',
                        args_list=[(raw_plan[env_index],)
                                   for env_index in range(n_envs)])

                pusher_positions = np_obs_dict['obs'][:, -1, 18:20]

                # Draw the complete MIND-projected execution chunk, rather
                # than only its first action (which is sent to the env below).
                # This makes the four sequential MIP corrections inspectable
                # in the diagnostic video.
                mind_chunk = np_action_dict.get('mind_projected_action_chunk')
                mind_projected_mask = np_action_dict.get(
                    'mind_projected_mask')
                env.call_each(
                    'set_mind_optimized_trajectory',
                    args_list=[(
                        None if mind_chunk is None
                        or (mind_projected_mask is not None
                            and not bool(mind_projected_mask[env_index]))
                        else mind_chunk[env_index, :, :2],
                        pusher_positions[env_index],
                    ) for env_index in range(n_envs)])

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]

                predicted_centers = np_action_dict.get('mind_predicted_center')
                env.call_each(
                    'set_mind_step_overlay',
                    args_list=[(
                        action[env_index, 0, :2],
                        None if predicted_centers is None
                        else predicted_centers[env_index, :2],
                    ) for env_index in range(n_envs)])

                # Send each MIND result to its own environment before the frame
                # is rendered by VideoRecordingWrapper.step().
                diagnostic_data = [None] * n_envs
                mind_is_active = np.ones(n_envs, dtype=np.bool_)
                if mind_projected_mask is not None:
                    mind_is_active = np.asarray(
                        mind_projected_mask, dtype=np.bool_).reshape(-1)
                if (self.pass_obstacles_to_policy
                        and hasattr(policy, 'mind_config')
                        and policy.mind_config.enabled):
                    infeasible = np_action_dict['mind_infeasible'].reshape(-1)
                    env.call_each(
                        'set_mind_visualization',
                        args_list=[(
                            (policy.mind_config.block_clearance_radius
                             if mind_is_active[env_index] else None),
                            policy.mind_config.clearance_scale,
                            policy.mind_config.center_keypoint_indices,
                            (bool(infeasible[env_index])
                             if mind_is_active[env_index] else False),
                            policy.mind_config.safety_keypoint_groups,
                            policy.mind_config.safety_proxy_radii,
                        ) for env_index in range(n_envs)])
                    diagnostic_data = [
                        (policy.mind_visualization_data(env_index)
                         if mind_is_active[env_index] else None)
                        for env_index in range(n_envs)]
                    env.call_each(
                        'set_mind_diagnostic_data',
                        args_list=[(diagnostic_data[env_index],)
                                   for env_index in range(n_envs)])
                control_step = len(actions)
                # Calling PyGame rendering through AsyncVectorEnv can block.
                # Save numbers here; diagnostic stills are composed from the
                # completed video in the parent process after the rollout.
                for env_index, data in enumerate(diagnostic_data):
                    if data is not None:
                        mind_step_diagnostics.append({
                            'seed': int(self.env_seeds[start + env_index]),
                            'control_step': int(control_step),
                            **data,
                        })

                # step env
                obs, reward, done, info = env.step(action)
                # A model-2 regrasp has an explicit action contract. Do not
                # let inherited task success truncate an active recovery chunk
                # before the policy returns control to the task diffusion.
                # The configured episode budget remains a hard cap, however:
                # otherwise a newly retriggered regrasp can override the
                # wrapper's terminal signal indefinitely after that cap.
                reaches_episode_cap = (
                    executed_steps + int(action.shape[1]) >= self.max_steps)
                if (regrasp_force_continue is not None
                        and not reaches_episode_cap):
                    active_mask = np.asarray(
                        regrasp_force_continue,
                        dtype=np.bool_).reshape(-1)
                    done = np.logical_and(
                        np.asarray(done, dtype=np.bool_),
                        np.logical_not(active_mask))
                done = np.all(done)
                past_action = action
                actions.append(action)
                executed_steps += int(action.shape[1])

                # Run the normal 30-second budget first. At that point,
                # continue only if at least one parallel rollout has not
                # achieved the requested accumulated task score. The wrapper
                # itself is capped at score_extension_max_steps.
                if (not done
                        and self.score_extension_trigger_steps is not None
                        and executed_steps >= self.score_extension_trigger_steps):
                    reward_histories = env.call('get_attr', 'reward')
                    active_scores = np.asarray([
                        (float(np.max(history)) if len(history) else 0.0)
                        for history in reward_histories[this_local_slice]
                    ], dtype=np.float64)
                    if np.all(active_scores >= self.score_extension_threshold):
                        done = True

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_actions[this_global_slice] = actions
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            all_infos[this_global_slice] = env.call('get_infos')[this_local_slice]
            self._write_partial_results(
                all_rewards, all_infos, end, n_inits,
                chunk_idx, n_chunks, policy)
            
            # Shape: (policy_steps, envs, action_steps, action_dim).
            actions = np.asarray(actions)
            # 创建一个新的图形
            fig, ax = plt.subplots()

            # Plot the first active environment over the full rollout.
            first_env_actions = actions[:, 0, :, :2].reshape(-1, 2)
            ax.plot(
                first_env_actions[:, 0], first_env_actions[:, 1], marker='o')

            # 绘制圆
            obs_pos = all_infos[this_global_slice][0]['obstacle_positions']
            obs_r = all_infos[this_global_slice][0]['obstacle_radius']
            for i in range(len(obs_pos)):
                center = (obs_pos[i][0,0],obs_pos[i][0,1])
                # print(obs_pos[i][0,0],obs_pos[i][0,1],obs_r[i][0])
                circle = plt.Circle(center, obs_r[i][0], fill=False, color='blue')

                # 将圆添加到图形中
                ax.add_artist(circle)

            # 显示图形
            fig.savefig(pathlib.Path(self.output_dir).joinpath(
                f'action_trajectory_chunk_{chunk_idx}.png'))
            plt.close(fig)
        # clear out video buffer
        _ = env.reset()

        if mind_step_diagnostics:
            self._atomic_json_dump(
                mind_step_diagnostics,
                pathlib.Path(self.output_dir) / 'mind_step_diagnostics.json')

        # import pdb; pdb.set_trace()

        # log
        max_rewards = collections.defaultdict(list)
        collision_counts = collections.defaultdict(int)
        success_counts = collections.defaultdict(int)
        safe_success_counts = collections.defaultdict(int)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            collision = bool(all_infos[i]['is_collision'][-1])
            success = bool(max_reward >= 1.0)
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            log_data[prefix+f'sim_rollout_steps_{seed}'] = int(
                len(all_rewards[i]))
            log_data[prefix+f'sim_collision_{seed}'] = collision
            collision_counts[prefix] += int(collision)
            success_counts[prefix] += int(success)
            safe_success_counts[prefix] += int(success and not collision)
            if collision:
                self.unsafe_count += 1

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                if collision:
                    log_data[prefix+f'unsafe_sim_video_{seed}'] = sim_video
                else:
                    log_data[prefix+f'sim_video_{seed}'] = sim_video

        log_data['unsafe / total'] = f'{self.unsafe_count} / {n_inits}'  # train和test所有加起来
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            count = len(value)
            log_data[prefix+'mean_score'] = float(np.mean(value))
            log_data[prefix+'success_count'] = success_counts[prefix]
            log_data[prefix+'success_rate'] = success_counts[prefix] / count
            log_data[prefix+'collision_count'] = collision_counts[prefix]
            log_data[prefix+'collision_rate'] = collision_counts[prefix] / count
            log_data[prefix+'safe_success_count'] = safe_success_counts[prefix]
            log_data[prefix+'safe_success_rate'] = (
                safe_success_counts[prefix] / count)

        return log_data


    @staticmethod
    def no_collision_dynamic_obstacle(previous_position: np.ndarray, current_position: np.ndarray) -> bool:
        '''
        input:
            previous_position: np.ndarray, shape (num_obstacles, 2)
            current_position: np.ndarray, shape (num_obstacles, 2)
        '''
        distance = np.linalg.norm(current_position - previous_position, axis=1)
        collision_threshold = 0.01  
        return np.all(distance < collision_threshold)
    
