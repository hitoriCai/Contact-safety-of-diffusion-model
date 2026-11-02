if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
# import wandb
import tqdm
import shutil
import re

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    @staticmethod
    def _prune_last_checkpoints(checkpoint_dir, keep_last):
        numbered = []
        for path in checkpoint_dir.glob('epoch=*.ckpt'):
            match = re.match(r'^epoch=(\d+)', path.name)
            if match is not None:
                numbered.append((int(match.group(1)), path))
        numbered.sort(key=lambda item: item[0])
        for _, old_path in numbered[:-keep_last]:
            old_path.unlink()

    @staticmethod
    def _prune_topk_checkpoints(checkpoint_dir, keep_topk):
        scored = []
        for path in checkpoint_dir.glob('*.ckpt'):
            match = re.search(
                r'test_mean_score=([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',
                path.name)
            if match is not None:
                scored.append((float(match.group(1)), path))
        scored.sort(key=lambda item: item[0], reverse=True)
        for _, old_path in scored[keep_topk:]:
            old_path.unlink()

    @staticmethod
    def _run_rollout(env_runner, policy, rollout_with_mind):
        """Run diagnostic rollouts without turning MIP into a training input.

        Baseline training keeps the exact origin objective and uses raw
        Diffusion actions for periodic scores.  The safety filter is restored
        immediately afterwards, so the resulting checkpoint remains directly
        usable by eval.py's safe rollout.
        """
        if rollout_with_mind or not hasattr(policy, 'mind_filter'):
            return env_runner.run(policy)
        mind_filter = policy.mind_filter
        policy.mind_filter = None
        try:
            return env_runner.run(policy)
        finally:
            policy.mind_filter = mind_filter

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        '''
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            # mode="offline",  # add
            settings=wandb.Settings(console="off"), # 核心：关闭控制台监控 add
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
        '''

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # A baseline YAML intentionally has no safety-only fields.  Default to
        # raw-policy rollout scores; safety is applied later by eval.py.
        rollout_with_mind = bool(cfg.training.get('rollout_with_mind', False))

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            # num_epochs is the absolute target epoch. On resume, train only
            # the remaining epochs instead of running num_epochs more epochs.
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                completed_epoch = self.epoch + 1
                should_run_rollout = (
                    completed_epoch % cfg.training.rollout_every) == 0
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            # wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if should_run_rollout:
                    runner_log = self._run_rollout(
                        env_runner, policy, rollout_with_mind)
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = train_sampling_batch
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        
                        # Training diagnostics use the origin policy API and
                        # never pass obstacle/safety inputs.
                        result = policy.predict_action(obs_dict)
                        if cfg.pred_action_steps_only:  # 默认是false
                            pred_action = result['action']
                            start = cfg.n_obs_steps - 1
                            end = start + cfg.n_action_steps
                            gt_action = gt_action[:,start:end]
                        else:
                            pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        # log
                        step_log['train_action_mse_error'] = mse.item()

                        # =========================================================
                        # [新增功能] 保存 npz 文件到 output_dir/npz_data/
                        # =========================================================
                        if (self.epoch % 100) == 0:
                            npz_dir = os.path.join(self.output_dir, 'npz_data')
                            os.makedirs(npz_dir, exist_ok=True)
                            save_path = os.path.join(npz_dir, f'epoch_{self.epoch}.npz')

                            # 3. 保存文件 (注意：必须使用 .detach().cpu().numpy() 将 Tensor 转为 Numpy)
                            np.savez_compressed(
                                save_path,
                                gt_action = gt_action.detach().cpu().numpy(),
                                pred_action = pred_action.detach().cpu().numpy(),
                                train_action_mse_error = mse.item(), 
                            )
                        # =========================================================


                        # release RAM
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                # wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

                # Always retain the newest checkpoints for resuming, and when
                # requested retain a separate score-ranked top-k set.
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(use_thread=False)

                    last_dir = checkpoint_dir.joinpath('last')
                    last_dir.mkdir(parents=True, exist_ok=True)
                    last_path = last_dir.joinpath(f'epoch={self.epoch:04d}.ckpt')
                    self.save_checkpoint(path=last_path, use_thread=False)
                    keep_last = int(cfg.checkpoint.get(
                        'keep_last', cfg.checkpoint.get('last_k', 5)))
                    self._prune_last_checkpoints(last_dir, keep_last)

                    topk_cfg = cfg.checkpoint.get('topk', None)
                    if topk_cfg is not None:
                        monitor_key = str(topk_cfg.monitor_key)
                        score = step_log.get(monitor_key)
                        # The historical origin config spells this key without
                        # a slash, while the current runner logs test/mean_score.
                        if score is None and monitor_key == 'test_mean_score':
                            score = step_log.get('test/mean_score')
                        if score is None:
                            print(
                                f'Skipping top-k checkpoint: missing {monitor_key}',
                                flush=True)
                        else:
                            top_dir = checkpoint_dir.joinpath('topk')
                            top_dir.mkdir(parents=True, exist_ok=True)
                            top_path = top_dir.joinpath(
                                topk_cfg.format_str.format(
                                    epoch=self.epoch,
                                    test_mean_score=float(score)))
                            self.save_checkpoint(path=top_path, use_thread=False)
                            self._prune_topk_checkpoints(
                                top_dir, int(topk_cfg.k))

                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
