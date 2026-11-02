# Contact safety of diffusion model

本仓库保存一次固定66场景的PushT评测：model1执行任务推送，MIND/CBF修正动作；卡住时由diffusion model2执行regrasp。
它对应远端归档实验 **`10.31_59px+有界软约束`**。

## 本轮采用的控制配置

| 项目 | model1 | regrasp / model2 |
| --- | --- | --- |
| Diffusion checkpoint | 第600轮 | v6.3，第500轮 |
| 三圆半径 | 57/57/57px | 59/59/59px |
| CBF | 硬约束，安全松弛为0 | 每个圆/障碍物约束的松弛≤300px² |
| 跟踪代价倍率 | 1 | 40 |

- NNDM：24维输入，原206条与model2的370条示范联合训练的动力学模型。
- 圆心：关键点`[1,3]`、`[2,3]`、`[0,4,7]`的平均位置。
- 每次直接在实测推手位置的每轴±25px内求解，执行1步后重新规划。
- model2每批采样12条轨迹，保留当前路线；失效/停滞时按离散Frechet距离50px、complete-link聚类、medoid代表路线尝试替代解。
- model1无解时立即进入regrasp；regrasp连续10次无解结束场景。
- 保留原有小动作、震荡、超时重试和连续5秒未接触触发。
- 常规300步；score不足0.95时最多延长至700步。

**本轮没有启用后来试验的“单步侵入2px、最大侵入5px”控制器。**
部分依赖目录仍叫`full_page`或`global_then_clip`，是历史继承/导入路径；它们不表示本轮在全屏求解后裁切。
300px²是CBF不等式的松弛上限，不是真实物体始终最多侵入某个固定像素数的保证。

## 记录结果

| 指标 | 固定66场景 |
| --- | ---: |
| 平均score | 0.8014207590 |
| 碰撞样例 | 8/66（12.12%） |
| 严格成功 | 5/66 |
| 安全成功 | 5/66 |
| score≥0.85 | 52/66 |
| score≥0.95 | 36/66 |
| 0分样例 | 6 |

其中10条为已核验相同参数的早先结果，另外56条重新评测。详见`experiment/reuse_audit.json`和`experiment/comparison_vs_previous.json`。
严格成功与`score≥0.95`分开统计。

## 文件结构

```text
checkpoints/        原始model1、model2、NNDM权重
diffusion_policy/   模型、动力学安全过滤器、物理环境、workspace等依赖
experiments/        本轮控制器入口及继承链依赖
scenarios/          本轮66份固定初始场景manifest
experiment/         原始冻结配置、结果汇总和复用核验记录
source_inventory/   原文件清单、大小、SHA256和环境快照
scripts/run_eval.py 可移植运行封装
```

清单中的138个原始源文件/模型/场景/记录按原字节保存。
本次上传新增README、Git LFS规则及可移植运行封装；核心控制代码和原始checkpoint没有改写。
`experiment/experiment_config.json`保留原服务器路径以供溯源，运行时由`scripts/run_eval.py`替换为本仓库的路径。
`scripts/run66_r57r59_reuse10_20261031.py`是当时的批量脚本原件，保留历史绝对路径；复现请使用下面的新封装。

## 获取权重

两个约1GB的diffusion checkpoint通过Git LFS保存，约32KB的NNDM直接保存在Git中。

```bash
git lfs install
git clone https://github.com/hitoriCai/Contact-safety-of-diffusion-model.git
cd Contact-safety-of-diffusion-model
git lfs pull
```

三个权重的来源、完整大小和SHA256见`source_inventory/manifest.json`。
运行前可校验所有原始文件：

```bash
python scripts/run_eval.py --verify-only
```

## 环境与运行

原实验环境为Linux、Python3.9、PyTorch1.12.1及CUDA。主要依赖包括diffusers0.11.1、Hydra1.2.0、OmegaConf2.2.3、
gym0.21.0、pygame2.1.2、pymunk6.2.1、gurobipy12.0.3、NumPy1.23.3、SciPy1.9.1、wandb0.13.3和av10.0.0。
完整已安装版本快照见`source_inventory/PYTHON_PACKAGES_CURRENT.txt`；它是环境记录，不是已经验证可直接pip安装的锁文件。
求解需要可用的Gurobi许可证。仓库不包含许可证、账号凭据或原训练数据。

先检查一条命令（不加载模型、不运行环境）：

```bash
python scripts/run_eval.py --seed 376336367 --dry-run
```

再做3步冒烟检查：

```bash
python scripts/run_eval.py --seed 376336367 --workers 1 --device cuda:0 \
  --max-steps 3 --output-dir outputs/smoke
```

完整66条评测：

```bash
python scripts/run_eval.py --all --workers 2 --device cuda:0 \
  --output-dir outputs/r57_hard_r59_soft_full66
```

默认保留原来的300/700步规则，只有显式传入`--max-steps`才改成短测试。
每个seed单独进程运行；模型采样仍使用原workspace随机数逻辑。不同依赖、GPU或求解器版本可能影响数值结果。
视频在新输出的`results/seed_编号/media/`下。

本仓库提供这一次评测的运行依赖、完整原始权重与固定场景，不包含训练数据、其他实验的输出视频或Gurobi许可证。
