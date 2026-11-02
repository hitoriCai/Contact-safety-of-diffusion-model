# r57硬CBF＋r59有界软CBF：文件清单

源项目：`/home/cps/czt/2026_summer/0928_lowdim/eval_lowdim_0.8964`
实验：`/home/cps/czt/2026_summer/0928_lowdim/eval_lowdim_0.8964/data/eval/10.31_59px+有界软约束`

这是文件清单，没有复制模型、修改项目或上传GitHub。路径映射是后续仓库整理建议；原代码中的绝对路径尚未重写。

## 文件数量

| 类别 | 文件数 | 总大小MiB |
| --- | ---: | ---: |
| project_code | 62 | 0.5074 |
| batch_launcher | 1 | 0.0025 |
| checkpoint | 3 | 2009.4430 |
| scenario_manifest | 66 | 0.1885 |
| installation_metadata | 1 | 0.0001 |
| frozen_run_config | 1 | 0.3439 |
| provenance | 2 | 0.0145 |
| result_summary | 2 | 0.0281 |

合计：138个源文件；代码（含安装脚本）＋权重：67个。

## 三个权重

| 用途 | 大小MiB | 原路径 |
| --- | ---: | --- |
| model1 | 1004.049 | `/home/cps/czt/2026_summer/0928_lowdim/eval_lowdim_0.8964/data/outputs/2026.09.27/model1_film200_action4_e600/checkpoints/last/epoch=0600.ckpt` |
| model2_v6_3_epoch500 | 1005.363 | `/home/cps/czt/2026_summer/model2_training_comparison_20261019/model2_v6_3_vanilla_seed43_20261019/runs/seed43/vanilla/checkpoints/epoch0500.ckpt` |
| NNDM_dim24 | 0.031 | `/home/cps/czt/2026_summer/nndm_combined370_20261025/full/nndm_24d_original206_plus_model2v62_370.pth` |

## 核对与边界

- 本轮记录的代码SHA256及三个权重SHA256均匹配。
- 入口为experiments/model2_cbf_local25_mixed_stages/eval.py；未启用depth_budget控制器。
- 包含继承/导入所需的full_page、global_then_clip等模块，不代表本轮启用了全域求解或裁切。
- 包含checkpoint实际动态加载的policy、workspace、model类，以及导入依赖。
- 66份场景manifest和冻结配置用于复现同一组初始场景；不是训练数据。
- manifest.tsv/manifest.json列出每个文件的原始绝对路径、建议仓库路径、字节数和SHA256。
- FILE_TREE.txt是建议仓库文件树。PYTHON_PACKAGES_CURRENT.txt是当前远端环境快照，不是实验时完整环境锁文件。
- 现有根README.md/reproduce_200.sh描述旧v5实验，未作为本轮复现入口纳入。
- 文件清单没有自动收录训练数据、缓存、视频、旧实验输出。若需要展示结果，可另选代表视频。
- GitHub普通Git不接受超过100MiB的单文件；较大ckpt使用Git LFS或Release。
  官方说明：https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github

## 文件树

```text
proposed_repository/
├── checkpoints/
│   ├── model1_epoch0600.ckpt
│   ├── model2_v6_3_epoch0500.ckpt
│   └── nndm_dim24.pth
├── diffusion_policy/
│   ├── common/
│   │   ├── json_logger.py
│   │   ├── pytorch_util.py
│   │   └── timestamp_accumulator.py
│   ├── dataset/
│   │   └── base_dataset.py
│   ├── env/
│   │   └── pusht/
│   │       ├── __init__.py
│   │       ├── pusht_env.py
│   │       ├── pusht_keypoints_env.py
│   │       ├── pymunk_keypoint_manager.py
│   │       └── pymunk_override.py
│   ├── env_runner/
│   │   └── base_lowdim_runner.py
│   ├── gym_util/
│   │   ├── async_vector_env.py
│   │   ├── multistep_wrapper.py
│   │   └── video_recording_wrapper.py
│   ├── model/
│   │   ├── common/
│   │   │   ├── dict_of_tensor_mixin.py
│   │   │   ├── lr_scheduler.py
│   │   │   ├── module_attr_mixin.py
│   │   │   └── normalizer.py
│   │   └── diffusion/
│   │       ├── conditional_unet1d.py
│   │       ├── conv1d_components.py
│   │       ├── mask_generator.py
│   │       └── positional_embedding.py
│   ├── policy/
│   │   ├── base_lowdim_policy.py
│   │   ├── diffusion_unet_lowdim_policy.py
│   │   ├── diffusion_unet_lowdim_policy_origin.py
│   │   ├── diffusion_unet_lowdim_policy_regrasp.py
│   │   └── mind.py
│   ├── real_world/
│   │   └── video_recorder.py
│   └── workspace/
│       ├── base_workspace.py
│       └── train_diffusion_unet_lowdim_workspace.py
├── experiment/
│   ├── comparison_vs_previous.json
│   ├── comparison_vs_previous.md
│   ├── experiment_config.json
│   ├── progress.json
│   └── reuse_audit.json
├── experiments/
│   ├── contact_gap_5s/
│   │   ├── __init__.py
│   │   ├── contact_env.py
│   │   ├── contact_wrapper.py
│   │   ├── policy_contact_gap.py
│   │   └── runner_contact_gap.py
│   ├── model2_cbf_diverse_hard/
│   │   └── policy.py
│   ├── model2_cbf_diverse_hard_infeasible_regrasp/
│   │   ├── __init__.py
│   │   ├── policy.py
│   │   └── runner.py
│   ├── model2_cbf_diverse_v2/
│   │   ├── policy.py
│   │   └── runner.py
│   ├── model2_cbf_full_page/
│   │   ├── __init__.py
│   │   ├── filter.py
│   │   └── policy.py
│   ├── model2_cbf_full_page_1step_p047/
│   │   ├── __init__.py
│   │   └── policy.py
│   ├── model2_cbf_global_then_clip/
│   │   ├── __init__.py
│   │   ├── filter.py
│   │   └── policy.py
│   ├── model2_cbf_local25_dedup/
│   │   ├── __init__.py
│   │   └── policy.py
│   ├── model2_cbf_local25_mixed_stages/
│   │   ├── __init__.py
│   │   ├── eval.py
│   │   ├── policy.py
│   │   ├── run66.py
│   │   └── runner.py
│   ├── model2_cbf_local25_route_memory/
│   │   ├── __init__.py
│   │   ├── policy.py
│   │   ├── routes.py
│   │   └── runner.py
│   ├── nndm_dim24_model2_v63_eval/
│   │   └── runner_core_stop10.py
│   └── __init__.py
├── scenarios/
│   ├── seed_1041083166.json
│   ├── seed_1131959911.json
│   ├── seed_115985547.json
│   ├── seed_1175287068.json
│   ├── seed_1295743469.json
│   ├── seed_1303113791.json
│   ├── seed_1319308783.json
│   ├── seed_1337735547.json
│   ├── seed_1344181256.json
│   ├── seed_137079570.json
│   ├── seed_1373672124.json
│   ├── seed_1419917087.json
│   ├── seed_1424418293.json
│   ├── seed_1487629675.json
│   ├── seed_1505725293.json
│   ├── seed_1524623171.json
│   ├── seed_1528120788.json
│   ├── seed_1532647514.json
│   ├── seed_1544284046.json
│   ├── seed_1670000248.json
│   ├── seed_1673420483.json
│   ├── seed_1733259484.json
│   ├── seed_1738193931.json
│   ├── seed_1772518337.json
│   ├── seed_1793481395.json
│   ├── seed_1810004900.json
│   ├── seed_1849087796.json
│   ├── seed_1857772030.json
│   ├── seed_1866707754.json
│   ├── seed_18685189.json
│   ├── seed_1879934180.json
│   ├── seed_1891222525.json
│   ├── seed_1898837579.json
│   ├── seed_1915220046.json
│   ├── seed_1921019152.json
│   ├── seed_1948800791.json
│   ├── seed_20000022.json
│   ├── seed_2061621831.json
│   ├── seed_2114184371.json
│   ├── seed_245421928.json
│   ├── seed_25793086.json
│   ├── seed_276109436.json
│   ├── seed_376336367.json
│   ├── seed_38070626.json
│   ├── seed_394657593.json
│   ├── seed_397946280.json
│   ├── seed_438457735.json
│   ├── seed_452469155.json
│   ├── seed_453988355.json
│   ├── seed_461133718.json
│   ├── seed_46807587.json
│   ├── seed_506543433.json
│   ├── seed_516533431.json
│   ├── seed_582597579.json
│   ├── seed_589211402.json
│   ├── seed_605147828.json
│   ├── seed_647631320.json
│   ├── seed_684952848.json
│   ├── seed_721779023.json
│   ├── seed_791545074.json
│   ├── seed_810178128.json
│   ├── seed_825929573.json
│   ├── seed_923571703.json
│   ├── seed_948040208.json
│   ├── seed_952948542.json
│   └── seed_969453231.json
├── scripts/
│   └── run66_r57r59_reuse10_20261031.py
├── eval.py
└── setup.py
```


## 源码完整性验证

- 将清单中的项目源码复制到隔离临时目录后，入口、workspace、checkpoint所用模型类均成功导入；没有回退到原项目代码路径。
- 旧的diffusion_policy.environment.push_t_env导入仅存在于未调用的test()演示函数中，不是本轮缺失依赖。
- source_import_check.json记录验证范围；这不是重新评测或训练。
