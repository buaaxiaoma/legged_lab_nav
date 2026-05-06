# legged_lab_nav sim-to-sim（Go2 Rough）

当前仅保留一条部署路线，并且仿真启动代码也已迁入本仓库：

- 控制器：`deploy/robots/go2`
- MuJoCo 仿真：`deploy/mujoco`


## 已对齐训练 run

- `/home/ws/projects/legged_lab_nav/logs/rsl_rl/unitree_go2_rough/2026-04-26_10-09-56_target_pos`

已同步到：

- `deploy/robots/go2/config/params/env.yaml`
- `deploy/robots/go2/config/params/agent.yaml`
- `deploy/robots/go2/config/params/deploy.yaml`

其中：

- `observations` 包含 `target_pos`，ONNX 输入维度为 `234`
- `commands.base_velocity.command_source = local_target`
- `commands.base_velocity.ranges` 已从训练 `velocity_ranges` 选择移动地形范围：`lin_vel_x=[0.0, 2.0]`，`ang_vel_z=[-1.5, 1.5]`
- MuJoCo 高程图参数已同步到 `deploy/mujoco/simulate_python/heightmap_config.yaml`
- `policy.onnx` / `policy.pt` 已从 `model_49999.pt` 导出到该 run 的 `exported` 目录，并同步到部署目录


## 0) 导出 / 同步部署配置

部署侧有两个入口，适用场景不同：

- `deploy/scripts/sync_go2_deploy_config_from_io.py`：用于训练 run 已经有 `io_descriptors/IO_descriptors.yaml` 的情况。它从训练导出的 IO 描述同步 deploy 配置。
- `deploy/scripts/export_policy_and_deploy.sh`：用于没有 `IO_descriptors.yaml` 或希望直接从 `env.yaml + checkpoint/exported policy` 生成 deploy 配置的情况。它内部调用 `generate_deploy_yaml.py`。

### 0.1 `sync_go2_deploy_config_from_io.py`

默认命令：

```bash
cd /home/ws/projects/legged_lab_nav
python3 deploy/scripts/sync_go2_deploy_config_from_io.py   --run-dir logs/rsl_rl/unitree_go2_rough/<run_dir>
```

必须输入：

- `<run-dir>/io_descriptors/IO_descriptors.yaml`
- `<run-dir>/params/env.yaml`
- `deploy/robots/go2/config/params/deploy.yaml`，作为被更新的目标 yaml
- `deploy/robots/go2/config/config.yaml`，用于更新 `FixStand.qs` 最后一个 waypoint

默认可选输入：

- `<run-dir>/exported/policy.onnx`
- `<run-dir>/exported/policy.pt`

常用参数：

- `--run-dir`：训练 run 目录。
- `--deploy-yaml`：输出/更新的 deploy yaml，默认 `deploy/robots/go2/config/params/deploy.yaml`。
- `--fsm-config`：输出/更新的 FSM yaml，默认 `deploy/robots/go2/config/config.yaml`。
- `--heightmap-config`：输出/更新的 MuJoCo heightmap yaml，默认 `deploy/mujoco/simulate_python/heightmap_config.yaml`。
- `--policy-output-dir`：策略复制目标目录，默认 `deploy/robots/go2/config/exported`。
- `--policy-source-dir`：策略来源目录，默认 `<run-dir>/exported`。
- `--force-local-target-command`：当策略 observation 不含 `target_pos`，但速度命令仍由 `rt/local_target_pos_b` 生成时使用。

输出/更新文件：

- `deploy/robots/go2/config/params/deploy.yaml`
- `deploy/robots/go2/config/config.yaml`
- `deploy/mujoco/simulate_python/heightmap_config.yaml`
- `deploy/robots/go2/config/exported/policy.onnx`
- `deploy/robots/go2/config/exported/policy.pt`

同步内容：

- 从 `IO_descriptors.yaml` 同步 joint 顺序、action scale/offset、observation scale/clip/history、默认关节姿态和 PD 参数。
- 从 `env.yaml` 同步 heightmap 参数和移动地形速度范围。
- 从策略来源目录复制 `policy.onnx` / `policy.pt`，并打印 SHA256。
- 根据默认关节姿态更新 `config.yaml` 里 `FixStand.qs` 最终姿态。

无 `target_pos` observation 但仍使用局部目标速度命令时：

```bash
python3 deploy/scripts/sync_go2_deploy_config_from_io.py   --run-dir logs/rsl_rl/unitree_go2_rough/<run_dir>   --force-local-target-command
```

如果策略文件不在 `<run-dir>/exported`：

```bash
python3 deploy/scripts/sync_go2_deploy_config_from_io.py   --run-dir logs/rsl_rl/unitree_go2_rough/<run_dir>   --policy-source-dir /path/to/exported
```

### 0.2 `export_policy_and_deploy.sh`

默认命令示例：

```bash
cd /home/ws/projects/legged_lab_nav
./deploy/scripts/export_policy_and_deploy.sh   --task Lab-Position-Rough-Unitree-Go2-Play-v0   --checkpoint logs/rsl_rl/unitree_go2_rough/<run_dir>/model_xxx.pt   --env-yaml logs/rsl_rl/unitree_go2_rough/<run_dir>/params/env.yaml   --agent-yaml logs/rsl_rl/unitree_go2_rough/<run_dir>/params/agent.yaml
```

必须输入：

- `--task`：Gym task id，例如 `Lab-Position-Rough-Unitree-Go2-Play-v0`。
- `--checkpoint`：RSL-RL checkpoint，例如 `<run-dir>/model_28500.pt`。如果只生成配置且不导出策略，可配合 `--skip-policy-export`。
- `--env-yaml`：训练 run 的 `params/env.yaml`。
- `--agent-yaml`：训练 run 的 `params/agent.yaml`。

默认可选输入：

- checkpoint 相邻目录的 `exported/policy.onnx` 和 `exported/policy.pt`，即 `<run-dir>/exported/`。
- 如果提供 `--policy-source-dir`，则优先从该目录复制 `policy.onnx` / `policy.pt`。

常用参数：

- `--output-dir`：部署配置输出根目录，默认 `deploy/robots/go2/config`。
- `--policy-source-dir`：策略来源目录。提供后直接复制策略，不从 checkpoint 重新导出。
- `--skip-policy-export`：只生成 deploy yaml，不导出策略；如果存在 `--policy-source-dir` 或 checkpoint 相邻 `exported/`，仍会优先复制已有策略。
- `--force-local-target-command`：策略 observation 不含 `target_pos`，但速度命令仍由局部目标生成时使用。
- `--num-envs`：创建导出环境的数量，通常为 `1`。

输出/更新文件：

- `<output-dir>/params/deploy.yaml`，默认 `deploy/robots/go2/config/params/deploy.yaml`
- `<output-dir>/params/env.yaml`
- `<output-dir>/params/agent.yaml`
- `<output-dir>/params/env.input.yaml`，如果提供了 `--env-yaml`
- `<output-dir>/params/agent.input.yaml`，如果提供了 `--agent-yaml`
- `<output-dir>/exported/policy.onnx`
- `<output-dir>/exported/policy.pt`

策略处理顺序：

- 如果 `--policy-source-dir` 存在，复制其中的 `policy.onnx` / `policy.pt` 到 `<output-dir>/exported`。
- 否则如果 checkpoint 相邻目录存在 `<run-dir>/exported/policy.onnx` 和 `policy.pt`，复制这两个文件。
- 否则如果没有 `--skip-policy-export`，从 `--checkpoint` 重新导出 `policy.onnx` / `policy.pt`。
- 每次复制或导出后都会打印 SHA256，用于确认 deploy 策略和 run 策略一致。

无 `target_pos` observation 但仍使用局部目标速度命令时：

```bash
./deploy/scripts/export_policy_and_deploy.sh   --task Lab-Position-Rough-Unitree-Go2-Play-v0   --checkpoint logs/rsl_rl/unitree_go2_rough/<run_dir>/model_xxx.pt   --env-yaml logs/rsl_rl/unitree_go2_rough/<run_dir>/params/env.yaml   --agent-yaml logs/rsl_rl/unitree_go2_rough/<run_dir>/params/agent.yaml   --force-local-target-command
```

### 0.3 导出后是否需要重新编译

只更新 yaml 或 policy 文件时不需要重新编译，但必须重启 `go2_ctrl`。只有修改 C++ 源码、CMake 或依赖时才需要重新编译。

## 运行时读取的配置文件

### 控制器 `go2_ctrl`

启动命令：

```bash
cd /home/ws/projects/legged_lab_nav/deploy/robots/go2/build
./go2_ctrl --network lo
```

运行时读取：

- `deploy/robots/go2/config/config.yaml`
  - 由 `param::load_config_file()` 读取。
  - 决定 FSM 状态、`policy_dir`、`FixStand` 姿态、PD 参数、heightmap topic、local target topic、运行时 command range 覆盖等。
- `deploy/robots/go2/config/params/deploy.yaml`
  - 由 `FSM.Velocity.policy_dir` 指向的目录决定，当前默认 `policy_dir: config`。
  - 决定 joint 映射、step dt、PD、action 配置、observation 配置、command range 和 `command_source`。
- `deploy/robots/go2/config/exported/policy.onnx`
  - ONNXRuntime 推理使用的实际策略文件。
- `deploy/robots/go2/config/exported/policy.pt`
  - 不被 `go2_ctrl` 直接读取，但保留用于一致性检查和后续调试。

注意：`go2_ctrl` 启动时读取这些文件。修改 yaml 或 policy 后，需要重启 `go2_ctrl`。

### MuJoCo 仿真

启动命令：

```bash
cd /home/ws/projects/legged_lab_nav/deploy/mujoco/simulate_python
python3 unitree_mujoco.py
```

运行时读取：

- `deploy/mujoco/simulate_python/config.py`
  - MuJoCo 主配置，包含 `ROBOT_SCENE`、`DOMAIN_ID`、`INTERFACE`、heightmap topic、viewer 相机等。
- `deploy/mujoco/unitree_robots/go2/scene_terrain.xml`
  - 当前默认 `ROBOT_SCENE`，描述 Go2 和地形场景。
- `deploy/mujoco/unitree_robots/go2/go2.xml`
  - 由 scene XML include，描述机器人本体。
- `deploy/mujoco/unitree_robots/go2/assets/*`
  - 机器人 mesh 资源。
- `deploy/mujoco/simulate_python/heightmap_config.yaml`
  - MuJoCo bridge 读取的训练 heightmap 参数，覆盖 `config.py` 中的默认 heightmap size/resolution/grid/ray offset。

### 运行时 topic 输入

- `rt/lowstate` / `rt/sportmodestate`：MuJoCo bridge 发布，控制器读取。
- `rt/lowcmd`：控制器发布，MuJoCo bridge 读取。
- `rt/heightmap`：MuJoCo bridge 发布，控制器读取。
- `rt/local_target_pos_b`：`publish_local_target.py` 发布，控制器读取；内容是机器人本体系局部目标 `[x_b, y_b]`。

## 1) 编译控制器

```bash
cd /home/ws/projects/legged_lab_nav/deploy/robots/go2
cmake -S . -B build
cmake --build build -j4
```

## 2) 启动 MuJoCo 仿真（终端 A）

```bash
cd /home/ws/projects/legged_lab_nav/deploy/mujoco/simulate_python
python3 unitree_mujoco.py
```

默认配置文件：

- `deploy/mujoco/simulate_python/config.py`

默认已设置：

- `ROBOT_SCENE = ../unitree_robots/go2/scene_terrain.xml`
- `ENABLE_PUSH_BOX_OBS = False`
- `DOMAIN_ID = 0`
- `INTERFACE = lo`

如果修改控制器侧网络参数，需要同步这里的 `DOMAIN_ID / INTERFACE`。


## 地形场景

MuJoCo 本地目录结构：

- `deploy/mujoco/simulate_python/`：仿真主程序与 DDS bridge。
- `deploy/mujoco/unitree_robots/go2/`：Go2 MuJoCo 场景与 mesh 资源。

默认 MuJoCo 场景已切换为：

- `deploy/mujoco/unitree_robots/go2/scene_terrain.xml`

重新生成地形：

```bash
cd /home/ws/projects/legged_lab_nav
python3 deploy/mujoco/tools/generate_go2_terrain_scene.py --terrain mixed --seed 7
```

`--terrain` 可选项：

- `flat`：仅平地。
- `rough`：类 Perlin 粗糙地面，噪声高度最高约 `0.10 m`，路线宽度 `3.8 m`。
- `stairs`：训练范围内的常规楼梯，包含只上楼、上/下、下/上组合。
- `stairs-high`：高金字塔/倒金字塔楼梯，台阶高度约 `0.08-0.45 m`。
- `pit`：随机高度高台和 `5 m` 平地间隔，不在高台前添加楼梯过渡。
- `gap`：训练范围内的简单/困难方形 gap。
- `mixed`：当前推荐混合路线，只包含 `rough`、`stairs up` 和 easy `gap`。

使用不同 `--seed` 可以生成确定性的不同布局。`stairs-high` 和 hard gap 包含较难样例，不保证所有 checkpoint 都能通过。

viewer 相机默认跟随 `base_link`。可在 `deploy/mujoco/simulate_python/config.py` 中调整：

- `ENABLE_TRACKING_CAMERA`
- `TRACK_CAMERA_DISTANCE`
- `TRACK_CAMERA_AZIMUTH`
- `TRACK_CAMERA_ELEVATION`

如果只想回到简单平地场景，把 `deploy/mujoco/simulate_python/config.py` 里的 `ROBOT_SCENE` 改回：

```python
ROBOT_SCENE = "../unitree_robots/" + ROBOT + "/scene.xml"
```

## 3) 启动策略控制器（终端 B）

```bash
cd /home/ws/projects/legged_lab_nav/deploy/robots/go2/build
./go2_ctrl --network lo
```

## 4) 发布局部目标点（终端 C）

```bash
cd /home/ws/projects/legged_lab_nav/deploy/robots/go2
python3 tools/publish_local_target.py --interface lo --domain-id 0 --target 1.2 0.0
```

输入是机器人本体系局部目标：`[x_b, y_b]`。

## 5) 切状态

- `L2 + A`：进入 `FixStand`
- `Start`：进入 `Velocity`
- `LT + B`：回到 `Passive`

也可以用键盘在 `go2_ctrl` 所在终端切状态：

- `2`：`Passive` -> `FixStand`
- `3`：`FixStand` -> `Velocity`
- `1`：回到 `Passive`

## 6) 调试 topic

如果进入 `Velocity` 后仍不动，先确认 local target 和 lowcmd 是否在更新：

```bash
cd /home/ws/projects/legged_lab_nav/deploy/mujoco/simulate_python
python3 test/monitor_sim2sim.py --interface lo --domain-id 0 --no-clear
```

重点看：

- `local_target` 是否显示 `target_b=[...]`，否则是目标点发布/Domain/网卡问题。
- `lowcmd` 是否在 `Velocity` 后变化，否则是控制器没有进入策略状态或策略输出近似默认站姿。
- `heightmap` 是否为 `17x11` 且有 187 个数据。

## 可选：运行时速度范围覆盖

同步脚本现在会自动把当前 run 的移动地形速度范围写入 `deploy.yaml`。如果仍需手动覆盖，可在：

- `deploy/robots/go2/config/config.yaml`

设置：

```yaml
FSM:
  Velocity:
    command_ranges:
      lin_vel_x: [0.0, 2.0]
      lin_vel_y: [0.0, 0.0]
      ang_vel_z: [-1.5, 1.5]
```
