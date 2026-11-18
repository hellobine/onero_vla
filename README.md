# onero_vla

A1/H1 本地推理的独立机器人与 ROS 相机插件。Torch 模型加载、预处理、动作预测和反归一化使用官方 LeRobot API；原生 JAX 后端使用公开 OpenPI。

## 安装

两个仓库放在同一目录。官方 LeRobot 固定为 v0.7.0，不修改其源码。

```bash
git clone --branch v0.7.0 https://github.com/huggingface/lerobot.git lerobot
git -C lerobot checkout --detach 410c095eacc0011d998a73383eb94abed2715af0
git clone --branch origin https://github.com/hellobine/onero_vla.git lerobot_robot_onero
cd lerobot_robot_onero
bash scripts/install.sh --lerobot-path ../lerobot --release 0.3.4
```

机器人站点基线为 Ubuntu 24.04 / Python 3.12 / ROS 2 Jazzy。
先加载站点 ROS 环境：`source /opt/ros/jazzy/setup.bash`。
虚拟环境不加载系统 site-packages；ROS 通过 setup.bash 的路径加载，避免系统旧 SciPy 与 NumPy 2 冲突。
PI 模型安装追加 `--extra pi`；SmolVLA 追加 `--extra smolvla`；Diffusion 追加 `--extra diffusion`。
macOS 用于本地开发和 CPU 测试。Thor GPU、驱动与 NVIDIA PyTorch 需单独验证，不以安装成功代替验收。
Linux 默认 PyPI CUDA 13 构建需 NVIDIA 驱动 580.65 或更新版本；其他 CUDA 组合需单独生成并验证依赖锁。

## 配置

```bash
cp configs/a1.example.yaml configs/customer.yaml
# H1 改用 configs/h1.example.yaml
```

填写本地模型目录、设备、任务和机器人话题。三个相机名为 head / left / right。
默认 Cyclone DDS；新版机器人选择 `rmw_zenoh_cpp` 并填写 `zenoh_router: tcp/机器人地址:7447`。
不做自动网络扫描或固件版本猜测。

模型目录需要权重、config.json、policy_preprocessor.json、policy_postprocessor.json，
以及处理器引用的统计、tokenizer 和其他资源。模型必须是 16 维绝对关节位置模型，
采用本项目的状态及动作顺序。由模型提供方核对训练契约后，将
`configs/onero_model.example.json` 填写为模型目录中的 `onero_model.json`。
这份声明不执行动作顺序转换，不能用来冒充不兼容模型已适配。
启用的相对动作处理器暂不支持。相机保留原始 RGB 像素；模型自身处理器负责后续变换。

PI05 模型可配置以下模型字段，三个相机保持原始采集分辨率：

```yaml
model:
  backend: lerobot_torch
  path: /absolute/path/to/pretrained_model
  device: cuda
  task: Fold the clothes
  image_resolution: [168, 224]
  tokenizer_path: /absolute/path/to/local_tokenizer
  compile_model: false
```

`image_resolution` 使用 AREA 等比例缩小，不允许拉伸或放大，之后由 PI05 补边到 224×224。
机器人状态始终为 16 维，归一化及提示词处理完成后，由上游 PI05 填充内部 32 维。
PI05 加载会严格检查所有权重，缺失、额外、非有限权重或转换丢失均报错。
tokenizer 必须随模型离线交付，报告记录其文件哈希。首轮验证关闭编译；编译性能另行测试。

## OpenPI 原生 JAX

使用独立进程和独立依赖环境，公开 OpenPI 源码固定为
`215abfb217dbac7d5f1273282331b9b1866c0479`，放在相邻的 `openpi` 目录。
安装脚本检查全部 Python 源文件哈希，不安装 OpenPI 训练环境。

```bash
git clone https://github.com/Physical-Intelligence/openpi.git ../openpi
git -C ../openpi checkout --detach 215abfb217dbac7d5f1273282331b9b1866c0479
bash scripts/install_jax.sh ../openpi
```

安装脚本按主机架构选择独立依赖锁：Thor/aarch64 使用 `jax_runtime`；
RTX 5090/x86_64 使用 `jax_runtime_5090`，该环境的 Torch/torchvision 为 CPU 版，模型计算由 JAX CUDA 执行。
两套环境的 worker 源码和 OpenPI 源码清单一致。

模型包由提供方附带 `onero_openpi.json`、原生 `params/`、16 维分位数统计和本地 SentencePiece。
此后端首版限定 PI05：32 维内部动作、50 步、200 个 token、10 次去噪、16 维绝对动作。
模型包需要来自已验证的训练配置；不能把 Torch safetensors 目录作为 JAX 模型。

```yaml
model:
  backend: openpi_jax
  path: /absolute/path/to/openpi_model
  device: cuda
  task: Fold the clothes
  image_resolution: [168, 224]
  jax_python: ../jax_runtime_5090/.venv/bin/python
  openpi_source: ../../openpi
  backend_timeout_s: 900
```

以上为 5090 示例，相对路径按 YAML 所在目录解析；Thor 的 `jax_python` 改用
`../jax_runtime/.venv/bin/python`。`check`、`preflight`、`shadow` 命令与 Torch 相同。
JAX 强制 CUDA、关闭预分配；Thor 配置关闭 Triton GEMM、command buffer 和自动调优。
使用 JAX 0.9.2 / Flax 0.12.5 / Orbax 0.11.33；Orbax 元数据接口由独立适配层处理。
模型结构、采样、归一化和 tokenizer 均使用公开 OpenPI 实现，不修改上游源码。

## 检查与推理

### Torch 与 JAX 并列部署

同一项目通过 `model.backend` 选择后端，不需要更换代码或覆盖已有模型。
保留站点当前配置为 `configs/customer-jax.yaml`，另建
`configs/customer-torch.yaml`；两份文件各自绑定模型、统计与离线 tokenizer。
Torch 可参考 `configs/a1.pi05-torch.example.yaml`，机器人话题、相机和 ROS
通信设置沿用该站点已经验证的值；H1 设置 `robot.model: h1`。

| 配置 | 后端 | 模型资源 |
| --- | --- | --- |
| `customer-jax.yaml` | `openpi_jax` | 原生 params、统计、SentencePiece，独立 JAX 环境 |
| `customer-torch.yaml` | `lerobot_torch` | safetensors、LeRobot 处理器与统计、离线 HF tokenizer，主环境 |

Torch 模型目录示例：

```text
models/pi05-torch/
├── pretrained_model/   # 原始 LeRobot 权重、配置、处理器、统计、onero_model.json
├── tokenizer/          # tokenizer.json、tokenizer.model、tokenizer_config.json 等
└── provenance/         # artifact manifest、下载校验和契约来源
```

每份配置分别执行 `check`、`preflight`。正式推理时选择其一，运行至 Ctrl+C：

```bash
source /opt/ros/jazzy/setup.bash
# Torch
uv run --no-sync onero-infer run --config configs/customer-torch.yaml --enable-motion
# JAX（结束上一个推理进程后运行）
uv run --no-sync onero-infer run --config configs/customer-jax.yaml --enable-motion
```

### 连接检查与无动作预检

```bash
uv run --no-sync onero-infer config --config configs/customer.yaml
uv run --no-sync onero-infer check --config configs/customer.yaml
uv run --no-sync onero-infer preflight --config configs/customer.yaml --output runs/preflight-001
uv run --no-sync onero-infer shadow --config configs/customer.yaml --duration 3 --output runs/shadow-001
uv run --no-sync onero-infer run --config configs/customer.yaml --enable-motion
```

`check` / `preflight` / `shadow` 不创建控制发布者。输出目录必须是新目录。
`preflight` 保存原始观测、原始预测和处理后动作，以及生效的平滑参数；
`shadow` 连续读取真实观测并执行实际的队列与滤波流程，只保存动作，不发送。
`check --no-hardware` 仅验证文件契约与环境，不证明模型能加载或硬件可用。
运行始终设置 Hub 离线模式；第一次安装仍需联网。完整本地权重不再下载视觉骨干初始化权重。
完全离线安装包尚未提供。

`run` 先只读采集并预热模型，再连接控制通道。正常启动持续推理，按 Ctrl+C 退出；
仅在显式传入 `--duration` 时限时运行，加载与预热另计。
`runtime.duration_s` 只作为 `shadow` 的默认预览时长，不限制正常推理。
动作序号按实际取出执行的动作推进，等待模型或控制循环变慢时不按墙钟跳过预测行；
预测块以最新已执行动作序号为锚点，与参考主线的异步客户端一致。
正在执行的模型调用返回后再检查期限，不承诺强制中断模型计算。
控制频率以实测为准；超时会报错，不能自动放宽期限。
动作下发与参考主线 `4fcfbcf2` 的 H1 逻辑一致：按 16 维绝对位置协议发送预测及处理后的关节目标，
执行层不添加反馈差值限幅或角度重分支；夹爪沿用底层协议的量化。
动作期限、反馈/相机过期或部分发布失败会阻断后续命令；重新启动前先排障。
程序不自动开启跟随、回零或恢复动作。

Ctrl+C 退出。软件停止发布不等于物理停止，现场必须保留急停并遵循底层控制器停止规程。
同用户控制锁只协调使用本插件的进程，不阻止其他 ROS 发布者。

每次 `run` 自动保存到 `runs/run-*`：`events.jsonl` 记录反馈、目标、动作序号、队列和预测耗时，
`result.json` 记录频率、最大动作间隔、发送数及退出原因；Ctrl+C 后同样保留。
相机仍实时读取，默认不把图像保存为数据集。日志写入失败会报告，不改变动作值或中止控制。

## 时序平滑与卡尔曼

A1/H1 配置模板默认使用卡尔曼，时序平滑关闭。原有未配置模式的 YAML 继续使用 sync。

```yaml
runtime:
  fps: 30
  inference_mode: kalman_filter
  kalman_temporal_smoothing: false
```

- `sync`：上游 `select_action`，不叠加本项目滤波。
- `kalman_filter`：异步动作块队列，每次取出动作后执行逐关节卡尔曼。
- `temporal_smoothing`：新旧动作块重叠区线性过渡，短重叠使用旧块末帧补齐。
- 组合：`kalman_filter` 加 `kalman_temporal_smoothing: true`，先融合动作块，再执行卡尔曼。

左右首关节使用与主线一致的圆周卡尔曼；默认动作块融合沿用上游逐维线性加权。
显式启用时序平滑时，重叠区的圆周关节使用最短角路径融合。
普通关节的卡尔曼公式及默认参数与参考主线一致。融合窗口 `min_smooth_steps` 默认 8；
预取阈值 `chunk_size_threshold` 默认 0.5，单独时序平滑可显式设为 0.8。
普通重叠区按同一时间步聚合，默认 `aggregate_fn_name: weighted_average`（旧 0.3、新 0.7），
与主线所用上游默认值一致；也可设为 latest_only、average 或 conservative。
动作块模式要求模型声明上游 `chunk_inference_spec()` 契约；观察历史或内置 temporal ensemble
不支持时直接报错。超过 `max_chunk_age_s` 的动作及已过期时间步不会执行。
每次新运行重置滤波器和队列，模型线程不访问机器人。

CLI 可覆盖本次模式，例如：

```bash
uv run --no-sync onero-infer shadow --config configs/customer.yaml --duration 3 --output runs/kalman-001 --inference-mode kalman_filter --no-kalman-temporal-smoothing
uv run --no-sync onero-infer shadow --config configs/customer.yaml --duration 3 --output runs/temporal-001 --inference-mode temporal_smoothing
uv run --no-sync onero-infer shadow --config configs/customer.yaml --duration 3 --output runs/combined-001 --inference-mode kalman_filter --kalman-temporal-smoothing
```

## 开发验证

```bash
uv sync --frozen
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync pytest -q
```

测试覆盖真实上游插件工厂、ROS 替身的只读/发布边界，以及临时生成的小型 ACT 模型在 CPU 上的
真实保存、加载、预处理和动作预测。随机测试模型不用于实机，不进入交付包。
单元测试关闭第三方 pytest 插件自动加载，避免 ROS 的 launch_testing 插件引入无关系统依赖。
项目未附客户权重；实机动作、任务效果和 GPU 性能需要另外验收。

## Thor 无动作测试

```bash
uv run --no-sync python scripts/smoke_thor.py --device cuda --output runs/gpu-smoke-001
# 使用真实观测但不发送动作：
uv run --no-sync python scripts/smoke_thor.py --device cuda --config configs/customer.yaml --output runs/ros-smoke-001
```

脚本生成随机 ACT 测试权重，检查 CUDA、TorchVision CUDA 算子和 16 维预测。
结果保存为 smoke-result.json；测试权重标记 test_only，run 会拒绝使用。
