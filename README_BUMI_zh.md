# Bumi 快速复现

本指南只介绍松延 Bumi 适配版 SuperMimic 的安装、演示、从头训练和续训。Bumi 模型、40 个处理后的 LAFAN1 动作和最新 checkpoint 放在仓库 Release 附件中。

本项目将 SuperMimic 适配到松延 Bumi。共享训练实现源自 [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)，Bumi MJCF 和配置工作参考了 [bumi_mujoco](https://github.com/diaskabdualiev/bumi_mujoco)。本项目不包含上游 G1 实验文档、媒体和 G1 专用工具；G1 实验结果不代表 Bumi 结果。复用代码的来源及许可文本见 `THIRD_PARTY_NOTICES.md`。

## 1. 准备机器

已验证配置：Ubuntu 24.04、Python 3.12.16、RTX 4090 24 GB、NVIDIA 驱动 595.99.02、PyTorch 2.14.1+cu130、mjlab `8bb2bc9`、MuJoCo 3.16.0。其他兼容 CUDA 13.0 的 NVIDIA Linux 机器可尝试；4090 上 2048 环境可训练，4096 环境曾显存不足。建议预留至少 25 GB 磁盘空间。

先用 `nvidia-smi` 确认显卡和驱动可用，再安装 `git`、`curl`、`tar`。下面的命令只在用户目录中安装 Python 和项目环境。

## 2. 克隆代码并下载 Release

在 GitHub 新仓库页面点绿色 **Code**，复制 HTTPS 地址。在 Linux 终端先建一个专用目录，再克隆：

```bash
mkdir -p "$HOME/gsj/supermimic_bumi"
cd "$HOME/gsj/supermimic_bumi"
git clone https://github.com/linyunzhizhe/MySuperMimic.git
```

GitHub 登录提示应使用自己的账号或凭据，不要把密码、令牌发给他人。

在仓库网页右侧点 **Releases**，进入最新版本，在 **Assets** 中下载下列三个文件，并上传或复制到 `$HOME/gsj/supermimic_bumi/`：

- `bumi-mujoco-assets.tar.gz`：Bumi 场景和 mesh，约 21 MB。
- `bumi-lafan1-40-fixed-motions.tar.gz`：实际训练使用的 40 个 NPZ，约 1 GB。
- `bumi_ckpt_0320000_latest.pt`：最新 checkpoint，约 194 MB，迭代 320000。

Release 还附有 `SHA256SUMS.txt`。下载后在该目录执行 `sha256sum -c SHA256SUMS.txt`，三个文件都应显示“成功”或 `OK`，再解压和运行。

在该目录执行：

```bash
tar -xzf bumi-mujoco-assets.tar.gz
tar -xzf bumi-lafan1-40-fixed-motions.tar.gz
mkdir -p MySuperMimic/logs/bumi_resume_2048_20261118_201522
cp bumi_ckpt_0320000_latest.pt MySuperMimic/logs/bumi_resume_2048_20261118_201522/ckpt_latest.pt
find bumi_work/motions/supermimic_bumi_lafan1_40_fixed -maxdepth 1 -name '*.npz' | wc -l
```

最后应显示 `40`。目录中还应有 `bumi_work/source/bumi_mujoco-main/bumi_robot/scene.xml`。checkpoint 只需下载最新的这一份；旧的 68000 轮文件不是演示或继续训练的必需品。

## 3. 安装独立 Python 环境

```bash
cd "$HOME/gsj/supermimic_bumi/MySuperMimic"
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/../bin" UV_NO_MODIFY_PATH=1 sh
export PATH="$PWD/../bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$PWD/../.uv-python"
export UV_CACHE_DIR="$PWD/../.uv-cache"
export XDG_CACHE_HOME="$PWD/../.cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/../.cache/torchinductor"
uv python install 3.12.16
uv sync --frozen --python 3.12.16
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
```

依赖根据 `uv.lock` 安装。`uv sync` 需要访问 GitHub 上固定提交的 mjlab，以及 PyPI、PyTorch 和 NVIDIA 包索引。不要复制旧机器的 `.venv`。

## 4. 演示训练好的策略

新开 Linux 终端时，先进入项目并执行：

```bash
cd "$HOME/gsj/supermimic_bumi/MySuperMimic"
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export BUMI_SCENE_XML="$PWD/../bumi_work/source/bumi_mujoco-main/bumi_robot/scene.xml"
export BUMI_MOTION_DIR="$PWD/../bumi_work/motions/supermimic_bumi_lafan1_40_fixed"
python scripts/play.py --ckpt logs/bumi_resume_2048_20261118_201522/ckpt_latest.pt --npz-dir "$BUMI_MOTION_DIR" --clips walk1_subject1 --robot bumi --scene-xml "$BUMI_SCENE_XML" --device cuda:0 --viewer viser
```

本机浏览器打开 `http://localhost:8080`。通过 SSH 使用远端机器时，在本地另开终端执行 `ssh -N -L 8080:127.0.0.1:8080 USER@HOST`，保持窗口打开后访问该地址。目标机器有图形桌面时可把 `--viewer viser` 改成 `native`。

## 5. 从零训练

在完成第 4 节的环境变量设置后，执行：

```bash
mapfile -t BUMI_BODIES < <(python -c 'from supermimic.robots.bumi import BUMI_BODY_NAMES; print("\n".join(BUMI_BODY_NAMES))')
RUN_NAME="bumi_fresh_$(date +%Y%m%d_%H%M%S)"
python -m supermimic.scripts.train --run-name "$RUN_NAME" --log-root logs --seed 0 --device cuda:0 --data.npz-dir "$BUMI_MOTION_DIR" --env.robot-cfg-factory supermimic.robots.bumi:get_bumi_robot_cfg --env.tracked-bodies "${BUMI_BODIES[@]}" --env.anchor-body-name base_link --env.root-body-name base_link --env.fk-model "$BUMI_SCENE_XML" --env.com-body-name base_link --env.num-envs 2048 --schedule.total-iterations 320000 --schedule.checkpoint-interval 4000
```

日志在 `logs/$RUN_NAME/`，每 4000 轮保存一次 checkpoint。训练到 320000 轮需要较长时间；策略能否复现效果还取决于 GPU、依赖和动作数据一致性。

## 6. 从最新 checkpoint 继续训练

第 320000 轮 checkpoint 已达到原训练上限。续训时把总迭代数提高，例如 400000：

```bash
RUN_NAME="bumi_resume_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/run_bumi_resume.sh "$RUN_NAME" 2048 "$PWD/logs/bumi_resume_2048_20261118_201522/ckpt_latest.pt" 400000 > "logs/${RUN_NAME}_console.log" 2>&1 < /dev/null &
echo "PID=$! RUN_NAME=$RUN_NAME"
```

这个脚本会使用全部 40 个动作，默认每 4000 轮保存。查看日志：`tail -f "logs/${RUN_NAME}_console.log"`；退出查看按 `Ctrl+C`，不会停止训练。要停止训练，执行 `kill -TERM PID`，将 `PID` 换成启动时打印的数字。下次再续训，从上次新运行目录的 `ckpt_latest.pt` 读取，并设一个更高的总轮数。

## 7. TensorBoard

训练中和训练结束后都能查看事件文件：

```bash
tensorboard --logdir logs --host 127.0.0.1 --port 6006
```

本机浏览器打开 `http://localhost:6006`；SSH 远程访问时先在本地终端执行 `ssh -N -L 6006:127.0.0.1:6006 USER@HOST`。关闭 TensorBoard 不会停止训练。

## 来源与许可

本项目方法参考 [SuperMimic 论文](https://arxiv.org/abs/2610.28480)，共享训练实现源自 [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)，Bumi MJCF 和配置工作参考 [bumi_mujoco](https://github.com/diaskabdualiev/bumi_mujoco)。复用代码的版权和许可文本见 `THIRD_PARTY_NOTICES.md`。`bumi_mujoco` 项目说明 Bumi URDF/网格模型权利归 Noetix Robotics；发布模型、重定向动作或 checkpoint 前需分别确认其再分发许可。本文档不把 G1 基准分数作为 Bumi 成绩，也未声称完成正式 Bumi 基准评测。
