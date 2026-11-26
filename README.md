# SuperMimic-Bumi

SuperMimic's differentiable world-model training pipeline adapted to the Songyan Bumi humanoid. This repository contains the Bumi integration and the shared training code. The Bumi experiment tracks 22 bodies and controls 21 joints using 40 retargeted LAFAN1 clips. The latest available checkpoint is at iteration 320000.

**This project adapts SuperMimic to Songyan Bumi.** The shared training implementation is derived from [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced); this copy contains the Bumi integration and the code needed for Bumi training and replay. It omits the upstream G1 experiment notes, media, and G1-specific utilities. G1 results are not Bumi results. For step-by-step instructions in Chinese, see [README_BUMI_zh.md](README_BUMI_zh.md).

## What is in the repository

| Path | Purpose |
|---|---|
| `src/supermimic/robots/bumi.py` | Bumi joint and body order, actuator gains, home pose and action scales |
| `src/supermimic/envs/` | Tracking environment, observations, rewards and motion library used by Bumi |
| `src/supermimic/world_model/`, `imagination/`, `policy/`, `training/` | Shared world model, BPTT policy updates and training loop |
| `src/supermimic/scripts/train.py` | Training command-line entry point |
| `scripts/play.py` | Policy replay with native MuJoCo or browser-based Viser viewer |
| `scripts/run_bumi_resume.sh` | Bumi checkpoint resume using paths relative to the clone |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python dependencies and pinned environment |
| `.gitignore`, `.gitattributes` | Exclude generated data and keep shell scripts in Linux line endings |
| `THIRD_PARTY_NOTICES.md` | Attributions and license texts for reused SuperMimic and Bumi simulator code |

The Bumi MJCF, 40 motion NPZ files and trained checkpoint are distributed separately from the source. The Bumi simulator source is [bumi_mujoco](https://github.com/diaskabdualiev/bumi_mujoco); its README states that rights to the robot URDF and meshes belong to Noetix Robotics. Confirm redistribution permission for those model assets, the retargeted motion data, and checkpoints before publishing them in a Release. Source-code license notices do not grant rights to these assets.

## Release assets

Download these four files from the latest release into a new workspace directory:

| File | Use |
|---|---|
| `bumi-mujoco-assets.tar.gz` | Bumi scene XML, model files and meshes |
| `bumi-lafan1-40-fixed-motions.tar.gz` | The 40 retargeted LAFAN1 clips used in training |
| `bumi_ckpt_0320000_latest.pt` | Latest Bumi policy and optimizer state, iteration 320000 |
| `SHA256SUMS.txt` | Download integrity checks |

Run `sha256sum -c SHA256SUMS.txt` before extraction. One latest checkpoint is sufficient for replay and further training; older checkpoints are not included in this release.

## Quick start

Tested on Ubuntu 24.04, NVIDIA RTX 4090 (24 GB), driver 595.99.02, Python 3.12.16, PyTorch 2.14.1+cu130 and MuJoCo 3.16.0. Use an NVIDIA driver compatible with CUDA 13.0. Training with 2048 parallel environments worked on this GPU; 4096 ran out of GPU memory.

From a new workspace directory containing the downloaded release assets:

```bash
git clone https://github.com/linyunzhizhe/MySuperMimic.git
sha256sum -c SHA256SUMS.txt
tar -xzf bumi-mujoco-assets.tar.gz
tar -xzf bumi-lafan1-40-fixed-motions.tar.gz
mkdir -p MySuperMimic/logs/bumi_resume_2048_20261118_201522
cp bumi_ckpt_0320000_latest.pt MySuperMimic/logs/bumi_resume_2048_20261118_201522/ckpt_latest.pt
find bumi_work/motions/supermimic_bumi_lafan1_40_fixed -maxdepth 1 -name '*.npz' | wc -l
```

The final command must print `40`. The scene must be at `bumi_work/source/bumi_mujoco-main/bumi_robot/scene.xml`.

Install an isolated Python environment in the clone:

```bash
cd MySuperMimic
curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$PWD/../bin" UV_NO_MODIFY_PATH=1 sh
export PATH="$PWD/../bin:$PATH"
export UV_PYTHON_INSTALL_DIR="$PWD/../.uv-python"
export UV_CACHE_DIR="$PWD/../.uv-cache"
export XDG_CACHE_HOME="$PWD/../.cache"
export TORCHINDUCTOR_CACHE_DIR="$PWD/../.cache/torchinductor"
uv python install 3.12.16
uv sync --frozen --python 3.12.16
```

`uv sync` needs access to the pinned mjlab Git commit, PyPI, the PyTorch CUDA index and the NVIDIA Python index. Do not copy a virtual environment from another machine.

## Replay the trained policy

Run from `MySuperMimic` in a new terminal:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
export BUMI_SCENE_XML="$PWD/../bumi_work/source/bumi_mujoco-main/bumi_robot/scene.xml"
export BUMI_MOTION_DIR="$PWD/../bumi_work/motions/supermimic_bumi_lafan1_40_fixed"
python scripts/play.py --ckpt logs/bumi_resume_2048_20261118_201522/ckpt_latest.pt --npz-dir "$BUMI_MOTION_DIR" --clips walk1_subject1 --robot bumi --scene-xml "$BUMI_SCENE_XML" --device cuda:0 --viewer viser
```

Open `http://localhost:8080` on the same machine. For a remote machine, run `ssh -N -L 8080:127.0.0.1:8080 USER@HOST` on the local computer. Use `--viewer native` when a desktop display is available on the training machine.

## Train from scratch

After setting the environment variables above, train on all 40 clips with 2048 environments:

```bash
mapfile -t BUMI_BODIES < <(python -c 'from supermimic.robots.bumi import BUMI_BODY_NAMES; print("\n".join(BUMI_BODY_NAMES))')
RUN_NAME="bumi_fresh_$(date +%Y%m%d_%H%M%S)"
python -m supermimic.scripts.train --run-name "$RUN_NAME" --log-root logs --seed 0 --device cuda:0 --data.npz-dir "$BUMI_MOTION_DIR" --env.robot-cfg-factory supermimic.robots.bumi:get_bumi_robot_cfg --env.tracked-bodies "${BUMI_BODIES[@]}" --env.anchor-body-name base_link --env.root-body-name base_link --env.fk-model "$BUMI_SCENE_XML" --env.com-body-name base_link --env.num-envs 2048 --schedule.total-iterations 320000 --schedule.checkpoint-interval 4000
```

Checkpoints are saved every 4000 iterations under `logs/$RUN_NAME/`.

## Continue training

The supplied checkpoint has already reached the original 320000-iteration target. Set a higher target, such as 400000:

```bash
RUN_NAME="bumi_resume_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/run_bumi_resume.sh "$RUN_NAME" 2048 "$PWD/logs/bumi_resume_2048_20261118_201522/ckpt_latest.pt" 400000 > "logs/${RUN_NAME}_console.log" 2>&1 < /dev/null &
echo "PID=$! RUN_NAME=$RUN_NAME"
```

Follow progress with `tail -f "logs/${RUN_NAME}_console.log"`. The next checkpoint will be under `logs/$RUN_NAME/`. For another resume, pass that run's `ckpt_latest.pt`, choose a new run name and a higher target iteration.

## TensorBoard

During or after training:

```bash
tensorboard --logdir logs --host 127.0.0.1 --port 6006
```

Open `http://localhost:6006`, or forward port 6006 from a remote machine with `ssh -N -L 6006:127.0.0.1:6006 USER@HOST`.

## Provenance and license

The method follows [SuperMimic](https://arxiv.org/abs/2610.28480). The shared training implementation derives from [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced). Bumi MJCF and configuration work also references [bumi_mujoco](https://github.com/diaskabdualiev/bumi_mujoco). Copyright and license notices for the reused code are in `THIRD_PARTY_NOTICES.md`; the robot model, retargeted LAFAN1 motions, and checkpoints have separate terms. This project does not claim G1 benchmark scores as Bumi results or report a formal Bumi benchmark result.
