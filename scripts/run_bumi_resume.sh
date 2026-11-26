#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
root="$(cd "$repo/.." && pwd)"
motion_dir="${BUMI_MOTION_DIR:-$root/bumi_work/motions/supermimic_bumi_lafan1_40_fixed}"
scene_xml="${BUMI_SCENE_XML:-$root/bumi_work/source/bumi_mujoco-main/bumi_robot/scene.xml}"

run_name=${1:?Usage: scripts/run_bumi_resume.sh RUN_NAME NUM_ENVS CHECKPOINT TOTAL_ITERATIONS}
num_envs=${2:-2048}
checkpoint=${3:?Pass the checkpoint path as the third argument.}
total_iterations=${4:-400000}
[[ $num_envs =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_ENVS must be positive.' >&2; exit 1; }
[[ $total_iterations =~ ^[1-9][0-9]*$ ]] || { echo 'TOTAL_ITERATIONS must be positive.' >&2; exit 1; }

cd "$repo"
[[ -f .venv/bin/activate ]] || { echo 'Run uv sync first.' >&2; exit 1; }
source .venv/bin/activate
export PYTHONPATH="$repo/src"
export BUMI_SCENE_XML="$scene_xml"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$root/.cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$root/.cache/torchinductor}"

for required in "$scene_xml" "$checkpoint"; do
    [[ -f "$required" ]] || { printf 'Missing file: %s\n' "$required" >&2; exit 1; }
done
[[ ! -e "logs/$run_name" ]] || { printf 'Run directory exists: logs/%s\n' "$run_name" >&2; exit 1; }

mapfile -t tracked_bodies < <(python -c 'from supermimic.robots.bumi import BUMI_BODY_NAMES; print("\n".join(BUMI_BODY_NAMES))')
motion_count=$(find "$motion_dir" -maxdepth 1 -type f -name '*.npz' | wc -l)
printf 'tracked_bodies=%s npz=%s\n' "${#tracked_bodies[@]}" "$motion_count"
[[ ${#tracked_bodies[@]} -eq 22 && $motion_count -eq 40 ]] || { echo 'Expected 22 bodies and 40 motions.' >&2; exit 1; }

exec python -m supermimic.scripts.train \
  --run-name "$run_name" --log-root logs --seed 0 --device cuda:0 \
  --data.npz-dir "$motion_dir" \
  --env.robot-cfg-factory supermimic.robots.bumi:get_bumi_robot_cfg \
  --env.tracked-bodies "${tracked_bodies[@]}" --env.anchor-body-name base_link \
  --env.root-body-name base_link --env.fk-model "$scene_xml" --env.com-body-name base_link \
  --env.num-envs "$num_envs" --resume "$checkpoint" \
  --schedule.total-iterations "$total_iterations" --schedule.checkpoint-interval 4000
