#!/usr/bin/env bash

set -euo pipefail

# Unified runner
# Usage example:
#   bash experiments/scripts/pipelines/run_experiment.sh \
#     --gpu 0 --agent sac --env antmaze-umaze-diverse-v0 \
#     --use_redq true --exp_name wsrl_from_calql --auto_resume_calql true

GPU_ID=0
AGENT="calql"           # calql | sac | closed_loop_sac | cql | iql
ENV_ID="antmaze-umaze-diverse-v0"
SEED=0

# Training defaults
NUM_OFFLINE_STEPS=1000000
NUM_ONLINE_STEPS=500000
UTD=4
BATCH_SIZE=1024
WARMUP_STEPS=5000
USE_REDQ="false"
EXP_NAME=""
SAVE_ROOT="/media/nudt3090/XYQ/ZCX/WSRL/wsrl_log"
RESUME_PATH=""
AUTO_RESUME_CALQL="false"
CALQL_PREFIX="calql_ensemble_highutd"  # prefix of CALQL runs for auto-resume

# Optional manual reward settings (auto by env if empty)
R_SCALE=""
R_BIAS=""

print_usage() {
  cat <<EOF
Usage: $0 [--gpu N] [--agent X] [--env ENV_ID] [--seed N] \
          [--num_offline_steps N] [--num_online_steps N] [--utd N] \
          [--batch_size N] [--warmup_steps N] \
          [--use_redq true|false] [--exp_name NAME] [--save_root PATH] \
          [--resume_path CKPT] [--auto_resume_calql true|false] [--calql_prefix STR] \
          [--reward_scale F] [--reward_bias F]
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU_ID="$2"; shift 2;;
    --agent) AGENT="$2"; shift 2;;
    --env) ENV_ID="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --num_offline_steps) NUM_OFFLINE_STEPS="$2"; shift 2;;
    --num_online_steps) NUM_ONLINE_STEPS="$2"; shift 2;;
    --utd) UTD="$2"; shift 2;;
    --batch_size) BATCH_SIZE="$2"; shift 2;;
    --warmup_steps) WARMUP_STEPS="$2"; shift 2;;
    --use_redq) USE_REDQ="$2"; shift 2;;
    --exp_name) EXP_NAME="$2"; shift 2;;
    --save_root) SAVE_ROOT="$2"; shift 2;;
    --resume_path) RESUME_PATH="$2"; shift 2;;
    --auto_resume_calql) AUTO_RESUME_CALQL="$2"; shift 2;;
    --calql_prefix) CALQL_PREFIX="$2"; shift 2;;
    --reward_scale) R_SCALE="$2"; shift 2;;
    --reward_bias) R_BIAS="$2"; shift 2;;
    -h|--help) print_usage; exit 0;;
    --*) EXTRA_ARGS+=("$1"); shift;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export D4RL_DATASET_DIR=/media/nudt3090/XYQ/ZCX/WSRL/datasets/d4rl
export WANDB_BASE_URL=https://api.bandw.top
export JAX_TRACEBACK_FILTERING=off

# Detect env type and defaults
CONFIG_SUFFIX=""
case "${AGENT}" in
  calql|cql) CONFIG_SUFFIX="cql";;
  sac) CONFIG_SUFFIX="wsrl";;
  closed_loop_sac) CONFIG_SUFFIX="closed_loop_sac";;
  iql) CONFIG_SUFFIX="iql";;
  *) echo "Unsupported agent: ${AGENT}"; exit 1;;
esac

CONFIG_PREFIX=""
if [[ "${ENV_ID}" == antmaze-* ]]; then
  CONFIG_PREFIX="antmaze"
  RS_DEFAULT=10.0; RB_DEFAULT=-5.0
elif [[ "${ENV_ID}" == kitchen-* ]]; then
  CONFIG_PREFIX="kitchen"
  RS_DEFAULT=1.0; RB_DEFAULT=-4.0
elif [[ "${ENV_ID}" == *halfcheetah* || "${ENV_ID}" == *hopper* || "${ENV_ID}" == *walker* ]]; then
  CONFIG_PREFIX="locomotion"
  RS_DEFAULT=1.0; RB_DEFAULT=0.0
elif [[ "${ENV_ID}" == pen-binary-v0 || "${ENV_ID}" == door-binary-v0 || "${ENV_ID}" == relocate-binary-v0 ]]; then
  CONFIG_PREFIX="adroit"
  RS_DEFAULT=10.0; RB_DEFAULT=5.0
  export DATA_DIR_PREFIX=/media/nudt3090/XYQ/ZCX/WSRL/datasets/adroit_data
else
  echo "Unknown env: ${ENV_ID}"; exit 1
fi

CONFIG_KEY="${CONFIG_PREFIX}_${CONFIG_SUFFIX}"

# Reward defaults unless overridden
if [[ -z "${R_SCALE}" ]]; then R_SCALE=${RS_DEFAULT}; fi
if [[ -z "${R_BIAS}" ]]; then R_BIAS=${RB_DEFAULT}; fi

# Auto-resume from latest CALQL checkpoint if requested and not explicitly set
if [[ "${AUTO_RESUME_CALQL}" == true && -z "${RESUME_PATH}" ]]; then
  EXP_DESC="${CALQL_PREFIX}_${ENV_ID}_calql_seed${SEED}"
  RUN_DIR=$(ls -1dt ${SAVE_ROOT}/wsrl/${EXP_DESC}_* 2>/dev/null | head -n 1 || true)
  if [[ -n "${RUN_DIR}" && -f "${RUN_DIR}/checkpoint_1000000" ]]; then
    RESUME_PATH="${RUN_DIR}/checkpoint_1000000"
    echo "Auto-resume from: ${RESUME_PATH}"
  else
    echo "WARN: No CALQL checkpoint found for ${EXP_DESC}, proceeding without resume." >&2
  fi
fi

CMD=(python3 finetune.py \
  --agent ${AGENT} \
  --config experiments/configs/train_config.py:${CONFIG_KEY} \
  --env ${ENV_ID} \
  --seed ${SEED} \
  --reward_scale ${R_SCALE} \
  --reward_bias ${R_BIAS} \
  --num_offline_steps ${NUM_OFFLINE_STEPS} \
  --num_online_steps ${NUM_ONLINE_STEPS} \
  --utd ${UTD} \
  --batch_size ${BATCH_SIZE} \
  --warmup_steps ${WARMUP_STEPS} \
  --save_dir ${SAVE_ROOT})

# Auto exp_name if not provided
if [[ -z "${EXP_NAME}" ]]; then
  # base by agent
  case "${AGENT}" in
    calql) EXP_NAME="calql";;
    sac) EXP_NAME="wsrl";;
    closed_loop_sac) EXP_NAME="clsac";;
    cql) EXP_NAME="cql";;
    iql) EXP_NAME="iql";;
    *) EXP_NAME="run";;
  esac
  # REDQ/ensemble suffix
  if [[ "${USE_REDQ}" == true ]]; then
    EXP_NAME+="_ensemble"
  fi
  # UTD suffix (highlight high UTD for calql; otherwise explicit utd)
  if [[ "${AGENT}" == "calql" && ${UTD} -ge 20 ]]; then
    EXP_NAME+="_highutd"
  elif [[ ${UTD} -gt 1 ]]; then
    EXP_NAME+="_utd${UTD}"
  fi
  # resume suffix if continuing from CALQL
  if [[ -n "${RESUME_PATH}" || "${AUTO_RESUME_CALQL}" == true ]]; then
    EXP_NAME+="_fromcalql"
  fi
fi

if [[ "${USE_REDQ}" == true ]]; then
  CMD+=(--use_redq True)
fi
if [[ -n "${RESUME_PATH}" ]]; then
  CMD+=(--resume_path "${RESUME_PATH}")
fi
CMD+=(--exp_name "${EXP_NAME}")

# Append any extra passthrough args (e.g., --config.agent_kwargs.* overrides)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

LOG_DIR="${SAVE_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/${AGENT}_${ENV_ID}_seed${SEED}.log"

echo "Running: ${CMD[*]}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"


