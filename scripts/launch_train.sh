#!/usr/bin/env bash
set -euo pipefail

# --- UX Helpers ---
if [[ -t 1 ]]; then
  BOLD=$(tput bold); CYAN=$(tput setaf 6); BLUE=$(tput setaf 4); RED=$(tput setaf 1); RESET=$(tput sgr0)
else
  BOLD=""; CYAN=""; BLUE=""; RED=""; RESET=""
fi
status_line() { printf "${BLUE}▶${RESET} ${BOLD}%-20s${RESET} %s\n" "$1:" "${2:-}"; }
rule()        { printf "${BLUE}%0.s-${RESET}" {1..50}; printf "\n"; }

# --- Defaults ---
export SLURM_PARTITION="gpu"
export SLURM_JOB_NAME="ldm-train"
export TIME_LIMIT="72:00:00"
export GPUS="1"
export ENV_NAME="compvis_ldm"  # Matches your environment.yaml

# Arrays to hold arguments for Python
PYTHON_ARGS=()

# --- Argument Parsing ---
while [[ $# -gt 0 ]]; do
  case $1 in
    # Infrastructure Args (Consumed here)
    --partition) export SLURM_PARTITION="$2"; shift 2 ;;
    --time)      export TIME_LIMIT="$2"; shift 2 ;;
    --gpus)      export GPUS="$2"; shift 2 ;;
    --name)      export SLURM_JOB_NAME="$2"; PYTHON_ARGS+=("-n" "$2"); shift 2 ;; # Pass name to python too
    --wandb_project) export WANDB_PROJECT="$2"; PYTHON_ARGS+=("--wandb_project" "$2"); shift 2 ;;
    --wandb_tags)    export WANDB_TAGS="$2";    PYTHON_ARGS+=("--wandb_tags" "$2"); shift 2 ;;
    --wandb_id)      export WANDB_ID="$2";      PYTHON_ARGS+=("--wandb_id" "$2"); shift 2 ;;
    # Pass-through everything else to Python
    *)           PYTHON_ARGS+=("$1"); shift ;;
  esac
done

# --- Git Metadata ---
export GIT_HASH=$(git rev-parse --short HEAD)
export GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
export GIT_IS_DIRTY=$(git diff --quiet || echo "-dirty")

# --- Submission ---
rule
status_line "🚀 Submitting Job" "$SLURM_JOB_NAME"
status_line "🔧 Partition"      "$SLURM_PARTITION"
status_line "⏱️  Time Limit"     "$TIME_LIMIT"
status_line "🐍 Environment"    "$ENV_NAME"
status_line "📎 Git Commit"     "${GIT_HASH}${GIT_IS_DIRTY} ($GIT_BRANCH)"
rule

# Submit to SLURM
# Note: We pass Python args as distinct arguments to the SLURM script
JOB_ID=$(sbatch \
  --partition="$SLURM_PARTITION" \
  --job-name="$SLURM_JOB_NAME" \
  --time="$TIME_LIMIT" \
  --gres=gpu:"$GPUS" \
  --export=ALL \
  scripts/train.slurm "${PYTHON_ARGS[@]}" | awk '{print $4}')

status_line "🎉 Submitted" "Job ID: $JOB_ID"
status_line "📝 Logs at"   "logs/${SLURM_JOB_NAME}-${JOB_ID}.out"

# Runs on 'gpu' partition, 1 GPU, 24h limit (defaults)
#./scripts/launch_train.sh \
#    -b configs/latent-diffusion/cin256-v2.yaml \
#    -t \
#    --gpus 0,


# Request specific partition/time, override batch size and max epochs
#./scripts/launch_train.sh \
#  -b configs/chest_xray_ldm.yaml \
#  -t \
#  --gpus 0 \
#  --wandb_project "compvis" \
#  --wandb_tags "experiment-1,TB,high-res"