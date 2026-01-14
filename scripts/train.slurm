#!/bin/bash
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

set -euo pipefail

# --- UX Helpers (duplicated for log readability) ---
CYN=$(printf '\033[36m'); BLU=$(printf '\033[34m'); BLD=$(printf '\033[1m'); RST=$(printf '\033[0m')
banner(){ printf "\n${BLU}${BLD}== %s ==${RST}\n" "$*"; }
kv(){ printf "  ${CYN}%-22s${RST} %s\n" "$1" "$2"; }

# --- Environment Setup ---
banner "Environment Setup"
source ~/.bashrc  # Or path to your conda/mamba init script
# Explicitly activate using mamba or conda
mamba activate "$ENV_NAME" || conda activate "$ENV_NAME"

kv "Node"        "$(hostname)"
kv "Python"      "$(which python)"
kv "Workdir"     "$(pwd)"
kv "Git Hash"    "${GIT_HASH:-unknown}"

# --- Execution ---
banner "Starting Training"

# $@ contains all arguments passed from launch_train.sh (the PYTHON_ARGS)
# We execute main.py located in the parent directory (assuming script is in scripts/)
python main.py "$@"

banner "Job Complete"