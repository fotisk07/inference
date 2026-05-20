#!/usr/bin/env bash
set -e

if [ "$#" -lt 2 ]; then
  echo "Usage:"
  echo "  $0 <prod10|prod20|prod40|prod80> <command> [args...]"
  exit 1
fi

PARTITION="$1"
shift
CMD=("$@")


JOB_NAME=$(basename "${CMD[0]}")

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=08:00:00
#SBATCH --output=${LOG_DIR}/%x_%j.out
#SBATCH --error=${LOG_DIR}/%x_%j.err

set -e
echo "Running on partition: ${PARTITION}"
echo "GPU spec: ${GPU_SPEC}"
echo "Command: ${CMD[@]}"

srun ${CMD[@]}
EOF
