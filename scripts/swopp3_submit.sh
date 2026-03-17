#!/bin/bash
# Submit the SWOPP3 pipeline: stage → atlantic + pacific (parallel)
#
# Usage:  bash scripts/swopp3_submit.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. Stage repo + data onto /scratch
STAGE_ID=$(sbatch --parsable "$SCRIPT_DIR/swopp3_slurm_stage.sh")
echo "Submitted staging job: $STAGE_ID"

# 2. Atlantic and Pacific run after staging completes
ATL_ID=$(sbatch --parsable --dependency=afterok:"$STAGE_ID" "$SCRIPT_DIR/swopp3_slurm_atlantic.sh")
echo "Submitted Atlantic job: $ATL_ID  (depends on $STAGE_ID)"

PAC_ID=$(sbatch --parsable --dependency=afterok:"$STAGE_ID" "$SCRIPT_DIR/swopp3_slurm_pacific.sh")
echo "Submitted Pacific job:  $PAC_ID  (depends on $STAGE_ID)"

echo ""
echo "Monitor with:  squeue -u $USER"
