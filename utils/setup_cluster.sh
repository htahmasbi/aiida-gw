#!/bin/bash
# -----------------------------------------------------------------------------
# Cluster Setup Script for aiida-relax-project — ROSI5 (FZ Rossendorf)
# -----------------------------------------------------------------------------
# Run this ON YOUR LOCAL VM where AiiDA is installed.
# It configures the remote cluster (rosi5) as an AiiDA computer + CP2K code.
#
# Usage:
#   bash setup_cluster.sh
#
# Prerequisites:
#   - AiiDA installed and verdi in PATH
#   - Passwordless SSH access to rosi5.fz-rossendorf.de
#   - verdi quicksetup already done (AiiDA profile exists)
# -----------------------------------------------------------------------------

set -euo pipefail

# ============================== CONFIGURATION ================================

# --- Computer settings ---
COMPUTER_LABEL="rosi5"
COMPUTER_HOST="rosi5.fz-rossendorf.de"
COMPUTER_DESCRIPTION="FZ Rossendorf ROSI5 cluster (SLURM)"
SCHEDULER="core.slurm"
TRANSPORT="core.ssh_async"
SSH_USER="tahmas41"
SSH_KEY="~/.ssh/id_rsa"
WORKDIR="/bigdata/casus/fwuk/tahmas41/work/aiida-runs"
MPI_PROCS_PER_MACHINE=64
NUM_MACHINES=1

# --- CP2K code settings ---
CP2K_CODE_LABEL="cp2k"
CP2K_EXEC="/data/rosi/shared/spack/turin/software/__spack_path_placeholder__/__spack_path_placeholder__/__spack_path_placeholder__/__spack_/linux-ubuntu22.04-zen4/gcc-14.3.0/cp2k-master/bin/cp2k.psmp"
CP2K_PREPEND_TEXT="ml purge
ml use /data/rosi/shared/spack/turin/modules/linux-ubuntu22.04-x86_64/Core/
ml use /data/rosi/shared/spack/turin/modules/linux-ubuntu22.04-x86_64/gcc/14.3.0/
ml use /data/rosi/shared/spack/turin/modules/linux-ubuntu22.04-x86_64/openmpi/5.0.8-x7ce7xb/gcc/14.3.0/
ml cp2k/master"

# =============================================================================

echo "=== 1. Ensuring AiiDA profile exists ==="
if ! verdi profile list &>/dev/null; then
    echo "No AiiDA profile found. Creating one..."
    verdi quicksetup
else
    echo "Profile(s) found:"
    verdi profile list
fi

echo ""
echo "=== 2. Setting up AiiDA computer: ${COMPUTER_LABEL} ==="

# Remove if re-running
verdi computer delete "${COMPUTER_LABEL}" 2>/dev/null || true

verdi computer setup --non-interactive \
    --label "${COMPUTER_LABEL}" \
    --hostname "${COMPUTER_HOST}" \
    --description "${COMPUTER_DESCRIPTION}" \
    --transport "${TRANSPORT}" \
    --scheduler "${SCHEDULER}" \
    --work-dir "${WORKDIR}" \
    --mpirun-command "mpirun" \
    --mpiprocs-per-machine "${MPI_PROCS_PER_MACHINE}"

echo ""
echo "=== 3. Configuring SSH transport for ${COMPUTER_LABEL} ==="

verdi computer configure core.ssh_async "${COMPUTER_LABEL}" --non-interactive \
    --host "${COMPUTER_HOST}"

echo ""
echo "=== 4. Testing connection to ${COMPUTER_LABEL} ==="
verdi computer test "${COMPUTER_LABEL}"

echo ""
echo "=== 5. Setting up CP2K code: ${CP2K_CODE_LABEL}@${COMPUTER_LABEL} ==="

verdi code delete "${CP2K_CODE_LABEL}@${COMPUTER_LABEL}" 2>/dev/null || true

verdi code setup --non-interactive \
    --label "${CP2K_CODE_LABEL}" \
    --computer "${COMPUTER_LABEL}" \
    --remote-abs-path "${CP2K_EXEC}" \
    --input-plugin "cp2k" \
    --prepend-text "${CP2K_PREPEND_TEXT}"

echo ""
echo "=== 6. Verifying code ==="
verdi code show "${CP2K_CODE_LABEL}@${COMPUTER_LABEL}"

echo ""
echo "============================================"
echo "  SETUP COMPLETE!"
echo "============================================"
echo ""
echo "Next steps on your VM:"
echo ""
echo "  1. Update config.toml (or set env vars):"
echo "     export ENGINE=cp2k"
echo "     export CODE_LABEL=${COMPUTER_LABEL}"
echo ""
echo "  2. Run a test calculation:"
echo "     aiida-relax run --mode single-point --engine cp2k --code ${CP2K_CODE_LABEL}@${COMPUTER_LABEL}"
echo ""
echo "  3. Or launch the MC2D batch:"
echo "     python launch_scripts/launch_mc2d_cp2k.py"
echo ""
echo "  4. Check status:"
echo "     verdi process list"
echo "     verdi process status <pk>"
