#!/bin/bash
# =============================================================================
# image/build.sh  —  build the training container image and import it to .sqsh
#
# Stages the compiled UCCL-EP into the build context, runs `podman build` on the
# Dockerfile in this directory, then converts the resulting image into the
# squashfs file that the enroot/CE runtime mounts (see IMAGE_ENV in paths.sh).
#
# Usage:
#   sbatch build.sh <tag>          # batch build on a compute node (usual path)
#   bash   build.sh <tag>          # same script, run by hand on a node you hold
#
#   UCCL_MODE=compile sbatch build.sh <tag>   # build UCCL from source in the
#                                             # image instead of reusing
#                                             # $UCCL_PREBUILT (see Dockerfile)
#
# Extra args go straight to `podman build`, e.g. to move the UCCL source pin:
#   sbatch build.sh <tag> --build-arg UCCL_REF=<sha>
#
# Grab an interactive node first if you want the second form:
#   srun -A infra01 --partition=normal --pty bash
#   (add --reservation=<name> when a reservation is active)
#
# After the build, point the .toml env file at the new .sqsh, e.g.
#   /iopsstor/scratch/cscs/gfu/ce-images/alps-pytorch2512.toml
# =============================================================================
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=72
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=build_%j.log
#SBATCH --error=build_%j.err

set -euo pipefail

: "${TAG:=${1:-}}"                                              # image tag == output .sqsh basename
: "${SQSH_DIR:=/iopsstor/scratch/cscs/gfu/ce-images}"           # where the runtime picks images up
# Build context (this directory). Hardcoded because SLURM copies the submitted
# script to its spool dir, so $0 / BASH_SOURCE point at /var/spool/... under sbatch.
: "${IMAGE_DIR:=/capstor/scratch/cscs/gfu/frameworks/myscripts/image}"

if [[ -z "$TAG" ]]; then
    echo "usage: [sbatch|bash] $0 <tag>" >&2
    exit 1
fi

cd "$IMAGE_DIR"

OUT="$SQSH_DIR/$TAG.sqsh"
if [[ -e "$OUT" ]]; then
    echo "error: $OUT already exists — remove it or pick another tag" >&2
    exit 1
fi

echo "==> podman build -t $TAG ${*:2} ."
podman build -t "$TAG" "${@:2}" .

echo "==> enroot import -> $OUT"
enroot import -x mount -o "$OUT" "podman://$TAG"

echo "==> done: $OUT"
