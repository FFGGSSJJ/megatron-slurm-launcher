#!/bin/bash
# =============================================================================
# image/build.sh  —  build the training container image and import it to .sqsh
#
# Runs `podman build` on the Dockerfile in this directory (base image + UCCL-EP
# + the local wheels), then converts the result into the squashfs file that the
# enroot/CE runtime mounts (see IMAGE_ENV in common/paths.sh).
#
# Usage:
#   sbatch build.sh <tag>          # batch build on a compute node (usual path)
#   bash   build.sh <tag>          # same script, run by hand on a node you hold
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
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --output=build_%j.log
#SBATCH --error=build_%j.err

set -euo pipefail

: "${TAG:=${1:-}}"                                              # image tag == output .sqsh basename
: "${SQSH_DIR:=/iopsstor/scratch/cscs/gfu/ce-images}"           # where the runtime picks images up
# Build context (this directory). Hardcoded because SLURM copies the submitted
# script to its spool dir, so $0 / BASH_SOURCE point at /var/spool/... under sbatch.
: "${IMAGE_DIR:=/capstor/scratch/cscs/gfu/frameworks/myscripts/image}"
# mksquashfs options for `enroot import` (no CLI flag for these). The site default
# in /etc/enroot/enroot.conf is "-comp zstd -Xcompression-level 1 -noD", where
# -noD leaves file data uncompressed: fast to read, ~27G. Compressing the data
# halves that, matching the _gzip.sqsh the training .toml uses.
# gzip's default level is 9, which costs a lot of time for nothing — measured on
# 4G of /usr/lib64, 8 procs: L1 6.5s/1.7G, L4 10s/1.6G, L6 19s/1.6G, L9 70s/1.6G.
# L4 is ~7x faster than the default at the same size.
export ENROOT_SQUASH_OPTIONS="${ENROOT_SQUASH_OPTIONS:--comp gzip -Xcompression-level 4 -b 1M -exit-on-error}"

if [[ -z "$TAG" ]]; then
    echo "usage: [sbatch|bash] build.sh <tag> [podman build args...]" >&2
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

echo "==> enroot import -> $OUT  [$ENROOT_SQUASH_OPTIONS]"
enroot import -x mount -o "$OUT" "podman://$TAG"

echo "==> done: $OUT"
