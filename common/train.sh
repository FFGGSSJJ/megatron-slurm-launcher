#!/bin/bash
# =============================================================================
# common/train.sh  —  Megatron-LM launch ORCHESTRATOR
#
# The engine is split into three sourced files under common/:
#   model.sh         model & optimizer: arch, attention, optimizer, numerics,
#                    LR schedule, init, tokenizer, MoE model/router args.
#   engine.sh        parallelism & performance: TP/PP/EP/ETP/CP + VPP, MoE perf
#                    optimizations (dispatcher, offloading, fp8 kernels), recompute.
#   train.sh (this)  glue: model env, datasets, naming/dirs, logging/checkpoint
#                    args, env, wandb, profiler, compute-env snapshot, srun.
#
# DO NOT submit this file with sbatch. Instead write a thin experiment script
# (see launch/*.sh) that:
#   1. Carries its own #SBATCH header. nodes / job-name / time / mem are
#      per-run and SLURM requires them literal at the top of the submitted file.
#   2. Picks a model:        MODEL_ENV=moe_117b_a11b_0   (a models/*.env basename)
#   3. Overrides only the knobs it needs. Model/optimizer knobs are documented
#      at the top of model.sh; parallel/perf knobs at the top of engine.sh;
#      orchestration knobs in the DEFAULTS block below.
#   4. Ends with:            source $SCRIPTS_ROOT/common/train.sh
# =============================================================================

# ---- Locate ourselves (works when sourced under sbatch) ---------------------
ENGINE_PATH="${BASH_SOURCE[0]}"
COMMON_DIR="$(cd "$(dirname "$ENGINE_PATH")" && pwd)"
SCRIPTS_ROOT="$(cd "$COMMON_DIR/.." && pwd)"

# ---- Environment-specific paths (edit common/paths.sh, not here) -----------
source "$COMMON_DIR/paths.sh"

# ---- Model architecture -----------------------------------------------------
# The experiment script must set MODEL_ENV to a models/*.env basename.
: "${MODEL_ENV:?Set MODEL_ENV to a models/*.env basename, e.g. MODEL_ENV=moe_117b_a11b_0}"
MODEL_ENV_FILE="$SCRIPTS_ROOT/models/$MODEL_ENV.env"
[ -f "$MODEL_ENV_FILE" ] || { echo "Model env not found: $MODEL_ENV_FILE" >&2; exit 1; }
source "$MODEL_ENV_FILE"

# nsys/tmp scratch
export TMPDIR=${SLURM_TMPDIR:-/tmp}
export NSYS_TMPDIR=$TMPDIR
mkdir -p "${TMPDIR}"

echo "START TIME: $(date)"
echo "Model: $MODEL_NAME"
DATE=$(date +%Y-%m-%d)

# =============================================================================
# DEFAULTS  —  orchestration knobs (model/optimizer knobs live in model.sh,
# parallel/perf knobs in engine.sh). `:=` only assigns when the experiment
# left it unset, so an experiment overrides a knob by assigning it beforehand.
# =============================================================================

# -- Vocabulary --
# The single switch for "which token id space is this run in". It picks BOTH the
# tokenizer and the matching pre-tokenized corpus, because the two are only
# meaningful together: shards are just token ids, so reading 200k ids through a
# 130k vocab does not crash, it silently trains on nonsense.
#
#   130k  Apertus-8B-2509 (131,072 + 1,000 added) + climbmix   — the long-standing setup
#   200k  apertus preliminary_mul_200k (200,064 + 124)         — ported from _research
#
: "${VOCAB:=130k}"                      # 130k | 200k
# VOCAB_NOMINAL is the base vocab (excluding added tokens, matching the
# convention in tools/flops_calculator.py) and is used ONLY for the model-size /
# FLOPs summary. It is deliberately not fed to Megatron as --vocab-size: leaving
# VOCAB_SIZE unset lets Megatron derive the true length from the tokenizer and
# pad it itself.
case "$VOCAB" in
	130k) VOCAB_TOKENIZER=$TOKENIZER_DIR/Apertus-8B-2509
	      VOCAB_NOMINAL=131072
	      : "${DATASET_NAME:=climbmix}" ;;
	200k) VOCAB_TOKENIZER=$TOKENIZER_DIR/apertus-mul-200k
	      VOCAB_NOMINAL=200064
	      : "${DATASET_NAME:=fineweb2hq-mul200k}" ;;
	*)    echo "[$(date)] ERROR: VOCAB=$VOCAB — expected 130k or 200k" >&2; exit 1 ;;
esac
# Assigned, not `:=`-defaulted: every launch/*.sh pins TOKENIZER_MODEL outright
# and is sourced BEFORE this file, so a default would never win and the switch
# would silently do nothing. VOCAB owns the tokenizer; DATASET_NAME stays a
# default above so you can still pick another corpus in the same id space
# (e.g. VOCAB=200k DATASET_NAME=swissai-blend).
TOKENIZER_MODEL=$VOCAB_TOKENIZER

# -- Data --
# climbmix | fineweb-edu-100B | fineweb2hq-mul200k | swissai-blend
: "${MOCK_DATA:=false}"
# (DATASET_CACHE_DIR comes from common/paths.sh)
# Blend sources for the globbed presets. Set DATA_ROOT/DATA_SOURCES directly to
# glob an arbitrary blend without adding a DATASET_NAME arm; whatever the preset
# below would have filled in is then left alone.
: "${DATA_ROOT:=}"
: "${DATA_SOURCES:=}"
# A pre-built blend file (tools/make_data_blend.py) handed straight to Megatron
# as --data-args-path. Use it when the glob's token-proportional-over-everything
# blend is not what you want: dropping subsets, reweighting them, or cutting the
# shard count down for a smoke test. It wins over DATASET_NAME/DATA_SOURCES, so
# no preset below fires and nothing gets globbed.
# DATA_BLEND_VOCAB declares which id space the file's shards were tokenized in
# (130k | 200k) so the vocab guard still applies; empty disables that check.
: "${DATA_BLEND_FILE:=}"
: "${DATA_BLEND_VOCAB:=}"
if [ -n "$DATA_BLEND_FILE" ]; then
	[ -f "$DATA_BLEND_FILE" ] || { echo "[$(date)] ERROR: DATA_BLEND_FILE not found: $DATA_BLEND_FILE" >&2; exit 1; }
	DATASET_NAME=blend-file
fi
# A blend of thousands of shards overflows the srun command line (E2BIG,
# "Argument list too long"). Above this many shards the list is written to a
# file and passed via --data-args-path instead of argv; Megatron reads the same
# token-proportional blend from the file.
: "${DATA_ARGS_FILE_THRESHOLD:=256}"
: "${NUM_WORKERS:=16}"
# Cross-document attention handling. CROSS_DOC_ATTENTION=true emits the
# --reset-position-ids/--reset-attention-mask/--eod-mask-loss trio.
#
# CREATE_ATTENTION_MASK=true (Megatron's default) makes the dataloader build a
# dense [1, seq, seq] mask per sample -- at seq=8192 that is a 256 MB fp32 tril
# reduced to a 64 MB bool, rebuilt for EVERY sample (the cache is disabled as
# soon as any of the reset_*/eod_mask_loss switches is on) and copied to the GPU.
# GPT layer specs pin attn_mask_type=causal, so TE's flash path never reads it.
# Set false unless something in the model actually consumes an explicit mask.
: "${CROSS_DOC_ATTENTION:=true}"
: "${CREATE_ATTENTION_MASK:=true}"
: "${PACKING_STRATEGY:=}"               # empty = Megatron default (greedy) | bfd

# Which id space each corpus was tokenized in, so an incoherent
# VOCAB/DATASET_NAME pair is caught below rather than trained on. Both globbed
# blends are mul_200k (_research/lib/common.sh feeds every one of its blends
# through that tokenizer); climbmix is Apertus-8B-2509 per its own comment.
# fineweb-edu-100B is deliberately absent: it lives under a `llama_tokenized`
# path, so it matches NEITHER vocab and is left unguarded as it was before.
case "$DATASET_NAME" in
	fineweb2hq-mul200k|swissai-blend) DATASET_VOCAB=200k ;;
	climbmix)                         DATASET_VOCAB=130k ;;
	blend-file)                       DATASET_VOCAB=$DATA_BLEND_VOCAB ;;
	*)                                DATASET_VOCAB= ;;
esac

# -- Container / codebase --
# (IMAGE_ENV and MEGATRON_LM_DIR come from common/paths.sh)

# -- Experiment naming --
: "${PROJECT_NAME:=large_scale_moe_performance}"
: "${EXP_NAME_SUFFIX:=$DATE}"

# -- Checkpointing & resuming --
: "${CHECKPOINT_STEPS:=100000}"
: "${LOAD_CKPT:=false}"
: "${AUTO_JOB_REQUEUE:=false}"
: "${BACKUP_CODEBASE:=false}"
: "${CKPT_FORMAT:=torch_dist}"

# -- Debugging / profiling --
: "${TENSORBOARD_LOG_INTERVAL:=}"       # empty = Megatron default (1)
: "${LOG_NCCL:=false}"
: "${NSYS_PROFILER:=false}"
: "${TORCH_PROFILER:=false}"
: "${RANK_TO_PROFILE:=0}"           # global rank(s) to profile, e.g. "0" or "0 32" (commas also ok)
RANK_TO_PROFILE=${RANK_TO_PROFILE//,/ }
: "${PROFILER_START_ITER:=15}"
: "${PROFILER_END_ITER:=16}"
: "${NSYS_PROFILER_START_ITER:=5}"
: "${NSYS_PROFILER_END_ITER:=6}"

# -- Dry run --
# DRY_RUN=true assembles everything, prints the srun + training command and
# exits without creating dirs, building UCCL, writing snapshots or launching.
# Works outside an allocation too:  DRY_RUN=true bash launch/<exp>.sh
: "${DRY_RUN:=false}"
if [ "$DRY_RUN" = true ]; then
	# The experiment's #SBATCH header wins over any inherited SLURM env (running
	# the dry run inside a small salloc would otherwise misrepresent the real
	# submission's node count); the rest gets GH200 defaults when unset.
	# Snapshot the header now: $0 may be relative and train.sh cd's away later.
	_SB_HEADER=$(grep '^#SBATCH' "$0" 2>/dev/null)
	_SB_NODES=$(sed -n 's/^#SBATCH --nodes=//p' "$0" | head -1)
	[ -n "$_SB_NODES" ] && SLURM_NNODES=$_SB_NODES
	: "${SLURM_NNODES:=1}"
	: "${SLURM_GPUS_PER_NODE:=4}"
	: "${SLURM_CPUS_PER_TASK:=72}"
	SLURM_NPROCS=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
	: "${SLURM_JOB_ID:=dryrun}"
fi

# =============================================================================
# DOMAIN CONFIGS
# Order matters: engine.sh reads GBS/MBS/OPTIMIZER set by model.sh.
# =============================================================================
source "$COMMON_DIR/model.sh"
source "$COMMON_DIR/engine.sh"

# ---- Model size / FLOPs summary ----------------------------------------------
MODEL_STATS=""
if command -v python3 >/dev/null 2>&1; then
	MODEL_STATS="$(python3 "$SCRIPTS_ROOT/tools/flops_calculator.py" "$MODEL_ENV" \
		--seq-len "$SEQ_LEN" --gbs "$GBS" --attention-type "$ATTENTION_TYPE" \
		--vocab-size "${VOCAB_SIZE:-$VOCAB_NOMINAL}" 2>&1)" || \
		echo "[$(date)] flops_calculator failed (non-fatal)" >&2
	echo "$MODEL_STATS"
fi

# =============================================================================
# DATASETS
# =============================================================================
if [ "$DATASET_NAME" == "fineweb-edu-100B" ]; then
	DATASETS="$FINEWEB_DIR/fineweb-edu-100B_00002_tokens 1.0 $FINEWEB_DIR/fineweb-edu-100B_00000_tokens 1.0 $FINEWEB_DIR/fineweb-edu-100B_00001_tokens"
fi

# nemotron-climbmix: hftokenizer + swiss-ai/Apertus-8B-2509 vocab
if [ "$DATASET_NAME" == "climbmix" ]; then
	NUM_FILES=100
	for (( i=0; i<$NUM_FILES; i++ ))
	do
		DATASETS+="$DATASET_DIR/climbmix/hftokenized/part_${i}_text_document "
	done
fi

# ---- Globbed blends (ported from Megatron-LM/_research) ---------------------
# These name SOURCE DIRECTORIES rather than shard prefixes; the glob below turns
# them into a flat prefix list. Each preset only fills what is still unset, so
# DATA_ROOT/DATA_SOURCES from the experiment win.
#
# fineweb2hq-mul200k: fineweb-2-hq mmbert quality_10, SPP-annotated, mul_200k
# tokenized (~716 shards). Only the fwedu split exists at this root — the dclm
# split has no copy here, and the sibling *_apertus_v2 dir is empty (0 shards).
if [ "$DATASET_NAME" == "fineweb2hq-mul200k" ]; then
	: "${DATA_ROOT:=$FW2HQ_DIR}"
	: "${DATA_SOURCES:=swissai-fineweb-2-hq-mmbert-full-quality_10-filterrobots-fwedu_spp_annotated}"
fi

# swissai-blend: the default swissai pretraining mixture (dclm-edu + fineweb-2
# euro-high/euro-mid/other-high).
if [ "$DATASET_NAME" == "swissai-blend" ]; then
	: "${DATA_ROOT:=$SWISSAI_DATA_DIR}"
	: "${DATA_SOURCES:=
		swissai-dclm-edu-filterrobots_fine-merge
		swissai-fineweb-2-quality_10-filterrobots-merge/euro-high
		swissai-fineweb-2-quality_10-filterrobots-merge/euro-mid
		swissai-fineweb-2-quality_10-filterrobots-merge/other-high
	}"
fi

# Glob every {bin,idx} shard under each source dir (they may be nested, e.g.
# dump-N/00000_tokens.*), strip the extension and hand Megatron the flat prefix
# list. Blend weights are inferred from shard lengths, so the mixture stays
# token-proportional across sources — do NOT add explicit weights here.
DATA_SHARDS=()
if [ -n "$DATA_SOURCES" ]; then
	for src in $DATA_SOURCES; do
		d="$DATA_ROOT/$src"
		[ -d "$d" ] || { echo "[$(date)] ERROR: data source not found: $d" >&2; exit 1; }
		while IFS= read -r p; do
			DATA_SHARDS+=("$p")
		done < <(find "$d" -type f \( -name '*.bin' -o -name '*.idx' \) \
		             | sed -E 's/\.[^.]+$//' | sort -u)
	done
	[ "${#DATA_SHARDS[@]}" -gt 0 ] || { echo "[$(date)] ERROR: no .bin/.idx shards under $DATA_ROOT" >&2; exit 1; }
	DATASETS="${DATA_SHARDS[*]}"
	echo "[$(date)] DATASET: $DATASET_NAME — ${#DATA_SHARDS[@]} shards under $DATA_ROOT"
	echo "[$(date)] TOKENIZER: $TOKENIZER_MODEL"
fi

# Vocab guard. VOCAB picks a coherent (tokenizer, corpus) pair on its own, so
# this only fires when DATASET_NAME was pointed at a corpus from the OTHER id
# space. That combination does not crash — it trains on nonsense for however
# many node-hours you booked — so stop here instead.
# ALLOW_TOKENIZER_MISMATCH=true if you really mean it.
if [ -n "$DATASET_VOCAB" ] && [ "$DATASET_VOCAB" != "$VOCAB" ] \
   && [ "${ALLOW_TOKENIZER_MISMATCH:-false}" != true ]; then
	echo "[$(date)] ERROR: VOCAB=$VOCAB selects TOKENIZER_MODEL=$TOKENIZER_MODEL," >&2
	echo "  but DATASET_NAME=$DATASET_NAME was tokenized in the $DATASET_VOCAB id space." >&2
	echo "  Its token ids do not mean the same thing in this vocab and training will be" >&2
	echo "  silently wrong. Use VOCAB=$DATASET_VOCAB, or pick a $VOCAB dataset, or set" >&2
	echo "  ALLOW_TOKENIZER_MISMATCH=true to force it." >&2
	exit 1
fi

# =============================================================================
# LOGGING DIRECTORIES & ARTIFACTS  (EXP_NAME mixes model + optimization knobs)
# =============================================================================
VPP_TAG=$([ -n "$VPP_LAYOUT" ] && echo "-vpp" || echo "")
FP8_MOE_PREFIX=$([ "$USE_FP8_MOE_PARAM" = true ] && echo "fp8moe-" || echo "")
_MEG_BRANCH_TAG="$(git -C "$MEGATRON_LM_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
_MEG_BRANCH_TAG="${_MEG_BRANCH_TAG//\//-}"   # sanitize slashes: branch name is embedded into paths/filenames below
_MEG_COMMIT_SHORT="${_MEG_BRANCH_TAG}-$(git -C "$MEGATRON_LM_DIR" rev-parse --short=8 HEAD 2>/dev/null || echo unknown)"
EXP_NAME=${FP8_MOE_PREFIX}${MODEL_NAME}-${ACTIVATION_FUNCTION}-${OPTIMIZER}-${SLURM_NNODES}n-${SEQ_LEN}sl-${GBS}gbsz-${MBS}mbsz-${LR}lr-${TP}tp-${PP}pp-${EP}ep-${ETP}etp-${CP}cp${VPP_TAG}-mockr${USE_MOCK_ROUTER}-off${USE_EXPERTS_OFFLOADING}-dbg${USE_OFFLOADING_DEBUG}-epoverlap${OVERLAP_MOE_EP_COMM}-${_MEG_COMMIT_SHORT}-${EXP_NAME_SUFFIX}
: "${LOAD_EXP_NAME:=$EXP_NAME}"
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME

EXP_DIR=$PROJECT_DIR/$EXP_NAME
SAVE_CKPT_DIR=$CKPT_BASE_DIR/$PROJECT_NAME/$EXP_NAME/checkpoints
LOAD_CKPT_DIR=$CKPT_BASE_DIR/$PROJECT_NAME/$LOAD_EXP_NAME/checkpoints
echo "[$(date)] SAVE_CKPT_DIR: $SAVE_CKPT_DIR"
echo "[$(date)] LOAD_CKPT_DIR: $LOAD_CKPT_DIR"

TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$PROJECT_DIR/$DATE/$MODEL_NAME/$SLURM_JOB_ID
COMPUTE_ENVIRONMENT_DIR=$DEBUG_DIR/compute_environment.txt
GPU_MEM_LOGGING=$DEBUG_DIR/memory_logging.txt
LOGGING_DIR=$EXP_DIR/logging
TENSORBOARD_DIR=$LOGGING_DIR/tensorboard
BACKUP_CODEBASE_DIR=$EXP_DIR/Megatron-LM

# =============================================================================
# ENVIRONMENT
# =============================================================================
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# torch.distributed wants MASTER_ADDR / MASTER_PORT / WORLD_SIZE before srun;
# RANK / LOCAL_RANK are set at the srun command.
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=25679
export WORLD_SIZE=$SLURM_NPROCS

ulimit -c 0

# =============================================================================
# ORCHESTRATOR ARGS  (logging / checkpointing / data — need the dirs above)
# =============================================================================
LOGGING_ARGS=(
	--log-throughput
	--log-progress
	--tensorboard-dir $TENSORBOARD_DIR
	# --no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
	--log-timers-to-tensorboard
	--log-params-norm
	--moe-per-layer-logging
	--log-memory-interval 100
)

if [ -n "$TENSORBOARD_LOG_INTERVAL" ]; then
	LOGGING_ARGS+=(--tensorboard-log-interval $TENSORBOARD_LOG_INTERVAL)
fi

if [ "$LOG_PP_TIME" = true ]; then
	LOGGING_ARGS+=(
		--log-pp-stage-timing
	)
fi

if [ "$LOAD_CKPT" = true ]; then
	CHECKPOINTING_ARGS=(
		--save $SAVE_CKPT_DIR
		--save-interval $CHECKPOINT_STEPS
		--ckpt-format $CKPT_FORMAT
		--load $LOAD_CKPT_DIR
		# --async-save
	)
else
	# If not loading from checkpoint, start fresh and ignore existing checkpoints
	CHECKPOINTING_ARGS=(
		--save $SAVE_CKPT_DIR
		--save-interval $CHECKPOINT_STEPS
		--ckpt-format $CKPT_FORMAT
		# --async-save
	)
fi

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
)

if [ "$CROSS_DOC_ATTENTION" = true ]; then
	DATA_ARGS+=(
		--reset-position-ids  # crossDocAttn
		--reset-attention-mask  # crossDocAttn
		--eod-mask-loss  # crossDocAttn
	)
fi

if [ "$CREATE_ATTENTION_MASK" != true ]; then
	DATA_ARGS+=(--no-create-attention-mask-in-dataloader)
fi

if [ -n "$PACKING_STRATEGY" ]; then
	DATA_ARGS+=(--pretraining-packing-strategy $PACKING_STRATEGY)
fi

DATA_ARGS+=(
	--num-workers $NUM_WORKERS
	--num-dataset-builder-threads 1
	# --goldfish-loss  # goldfish
	# --goldfish-k 50  # goldfish
	# --goldfish-h 50  # goldfish
)

# =============================================================================
# DIRECTORIES
# =============================================================================
if [ "$DRY_RUN" != true ]; then
	mkdir -p $LOAD_CKPT_DIR
	mkdir -p $SAVE_CKPT_DIR
	mkdir -p $PROJECT_DIR
	mkdir -p $TRIGGER_DIR
	mkdir -p $DEBUG_DIR
	mkdir -p $LOGGING_DIR
fi

# Backup codebase
if [ "$BACKUP_CODEBASE" == true ]; then
  if [ "$DRY_RUN" != true ] && [ -z "$(ls -A "$BACKUP_CODEBASE_DIR" 2>/dev/null)" ]; then
  	echo "[$(date)] Copying codebase in $MEGATRON_LM_DIR to $BACKUP_CODEBASE_DIR..."
  	rsync -av --exclude-from=$MEGATRON_LM_DIR/.gitignore $MEGATRON_LM_DIR/ $BACKUP_CODEBASE_DIR/ &> /dev/null
  fi
  MEGATRON_LM_DIR=$BACKUP_CODEBASE_DIR
fi

# ---- Log Megatron-LM git info ------------------------------------------------
_MEG_BRANCH="$(git -C "$MEGATRON_LM_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
_MEG_COMMIT="$(git -C "$MEGATRON_LM_DIR" rev-parse --short=12 HEAD 2>/dev/null || echo unknown)"
_MEG_DIRTY=""
if [ -n "$(git -C "$MEGATRON_LM_DIR" status --porcelain 2>/dev/null)" ]; then
	_MEG_DIRTY=" (dirty)"
fi
echo "[$(date)] Using codebase in $MEGATRON_LM_DIR  branch=$_MEG_BRANCH  commit=$_MEG_COMMIT$_MEG_DIRTY"

cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# ---- UCCL flex/DeepEP dispatcher: shared build + expose on PYTHONPATH --------
# Built once into UCCL_INSTALL_TARGET and reused by later jobs; set
# RUN_UCCL_INSTALL=true to force a rebuild (see common/uccl.sh).
# The wrapper's deep_ep must shadow the container's pre-installed stock deep_ep
# (/opt/venv/.../deep_ep), otherwise Buffer() runs stock DeepEP's NVLink-P2P
# check which fails on Alps GH200. Host PYTHONPATH does NOT reliably reach the
# pyxis container (training only finds Megatron because pretrain_gpt.py sits on
# sys.path[0]), so we ALSO re-export it *inside* the srun step via
# UCCL_ENV_PREFIX to guarantee the wrapper precedes site-packages on sys.path.
UCCL_ENV_PREFIX=""
if [ "$USE_UCCL" = true ]; then
	source "$COMMON_DIR/uccl.sh"
	# if [ "$DRY_RUN" != true ] && { [ "$RUN_UCCL_INSTALL" = true ] || [ ! -d "$UCCL_INSTALL_TARGET" ]; }; then
	# 	install_uccl || { echo "UCCL install failed" >&2; exit 1; }
	# fi
	# export PYTHONPATH="$UCCL_PYTHONPATH:$PYTHONPATH"
	# UCCL_ENV_PREFIX="export PYTHONPATH=$UCCL_PYTHONPATH:\$PYTHONPATH;"
	# echo "[$(date)] UCCL on PYTHONPATH: $UCCL_PYTHONPATH"
fi

# Data Args
if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS="${DATA_ARGS[@]} --mock-data"
elif [ -n "$DATA_BLEND_FILE" ]; then
  # Already a Megatron blend list (optionally weighted); hand it over untouched.
  echo "[$(date)] DATASET: pre-built blend -> --data-args-path $DATA_BLEND_FILE"
  DATA_ARGS="${DATA_ARGS[@]} --data-args-path $DATA_BLEND_FILE --data-cache-path $DATASET_CACHE_DIR"
elif [ "${#DATA_SHARDS[@]}" -gt "$DATA_ARGS_FILE_THRESHOLD" ]; then
  # Too many shards to fit on the command line: hand Megatron a file instead.
  # --data-args-path is mutually exclusive with --data-path (arguments.py), so
  # this replaces it rather than adding to it. Same blend either way.
  DATA_ARGS_FILE="$DATASET_CACHE_DIR/train_data_paths-${SLURM_JOB_ID:-local}.txt"
  if [ "$DRY_RUN" != true ]; then
    mkdir -p "$DATASET_CACHE_DIR"
    printf '%s\n' "${DATA_SHARDS[@]}" > "$DATA_ARGS_FILE"
  fi
  echo "[$(date)] DATASET: ${#DATA_SHARDS[@]} shards > $DATA_ARGS_FILE_THRESHOLD -> --data-args-path $DATA_ARGS_FILE"
  DATA_ARGS="${DATA_ARGS[@]} --data-args-path $DATA_ARGS_FILE --data-cache-path $DATASET_CACHE_DIR"
else
  DATA_ARGS="${DATA_ARGS[@]} --data-path $DATASETS --data-cache-path $DATASET_CACHE_DIR"
fi

CMD_PREFIX="numactl --membind=0-3"

TRAINING_CMD="python3 $MEGATRON_LM_DIR/pretrain_gpt.py \
    ${TRANSFORMER_ENGINE_ARGS[@]} \
    ${NETWORK_SIZE_ARGS[@]} \
    ${LOGGING_ARGS[@]} \
    ${REGULARIZATION_ARGS[@]} \
    ${RECOMPUTE_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${INITIALIZATION_ARGS[@]} \
    ${LEARNING_RATE_ARGS[@]} \
    ${CHECKPOINTING_ARGS[@]} \
    ${MIXED_PRECISION_ARGS[@]} \
    ${DISTRIBUTED_ARGS[@]} \
    ${TOKENIZER_ARGS[@]} \
	${OPTIMIZATION_ARGS[@]} \
	${MOE_ARGS[@]} \
	${MLA_ARGS[@]} \
    $DATA_ARGS"

# =============================================================================
# WANDB
# =============================================================================
export TRANSFORMERS_NO_SLOW_TOKENIZER=1

if [ -n "$WANDB_API_KEY" ]; then
  echo "[$(date)] WANDB API key detected. Enabling WANDB logging."
  if [ "$DRY_RUN" != true ] && [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
    echo "[$(date)] Syncing WANDB from previous run"
    wandb sync "$LOGGING_DIR/wandb/latest-run"
  fi
  TRAINING_CMD="$TRAINING_CMD \
    --wandb-save-dir $LOGGING_DIR \
    --wandb-project $PROJECT_NAME \
    --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
  export WANDB_MODE=disabled
  echo "[$(date)] No WANDB API key found. WANDB logging disabled."
fi

# =============================================================================
# PROFILING
# =============================================================================
# NCCL Debug
if [ "$LOG_NCCL" = true ]; then
  CMD_PREFIX="NCCL_DEBUG=INFO NCCL_DEBUG_FILE=$DEBUG_DIR/nccl-info-hostname-\$SLURMD_NODENAME-local-rank-\$SLURM_LOCALID-procid-\$SLURM_PROCID.txt $CMD_PREFIX"
fi

NSYS_DIR="$NSYS_LOG_DIR/$DATE/nsys/$SLURM_JOB_ID"
[ "$DRY_RUN" = true ] || mkdir -p "$NSYS_DIR"

# NOTE: SLURM_PROCID is only defined inside the srun tasks, NOT in this batch
# script. So the nsys wrapping is selected per-rank inside the `srun bash -c`
# below. Here we only build the launcher string and append the Megatron profiler
# flags (Megatron itself gates the cudaProfilerApi range on --profile-ranks, so
# it is safe to pass these flags to every rank).
NSYS_LAUNCHER=""
if [ "$NSYS_PROFILER" = true ]; then
	NSYS_LAUNCHER="nsys profile -s none --trace=nvtx,cudnn,cublas,cuda \
		--output=${NSYS_DIR}/nsys-${MODEL_NAME}-${SLURM_JOB_ID}-rank\$SLURM_PROCID \
		--force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop"
	TRAINING_CMD="$TRAINING_CMD --profile --profile-step-start $NSYS_PROFILER_START_ITER --profile-step-end $NSYS_PROFILER_END_ITER --profile-ranks $RANK_TO_PROFILE"
fi

if [ "$TORCH_PROFILER" = true ]; then
	TRAINING_CMD="$TRAINING_CMD --use-pytorch-profiler --profile-step-start $PROFILER_START_ITER --profile-step-end $PROFILER_END_ITER --profile-ranks $RANK_TO_PROFILE"
fi

# =============================================================================
# COMPUTE ENVIRONMENT SNAPSHOT
# =============================================================================
# Save sbatch script (the thin experiment file) + the engine files + model env
if [ "$DRY_RUN" != true ]; then   # whole snapshot writes files → skipped on dry run
cp "$0" "$DEBUG_DIR"
cp "$ENGINE_PATH" "$DEBUG_DIR"
cp "$COMMON_DIR/paths.sh" "$DEBUG_DIR"
cp "$COMMON_DIR/model.sh" "$DEBUG_DIR"
cp "$COMMON_DIR/engine.sh" "$DEBUG_DIR"
cp "$RECIPE_FILE" "$DEBUG_DIR"          # optimizer recipe picked by $OPTIMIZER
[ "$USE_UCCL" = true ] && cp "$COMMON_DIR/uccl.sh" "$DEBUG_DIR"
cp "$MODEL_ENV_FILE" "$DEBUG_DIR"

# Record dirty changes in the Megatron-LM codebase
_MEG_DIRTY_DIFF="$DEBUG_DIR/megatron-dirty.diff"
if [ -n "$(git -C "$MEGATRON_LM_DIR" status --porcelain 2>/dev/null)" ]; then
	git -C "$MEGATRON_LM_DIR" diff > "$_MEG_DIRTY_DIFF" 2>/dev/null
	git -C "$MEGATRON_LM_DIR" diff --cached >> "$_MEG_DIRTY_DIFF" 2>/dev/null
	git -C "$MEGATRON_LM_DIR" status --short >> "$_MEG_DIRTY_DIFF" 2>/dev/null
	echo "[$(date)] Dirty changes saved to $_MEG_DIRTY_DIFF"
else
	echo "[$(date)] Megatron-LM working tree clean, no dirty diff to save."
fi

# Clean triggers
rm -f $TRIGGER_DIR/save
rm -f $TRIGGER_DIR/exit

echo "Current Path: ${PWD}"
echo -e "$(date)" > $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nCMD: $CMD_PREFIX $TRAINING_CMD" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nExperiment file: $0\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $0 >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nEngine files: $ENGINE_PATH , paths.sh , model.sh , engine.sh , $RECIPE_FILE\n" >> $COMPUTE_ENVIRONMENT_DIR
cat "$ENGINE_PATH" "$COMMON_DIR/paths.sh" "$COMMON_DIR/model.sh" "$COMMON_DIR/engine.sh" "$RECIPE_FILE" >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nTOML file: $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nNODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nModel stats (tools/flops_calculator.py):\n$MODEL_STATS" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nMegatron path: $MEGATRON_LM_DIR" >> $COMPUTE_ENVIRONMENT_DIR
echo "  Branch:  $_MEG_BRANCH" >> $COMPUTE_ENVIRONMENT_DIR
echo "  Commit:  $_MEG_COMMIT$_MEG_DIRTY" >> $COMPUTE_ENVIRONMENT_DIR
echo "  Remote:  $(git -C "$MEGATRON_LM_DIR" remote get-url origin 2>/dev/null || echo N/A)" >> $COMPUTE_ENVIRONMENT_DIR
echo "  Describe: $(git -C "$MEGATRON_LM_DIR" describe --tags --always 2>/dev/null || echo N/A)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\n$(python -m pip list 2>/dev/null || python3 -m pip list 2>/dev/null || echo 'pip not available')" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\n$(nvidia-smi)" >> $COMPUTE_ENVIRONMENT_DIR # CUDA Version & Driver
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nEnvironment Variables:\n\n$(printenv)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
fi   # end DRY_RUN snapshot skip

# =============================================================================
# LAUNCH
# =============================================================================
# Only the Triton cache is persisted, in $JIT_CACHE_BASE (common/paths.sh;
# override it in a launch script before sourcing this file), and SHARED by all
# ranks: it writes via temp-file + atomic rename, so concurrent ranks are safe,
# a kernel is compiled once instead of once per rank, and the cache stays valid
# when the rank layout changes. That is what keeps kernels from recompiling at
# every launch.
#
# Inductor and cpp_extension caches go to node-local /tmp, per job and per rank:
# both guard writes with file locks that can stall when hundreds of ranks
# contend on Lustre, so they are rebuilt per job rather than persisted.
#
# The bases expand here ($SLURM_JOB_ID is the same for every task), \$SLURM_PROCID
# does NOT: in this batch script it is pinned to 0, so exporting it here would
# hand every rank the same value (srun propagates our env).
CACHE_ENV="export TRITON_HOME=$JIT_CACHE_BASE/.triton \
TRITON_CACHE_DIR=$JIT_CACHE_BASE/.triton/cache \
TORCHINDUCTOR_CACHE_DIR=/tmp/$SLURM_JOB_ID/.torch_inductor/\$SLURM_PROCID \
TORCH_EXTENSIONS_DIR=/tmp/$SLURM_JOB_ID/.torch_ext/\$SLURM_PROCID; \
mkdir -p \$TRITON_CACHE_DIR \$TORCHINDUCTOR_CACHE_DIR \$TORCH_EXTENSIONS_DIR;"

# The launch command is built ONCE here: a normal run executes it, DRY_RUN
# prints it. PER_RANK_CMD runs inside every task's container shell (it picks
# the nsys LAUNCHER per rank, since SLURM_PROCID only exists inside the task).
PER_RANK_CMD="$CACHE_ENV LAUNCHER=''; if [[ \" $RANK_TO_PROFILE \" == *\" \$SLURM_PROCID \"* ]]; then LAUNCHER=\"$NSYS_LAUNCHER\"; fi; $UCCL_ENV_PREFIX RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX \$LAUNCHER $TRAINING_CMD"
PER_RANK_CMD=$(printf '%s' "$PER_RANK_CMD" | tr -s ' \t' ' ')
# --wait 60 / --kill-on-bad-exit=1: once one task exits, give the rest 60s and
# then tear the step down, so a single dead rank can't leave the others spinning
# until the wall clock. NOTE the '=': --kill-on-bad-exit takes an OPTIONAL value,
# so the space form ("--kill-on-bad-exit 1") makes srun treat 1 as the command to
# exec ("execve(): 1: No such file or directory" on every rank). --wait takes a
# required value, so the space form is correct there.
# (--jobid is NOT passed: srun inherits the allocation from sbatch, and a literal
# job id would break the pasteable DRY_RUN form below.)
SRUN_LAUNCH="srun --cpus-per-task $SLURM_CPUS_PER_TASK --mpi=pmix --distribution=block:block --network=disable_rdzv_get --environment=$IMAGE_ENV --wait 60 --kill-on-bad-exit=1 -lu"

if [ "$DRY_RUN" = true ]; then
	echo
	echo "============================ DRY RUN ============================"
	echo "# Everything below is a complete sbatch script: save it to a file and"
	echo "# submit with \`sbatch <file>\`, or paste the exports + srun into an"
	echo "# salloc shell of the same shape (the #SBATCH lines are ignored there)."
	echo "#!/bin/bash"
	echo "$_SB_HEADER"
	echo
	echo "export MASTER_ADDR=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)"
	echo "export MASTER_PORT=$MASTER_PORT WORLD_SIZE=\$((SLURM_NNODES * 4))"
	echo "export OMP_NUM_THREADS=$OMP_NUM_THREADS CUDA_DEVICE_MAX_CONNECTIONS=1"
	echo "export TORCH_NCCL_AVOID_RECORD_STREAMS=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1"
	echo "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRANSFORMERS_NO_SLOW_TOKENIZER=1"
	[ -z "$WANDB_API_KEY" ] && echo "export WANDB_MODE=disabled"
	if [ "$USE_UCCL" = true ]; then
		echo "export NUM_MAX_NVL_PEERS=$NUM_MAX_NVL_PEERS UCCL_EP_TRANSPORT=$UCCL_EP_TRANSPORT${UCCL_EP_CPU_TIMEOUT_SECS:+ UCCL_EP_CPU_TIMEOUT_SECS=$UCCL_EP_CPU_TIMEOUT_SECS}"
	fi
	# Pretty-print: srun options one per line; the single-quoted bash -c body is
	# split before each --arg into concatenated 'fragments' ("...'\" + newline +
	# "' --..."), which the pasted shell glues back into ONE bash -c argument —
	# so continuation lines of the body must start at column 0.
	echo "srun --cpus-per-task $SLURM_CPUS_PER_TASK \\"
	echo "    --mpi=pmix \\"
	echo "    --distribution=block:block \\"
	echo "    --network=disable_rdzv_get \\"
	echo "    --environment=$IMAGE_ENV \\"
	echo "    --wait 60 \\"
	echo "    --kill-on-bad-exit=1 \\"
	echo "    -lu bash -c \\"
	printf "'%s'\n" "$(printf '%s' "$PER_RANK_CMD" | sed "s/'/'\\\\''/g")" | sed "s/ --/'\\\\\n' --/g"
	echo "================================================================="
	echo "DRY_RUN=true — nothing was created, built or launched."
	exit 0
fi
# NOTE: per-rank Triton/inductor/torch-extension caches are set in $CACHE_ENV
# above, inside PER_RANK_CMD — not here, where SLURM_PROCID is always 0.

if [ "$AUTO_JOB_REQUEUE" = true ]; then
	echo "[$(date)] $(sbatch --dependency=singleton $0)"
fi

# export CUDA_LAUNCH_BLOCKING=1
$SRUN_LAUNCH bash -c "$PER_RANK_CMD"

echo "END TIME: $(date)"

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -rf $TRIGGER_DIR/exit
   scancel --jobname $SLURM_JOB_NAME
fi
