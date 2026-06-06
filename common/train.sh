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
#   4. Ends with:            source "$(dirname "$0")/../../common/train.sh"
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

# -- Data --
: "${DATASET_NAME:=climbmix}"           # climbmix | fineweb-edu-100B
: "${MOCK_DATA:=false}"
# (DATASET_CACHE_DIR comes from common/paths.sh)

# -- Container / codebase --
# (IMAGE_ENV and MEGATRON_LM_DIR come from common/paths.sh)

# -- Experiment naming --
: "${PROJECT_NAME:=large_scale_moe_performance}"
: "${EXP_NAME_SUFFIX:=newmegatron_py2512}"

# -- Checkpointing & resuming --
: "${CHECKPOINT_STEPS:=100000}"
: "${LOAD_CKPT:=false}"
: "${AUTO_JOB_REQUEUE:=false}"
: "${BACKUP_CODEBASE:=false}"

# -- Debugging / profiling --
: "${LOG_NCCL:=false}"
: "${NSYS_PROFILER:=false}"
: "${TORCH_PROFILER:=false}"
: "${RANK_TO_PROFILE:=0}"
: "${PROFILER_START_ITER:=15}"
: "${PROFILER_END_ITER:=16}"
: "${NSYS_PROFILER_START_ITER:=5}"
: "${NSYS_PROFILER_END_ITER:=6}"

# =============================================================================
# DOMAIN CONFIGS
# Order matters: engine.sh reads GBS/MBS/OPTIMIZER set by model.sh.
# =============================================================================
source "$COMMON_DIR/model.sh"
source "$COMMON_DIR/engine.sh"

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

# =============================================================================
# LOGGING DIRECTORIES & ARTIFACTS  (EXP_NAME mixes model + optimization knobs)
# =============================================================================
EXP_NAME=${MODEL_NAME}-${OPTIMIZER}-${MUON_SCALE_MODE}-nestrov_${USE_NESTEROV}-${DATASET_NAME}-${SLURM_NNODES}n-${SEQ_LEN}sl-${GBS}gbsz-${MBS}mbsz-${LR}lr-${TP}tp-${PP}pp-${EP}ep-${ETP}etp-${CP}cp-fp8act${USE_FP8_ACTIVATION}-mockr${USE_MOCK_ROUTER}-off${USE_EXPERTS_OFFLOADING}-${EXP_NAME_SUFFIX}
LOAD_EXP_NAME=$EXP_NAME
PROJECT_DIR=$MEGATRON_LM_DIR/logs/Meg-Runs/$PROJECT_NAME

EXP_DIR=$PROJECT_DIR/$EXP_NAME
SAVE_CKPT_DIR=$CKPT_BASE_DIR/$PROJECT_NAME/$EXP_NAME/checkpoints
LOAD_CKPT_DIR=$CKPT_BASE_DIR/$PROJECT_NAME/$LOAD_EXP_NAME/checkpoints

TRIGGER_DIR=$EXP_DIR/triggers
DEBUG_DIR=$EXP_DIR/debug/$SLURM_JOB_ID
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
	--no-log-loss-scale-to-tensorboard
	--log-memory-to-tensorboard
	--log-timers-to-tensorboard
	--log-memory-interval 100
)

if [ "$LOAD_CKPT" = true ]; then
	CHECKPOINTING_ARGS=(
		--save $SAVE_CKPT_DIR
		--save-interval $CHECKPOINT_STEPS
		--ckpt-format torch_dist
		--load $LOAD_CKPT_DIR
		--async-save
	)
else
	# If not loading from checkpoint, start fresh and ignore existing checkpoints
	CHECKPOINTING_ARGS=(
		--save $SAVE_CKPT_DIR
		--save-interval $CHECKPOINT_STEPS
		--ckpt-format torch_dist
		--async-save
	)
fi

DATA_ARGS=(
	--split 100,0,0
	--seq-length $SEQ_LEN
	--reset-position-ids  # crossDocAttn
	--reset-attention-mask  # crossDocAttn
	--eod-mask-loss  # crossDocAttn
	--num-workers 16
	--num-dataset-builder-threads 1
	# --goldfish-loss  # goldfish
	# --goldfish-k 50  # goldfish
	# --goldfish-h 50  # goldfish
)

# =============================================================================
# DIRECTORIES
# =============================================================================
mkdir -p $LOAD_CKPT_DIR
mkdir -p $SAVE_CKPT_DIR
mkdir -p $PROJECT_DIR
mkdir -p $TRIGGER_DIR
mkdir -p $DEBUG_DIR
mkdir -p $LOGGING_DIR

# Backup codebase
if [ "$BACKUP_CODEBASE" == true ]; then
  if [ -z "$(ls -A "$BACKUP_CODEBASE_DIR")" ]; then
  	echo "[$(date)] Copying codebase in $MEGATRON_LM_DIR to $BACKUP_CODEBASE_DIR..."
  	rsync -av --exclude-from=$MEGATRON_LM_DIR/.gitignore $MEGATRON_LM_DIR/ $BACKUP_CODEBASE_DIR/ &> /dev/null
  fi
  MEGATRON_LM_DIR=$BACKUP_CODEBASE_DIR
fi

echo "[$(date)] Using codebase in $MEGATRON_LM_DIR"

cd $MEGATRON_LM_DIR
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH

# Data Args
if [ "$MOCK_DATA" = true ]; then
  DATA_ARGS="${DATA_ARGS[@]} --mock-data"
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
  if [ -d "$LOGGING_DIR/wandb/latest-run" ]; then
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

NSYS_DIR="$NSYS_LOG_DIR/$DATE/nsys/"
mkdir -p "$NSYS_DIR"

# NOTE: SLURM_PROCID is only defined inside the srun tasks, NOT in this batch
# script. So the nsys wrapping is selected per-rank inside the `srun bash -c`
# below. Here we only build the launcher string and append the Megatron profiler
# flags (Megatron itself gates the cudaProfilerApi range on --profile-ranks, so
# it is safe to pass these flags to every rank).
NSYS_LAUNCHER=""
if [ "$NSYS_PROFILER" = true ]; then
	NSYS_LAUNCHER="nsys profile -s none --trace=nvtx,cudnn,cublas,cuda \
		--output=${NSYS_DIR}/nsys-${EXP_NAME}-rank${RANK_TO_PROFILE} \
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
cp "$0" "$DEBUG_DIR"
cp "$ENGINE_PATH" "$DEBUG_DIR"
cp "$COMMON_DIR/model.sh" "$DEBUG_DIR"
cp "$COMMON_DIR/engine.sh" "$DEBUG_DIR"
cp "$MODEL_ENV_FILE" "$DEBUG_DIR"

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
echo -e "\nEngine files: $ENGINE_PATH , model.sh , engine.sh\n" >> $COMPUTE_ENVIRONMENT_DIR
cat "$ENGINE_PATH" "$COMMON_DIR/model.sh" "$COMMON_DIR/engine.sh" >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nTOML file: $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment\n" >> $COMPUTE_ENVIRONMENT_DIR
cat $SLURM_SPANK__SLURM_SPANK_OPTION_pyxis_environment >> $COMPUTE_ENVIRONMENT_DIR
echo -e "" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nNODES: $(scontrol show hostnames $SLURM_JOB_NODELIST)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nMegatron path: $MEGATRON_LM_DIR ($(git -C $MEGATRON_LM_DIR rev-parse --verify HEAD))" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\n$(python -m pip list 2>/dev/null || python3 -m pip list 2>/dev/null || echo 'pip not available')" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\n$(nvidia-smi)" >> $COMPUTE_ENVIRONMENT_DIR # CUDA Version & Driver
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR
echo -e "\nEnvironment Variables:\n\n$(printenv)" >> $COMPUTE_ENVIRONMENT_DIR
printf '=%.0s' {1..100} >> $COMPUTE_ENVIRONMENT_DIR

# =============================================================================
# LAUNCH
# =============================================================================
# Per-rank local caches (compiled kernels) on node-local storage
export LOCAL_CACHE_BASE=${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}
export TRITON_CACHE_DIR=${LOCAL_CACHE_BASE}/triton_cache/${SLURM_PROCID}
export TORCHINDUCTOR_CACHE_DIR=${LOCAL_CACHE_BASE}/inductor_cache/${SLURM_PROCID}
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

SRUN_ARGS=" \
	-lu \
	--cpus-per-task $SLURM_CPUS_PER_TASK \
	--wait 60 \
	--jobid $SLURM_JOB_ID \
	--kill-on-bad-exit 1 \
	"

if [ "$AUTO_JOB_REQUEUE" = true ]; then
	echo "[$(date)] $(sbatch --dependency=singleton $0)"
fi

srun --cpus-per-task $SLURM_CPUS_PER_TASK --mpi=pmix \
	--distribution=block:block \
	--network=disable_rdzv_get \
    --environment=$IMAGE_ENV \
	-lu bash -c "
		LAUNCHER=''
		if [ \"\$SLURM_PROCID\" -eq \"$RANK_TO_PROFILE\" ]; then
			LAUNCHER=\"$NSYS_LAUNCHER\"
		fi
		RANK=\$SLURM_PROCID LOCAL_RANK=\$SLURM_LOCALID $CMD_PREFIX \$LAUNCHER $TRAINING_CMD"

echo "END TIME: $(date)"

if [ -f $TRIGGER_DIR/exit ]; then
   echo "[$(date)] Detected exit trigger in $TRIGGER_DIR/exit, cancelling pending jobs"
   rm -rf $TRIGGER_DIR/exit
   scancel --jobname $SLURM_JOB_NAME
fi
