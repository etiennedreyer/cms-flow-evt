#!/bin/bash
#PBS -N parnassus_timing
#PBS -q N
#PBS -l walltime=02:00:00
#PBS -l io=1
#PBS -o /storage/agrp/dreyet/f_delphes/cms-flow-evt/logs/timing.out
#PBS -e /storage/agrp/dreyet/f_delphes/cms-flow-evt/logs/timing.err

# Unified timing driver
# Interactive:
#   scripts/run_timing.sh                              # default: gpu, default test file
#   scripts/run_timing.sh cpu
#   scripts/run_timing.sh gpu /path/to/other.root      # override input file
#
# PBS submission (DEVICE/TEST_PATH/NUM_EVENTS/BATCH_SIZE/N_STEPS overridable via -v):
#   qsub -v DEVICE=gpu,TEST_PATH=/path/to/other.root -l ngpus=1,ncpus=15,mem=24gb gputype=A6000 scripts/run_timing.sh
#   qsub -v DEVICE=cpu -l ncpus=1,mem=8gb scripts/run_timing.sh

IOTHROTTLE_LIMIT=0

set -e

source ~/.bashrc
conda activate common

TOP_DIR="/storage/agrp/dreyet/f_delphes/cms-flow-evt"
cd "$TOP_DIR"

# Device selection: CLI arg wins, then $DEVICE env (from qsub -v), else gpu.
device="${1:-${DEVICE:-gpu}}"
# Optional input file override: $2 wins, then $TEST_PATH env, else default below.
test_path_arg="${2:-${TEST_PATH:-}}"

config="${CONFIG:-${TOP_DIR}/saved_models/rcfm_atlas_part_mc23ae_JZ1-9_nsteps10_20260415-T212223/part_atlas.yaml}"
ckpt="${CKPT:-${TOP_DIR}/saved_models/rcfm_atlas_part_mc23ae_JZ1-9_nsteps10_20260415-T212223/ckpts/rcfm_atlas_part_mc23ae_JZ1-9_nsteps10_20260415-T212223-epoch=76-val_loss_avg=1.3879.ckpt}"
evt_config="${ECONFIG:-${TOP_DIR}/saved_models/fm_atlas_evt_mc23ae_JZ1-9_nsteps25_20260415-T174030/evt_atlas.yaml}"
evt_ckpt="${ECKPT:-${TOP_DIR}/saved_models/fm_atlas_evt_mc23ae_JZ1-9_nsteps25_20260415-T174030/ckpts/fm_atlas_evt_mc23ae_JZ1-9_nsteps25_20260415-T174030-epoch=144-val_loss_avg=0.5564.ckpt}"
test_path="${test_path_arg:-/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/test_JZ1-9_FullSim.root}"

n_steps="${N_STEPS:-50}"

if [ "$device" = "cpu" ]; then
  device_args="--cpu"
  device_tag="cpu"
  num_events="${NUM_EVENTS:-100}"
  batch_size="${BATCH_SIZE:-1}"
  # Pull the CPU model name from /proc/cpuinfo (first 'model name' line).
  hardware=$(awk -F': ' '/^model name/ {print $2; exit}' /proc/cpuinfo)
  [ -z "${hardware}" ] && hardware="unknown-cpu"
else
  # Pin to the GPU PBS allocated us so multiple jobs on the same node don't
  # all stampede onto cuda:0. $PBS_GPUFILE lists "host-gpuN" per allocation;
  # extract the numeric ids and feed them to CUDA_VISIBLE_DEVICES. Then we
  # always pass --gpu 0 to time.py because CUDA renumbers from there.
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${PBS_GPUFILE:-}" ] && [ -r "${PBS_GPUFILE}" ]; then
    gpu_ids=$(awk -F'-gpu' '{print $2}' "${PBS_GPUFILE}" | paste -sd,)
    if [ -n "${gpu_ids}" ]; then
      export CUDA_VISIBLE_DEVICES="${gpu_ids}"
      echo "[run_timing] PBS_GPUFILE pinned CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi
  fi
  device_args="--gpu 0"
  device_tag="gpu"
  num_events="${NUM_EVENTS:-10240}"
  batch_size="${BATCH_SIZE:-1024}"
  # GPU type is hardcoded for now (matches qsub -l gputype=...).
  hardware="${GPUTYPE:-A6000}"
fi

# Avoid torch.compile spawning ~15 inductor workers per job; one or two is
# plenty for a single forward graph and saves a swarm of helper processes
# when multiple jobs share a node.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-2}"

output_file="${OUTPUT:-timing_results_${device_tag}_bs${batch_size}_nsteps${n_steps}.txt}"

echo "[run_timing] device=${device_tag} hardware=${hardware} num_events=${num_events} batch_size=${batch_size} n_steps=${n_steps}"
echo "[run_timing] test_path=${test_path}"
echo "[run_timing] output -> ${output_file}"

python time.py \
  --config "${config}" \
  --checkpoint "${ckpt}" \
  --config_evt "${evt_config}" \
  --checkpoint_evt "${evt_ckpt}" \
  --n_steps "${n_steps}" \
  ${device_args} \
  --num_events "${num_events}" \
  --batch_size "${batch_size}" \
  --test_path "${test_path}" \
  --warmup_batches 1 \
  --output "${output_file}" \
  --hardware "${hardware}"
