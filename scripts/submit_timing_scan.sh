#!/bin/bash
# Submit a GPU and CPU timing job for each jz*_5kevts.root timing sample.
# GPU runs cover all 4096 events; CPU runs are capped at 100 events.

set -e

TOP_DIR="/storage/agrp/dreyet/f_delphes/cms-flow-evt"
PAYLOAD="${TOP_DIR}/scripts/run_timing.sh"
LOG_DIR="${TOP_DIR}/logs"
SAMPLE_GLOB="/storage/agrp/dreyet/f_delphes/data/JZ_ATLAS_mc23e_FullSim/timing_samples/jz*_5kevts.root"

mkdir -p "${LOG_DIR}"

# Multiple GPU jobs may co-tenant on a node (~4 A6000s/node). Per-job GPU
# isolation comes from $PBS_GPUFILE -> CUDA_VISIBLE_DEVICES inside run_timing.sh,
# so we just ask for ngpus=1 per job and skip place=excl.
GPU_RESOURCES="walltime=01:00:00,mem=24gb,ncpus=4,ngpus=1,gputype=A6000,io=1"
CPU_RESOURCES="walltime=04:00:00,mem=8gb,ncpus=1,io=1"

GPU_NUM_EVENTS=4096
GPU_BATCH_SIZE=1024
CPU_NUM_EVENTS=100
CPU_BATCH_SIZE=1
N_STEPS=50

shopt -s nullglob
samples=( ${SAMPLE_GLOB} )
shopt -u nullglob

if [ ${#samples[@]} -eq 0 ]; then
  echo "No samples matched ${SAMPLE_GLOB}" >&2
  exit 1
fi

echo "Submitting timing scan over ${#samples[@]} samples"

for test_path in "${samples[@]}"; do
  tag=$(basename "${test_path}" .root)

  # for device in gpu cpu; do
  for device in gpu; do
    if [ "${device}" = "gpu" ]; then
      num_events=${GPU_NUM_EVENTS}
      batch_size=${GPU_BATCH_SIZE}
    else
      num_events=${CPU_NUM_EVENTS}
      batch_size=${CPU_BATCH_SIZE}
    fi

    output_file="timing_${tag}_${device}_bs${batch_size}_nsteps${N_STEPS}.txt"
    jobname="time_${device}_${tag}"

    if [ "${device}" = "gpu" ]; then
      resources="${GPU_RESOURCES}"
    else
      resources="${CPU_RESOURCES}"
    fi

    qsub_cmd=(qsub
      -N "${jobname}"
      -q N
      -o "${LOG_DIR}/${jobname}.out"
      -e "${LOG_DIR}/${jobname}.err"
      -l "${resources}"
      -v "DEVICE=${device},TEST_PATH=${test_path},NUM_EVENTS=${num_events},BATCH_SIZE=${batch_size},N_STEPS=${N_STEPS},OUTPUT=${output_file}"
      "${PAYLOAD}"
    )

    echo "${qsub_cmd[*]}"
    "${qsub_cmd[@]}"
  done
done
