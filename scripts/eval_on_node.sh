source ~/.bashrc
conda activate common
TOP_DIR="/storage/agrp/dreyet/f_delphes/cms-flow-evt"
cd $TOP_DIR
IOTHROTTLE_LIMIT=0

python eval.py \
  --config ${CONFIG} \
  --checkpoint ${CKPT} \
  --config_evt ${ECONFIG} \
  --checkpoint_evt ${ECKPT} \
  --n_steps 50 \
  --gpu 0 \
  --num_events ${NUM_EVENTS} \
  --batch_size ${BS} \
  --test_path "${TEST_PATH}" \
  --prefix ${PREFIX} \
  --eval_dir evals

# echo "Inference completed."

# ### PARSE EVAL_FILE FROM OUTPUT LOG
# EVAL_FILE=$(grep "Saved to" "${TOP_DIR}/logs/output.log" | tail -1 | awk '{print $NF}')
# EVAL_PATH="${TOP_DIR}/${EVAL_FILE}"
# echo "Evaluation file: ${EVAL_PATH}"

# python evaluation/preprocess_eval_fast.py \
#   -e ${TOP_DIR}/${EVAL_PATH} \
#   -d ${TEST_PATH} \
#   -o "${TOP_DIR}/eval_${EVAL_FILE}" \
#   -n -1 \
#   -dr 0.4 \
#   --eta 2.5