
model=

FUSELORA_PROFILE=${FUSELORA_PROFILE:-0} \
RAY_DEDUP_LOGS=0 \
TOKENIZERS_PARALLELISM=true \
python -m vllm.entrypoints.api_server \
  --model $model \
  --tokenizer $model \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype half \
  -tp 1 \
  --num-groups 1 \
  --num-models 100 \
  --use-dummy-weights \
  --disable-log-requests \
  --exec-type 3 \
  --migration-type 3 \
  --policy credit \
  --engine-use-ray \
  --worker-use-ray \
  --swap-space 16
