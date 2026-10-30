cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

model=/home/data/huggingface/hub/Llama-2-13b-chat-hf
# model=/home/data/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590

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
