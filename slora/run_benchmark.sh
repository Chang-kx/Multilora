#!/bin/bash

cd -- "$(dirname -- "${BASH_SOURCE[0]}")" || exit 1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

RATES=(8 20 40)
# RATES=(56 72 80)
BACKEND=vllm
# TOKENIZER=/home/data/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590
TOKENIZER=/home/data/huggingface/hub/Llama-2-13b-chat-hf
DATASET=/home/ckx/project/FuseLoRA/ShareGPT_V3_unfiltered_cleaned_split.json
TRACE_NAME=azure_v1
TRACE_PATH=/home/ckx/project/FuseLoRA/maf1/
HOST=127.0.0.1
PORT=8001

NUM_MODELS=100
NUM_PROMPTS=1000
EXEC_TYPE=3
OUTPUT_STYLE=4

START_TIME=0.0.0
END_TIME=0.5.0
INTERVAL=30

# balance_mode=round_robin
# balance_mode=

for RATE in "${RATES[@]}"; do
    OUTPUT_FILE=/home/ckx/project/FuseLoRA/slora/output/fig9_maf1_7b_exec_rate${RATE}.csv

    BREAKDOWN_OUTPUT_FILE="${OUTPUT_FILE%.csv}_breakdown.csv"
    ADAPTER_RANKS_OUTPUT_FILE="${OUTPUT_FILE%.csv}_adapter_ranks.csv"
    mkdir -p "$(dirname -- "$OUTPUT_FILE")"

    echo "Running benchmark with request-rate=${RATE}"
    python /home/ckx/project/FuseLoRA/slora/benchmarks/benchmark_serving.py \
        --backend ${BACKEND} \
        --tokenizer ${TOKENIZER} \
        --dataset ${DATASET} \
        --request-rate ${RATE} \
        --num-models ${NUM_MODELS} \
        --num-prompts ${NUM_PROMPTS} \
        --trace_name ${TRACE_NAME} \
        --trace_path ${TRACE_PATH} \
        --start_time ${START_TIME} \
        --end_time ${END_TIME} \
        --interval ${INTERVAL} \
        --host ${HOST} \
        --port ${PORT} \
        --exec_type ${EXEC_TYPE} \
        --output_style ${OUTPUT_STYLE} \
        --breakdown-output "${BREAKDOWN_OUTPUT_FILE}" \
        --adapter-ranks-output "${ADAPTER_RANKS_OUTPUT_FILE}" \
        --output ${OUTPUT_FILE}  \
        --arrival_distribution balance_mode \

    echo "Finished rate=${RATE}"
    # 停止300秒
    sleep 60
    echo "----------------------------------------"
done

