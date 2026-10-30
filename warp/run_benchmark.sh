#!/bin/bash

RATES=
BACKEND=vllm
TOKENIZER=
DATASET=
TRACE_NAME=azure_v1
TRACE_PATH=
HOST=127.0.0.1
PORT=8001

NUM_MODELS=100
NUM_PROMPTS=1000
EXEC_TYPE=3
OUTPUT_STYLE=4

START_TIME=0.0.0
END_TIME=0.5.0
INTERVAL=30

cd 

for RATE in "${RATES[@]}"; do
    OUTPUT_FILE=
    BREAKDOWN_OUTPUT_FILE=

    echo "Running benchmark with request-rate=${RATE}"
    python  \
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
        --output ${OUTPUT_FILE} \
        --breakdown-output ${BREAKDOWN_OUTPUT_FILE} \
        --adapter-ranks-output 

    echo "Finished rate=${RATE}"

    sleep 60
    echo "----------------------------------------"
done

