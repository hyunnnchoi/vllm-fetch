# ============================================================
# SECTION 1: vLLM 서버 시작 (A100 80G * 2 환경)
# ============================================================
# NOTE, hyunnnchoi, 2025.12.23 - 벤치마크 파라미터 및 타임스탬프 설정
export VLLM_SCHEDULER_LOGGING=1
export VLLM_SCHEDULER_LOG_DIR=/home/work/hyunmokchoi/multi_turn_0.13.0/scheduler_logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_CLIENTS=256
MAX_ACTIVE_CONV=256
MODEL_NAME="gpt-oss-20b"
MODEL_SHORT_NAME="gpt-oss-20b"  # 파일명용 짧은 이름

# NOTE, hyunnnchoi, 2025.12.28 - 서버 실행 로그 저장
SERVER_LOG_FILE="/home/work/hyunmokchoi/multi_turn_0.13.0/results/server_${MODEL_SHORT_NAME}_${TIMESTAMP}_c${NUM_CLIENTS}_mac${MAX_ACTIVE_CONV}.log"

vllm serve /home/work/huggingface/huggingface/${MODEL_NAME} \
  --port 8000 \
  --host 0.0.0.0 \
  --served-model-name ${MODEL_NAME} \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --max-num-batched-tokens 163840 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 256 \
  --enable-prefix-caching \
  --dtype auto \
  2>&1 | tee "$SERVER_LOG_FILE"

# ============================================================
# SECTION 2: 데이터 준비 (최초 1회만 실행)
# ============================================================
pip3 install pandas
cd /vllm/benchmarks/multi_turn
python3 convert_sharegpt_to_openai.py /home/work/hyunmokchoi/multi_turn/sharegpt.json /home/work/hyunmokchoi/multi_turn/sharegpt_conv_full.json 

# ============================================================
# SECTION 3: 벤치마크 실행 (컨테이너 내부에서)
# ============================================================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
NUM_CLIENTS=256
MAX_ACTIVE_CONV=256
MODEL_NAME="gpt-oss-20b"
MODEL_SHORT_NAME="gpt-oss-20b"  # 파일명용 짧은 이름

cd /vllm
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=dummy
cd /vllm/benchmarks/multi_turn

# 모델 경로 설정
export MODEL_PATH=/home/work/huggingface/huggingface/${MODEL_NAME}


# 벤치마크 실행
# NOTE, hyunnnchoi, 2026.01.08 - 벤치마크 로그 파일 경로 설정
LOG_FILE="/home/work/hyunmokchoi/multi_turn_0.13.0/results/benchmark_${MODEL_SHORT_NAME}_${TIMESTAMP}_c${NUM_CLIENTS}_mac${MAX_ACTIVE_CONV}.log"

python3 benchmark_serving_multi_turn.py \
  --model "$MODEL_PATH" \
  --served-model-name ${MODEL_NAME} \
  --input-file /home/work/hyunmokchoi/multi_turn/sharegpt_conv_full_even.json \
  --num-clients $NUM_CLIENTS \
  --max-active-conversations $MAX_ACTIVE_CONV \
  --no-early-stop \
  --excel-output \
  --verbose \
  2>&1 | tee "$LOG_FILE"

echo "=========================================="
echo "벤치마크 완료!"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Server log: $SERVER_LOG_FILE"
echo "Benchmark log: $LOG_FILE"
echo "Scheduler logs: $VLLM_SCHEDULER_CSV_LOG_DIR"
echo "Server decode timings: $SERVER_DECODE_TIMINGS_DIR"
echo "=========================================="

# ============================================================
# TODO: 추가할 내용
# ============================================================
# - Request 별 Waiting time, Execution time 등 
# - Request 상세 메트릭
