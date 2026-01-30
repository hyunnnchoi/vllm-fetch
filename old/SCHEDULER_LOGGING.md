# vLLM v1 Scheduler Logging

## 개요

vLLM v1 스케줄러의 동작을 상세히 분석하기 위한 로깅 시스템입니다. Request-level과 iteration-level 메트릭을 수집하여 CSV 파일로 출력합니다.

## 기능

### 수집하는 메트릭

#### 1. Request-level 메트릭 (`per_request.csv`)
- **타이밍 정보**: arrival_time, first_scheduled_time, completion_time
- **대기/실행 시간**: waiting_time, execution_time, e2e_latency
- **토큰 정보**: input_tokens, output_tokens
- **성능 지표**: TTFT (Time To First Token), TPOT (Time Per Output Token), throughput
- **스케줄링 정보**: preemption_count, eviction_count, kv_cache_blocks, kv_cache_memory_bytes, priority_score

#### 2. Iteration-level 메트릭 (`per_iteration.csv`)
- **타이밍**: iteration duration, scheduling overhead, forward pass duration
- **배치 정보**: batch_size, total_tokens, prefill_tokens, decode_tokens
- **큐 상태**: running/waiting/preempted request 수
- **시스템 리소스**: KV cache 사용량 및 사용률

#### 3. Per-iteration request 상태 (`per_iteration_requests.csv`)
- 각 iteration에서 각 request의 상태 (running/waiting/preempted/completed)
- Queue position, waiting reason
- 현재까지 생성된 토큰 수
- 현재 상태에 머문 시간

#### 4. 이벤트 로그 (`events.csv`)
- 시간순으로 기록된 모든 주요 이벤트
- request_arrived, request_scheduled, request_preempted, request_completed 등

#### 5. 상태 전이 로그 (`state_transitions.csv`)
- Request별 상태 전이 히스토리
- 각 상태에서 보낸 시간
- 상태별 duration 분석 가능

## 사용 방법

### 1. 환경변수 설정

```bash
# 로깅 활성화
export VLLM_SCHEDULER_LOGGING=1

# 로그 출력 디렉토리 지정 (선택사항, 기본값: ./scheduler_logs)
export VLLM_SCHEDULER_LOG_DIR=/path/to/logs
```

### 2. vLLM 실행

```bash
# API 서버 실행 예시
python -m vllm.entrypoints.api_server \
    --model meta-llama/Llama-2-7b \
    --max-num-seqs 256
```

### 3. 로그 분석

실험 종료 후 생성된 CSV 파일을 분석할 수 있습니다:

```bash
# 로그 파일 확인
ls ./scheduler_logs/
# per_request.csv
# per_iteration.csv
# per_iteration_requests.csv
# events.csv
# state_transitions.csv

# 분석 스크립트 실행
python examples/analyze_scheduler_logs.py ./scheduler_logs
```

## 분석 예시

### Python을 사용한 커스텀 분석

```python
import pandas as pd

# Request 요약 통계
requests = pd.read_csv('scheduler_logs/per_request.csv')

# TTFT를 초 단위로 변환 (nanoseconds -> seconds)
requests['ttft_sec'] = requests['ttft'] / 1e9

print(f"Average TTFT: {requests['ttft_sec'].mean():.4f} sec")
print(f"P50 TTFT: {requests['ttft_sec'].quantile(0.50):.4f} sec")
print(f"P90 TTFT: {requests['ttft_sec'].quantile(0.90):.4f} sec")
print(f"P99 TTFT: {requests['ttft_sec'].quantile(0.99):.4f} sec")

# TPOT 분석
requests['tpot_sec'] = requests['tpot'] / 1e9
print(f"Average TPOT: {requests['tpot_sec'].mean():.4f} sec")

# Preemption 분석
preemption_rate = (requests['preemption_count'] > 0).mean()
print(f"Preemption rate: {preemption_rate:.2%}")

# Iteration별 리소스 사용률
iterations = pd.read_csv('scheduler_logs/per_iteration.csv')
iterations['kv_cache_util'] = iterations['kv_cache_used'] / iterations['kv_cache_total']
print(f"Average KV cache utilization: {iterations['kv_cache_util'].mean():.2%}")

# 이벤트 로그 분석
events = pd.read_csv('scheduler_logs/events.csv')
preemption_events = events[events['event_type'] == 'request_preempted']
print(f"Total preemptions: {len(preemption_events)}")
```

### 시각화 예시

```python
import matplotlib.pyplot as plt

# TTFT 분포 히스토그램
requests['ttft_sec'] = requests['ttft'] / 1e9
plt.figure(figsize=(10, 6))
plt.hist(requests['ttft_sec'], bins=50, edgecolor='black')
plt.xlabel('TTFT (seconds)')
plt.ylabel('Frequency')
plt.title('Time To First Token Distribution')
plt.savefig('ttft_distribution.png')

# Iteration별 배치 크기 추이
plt.figure(figsize=(12, 6))
plt.plot(iterations['iteration_id'], iterations['batch_size'])
plt.xlabel('Iteration')
plt.ylabel('Batch Size')
plt.title('Batch Size Over Time')
plt.savefig('batch_size_over_time.png')

# KV cache 사용률 추이
plt.figure(figsize=(12, 6))
plt.plot(iterations['iteration_id'], iterations['kv_cache_util'] * 100)
plt.xlabel('Iteration')
plt.ylabel('KV Cache Utilization (%)')
plt.title('KV Cache Utilization Over Time')
plt.savefig('kv_cache_utilization.png')
```

## 성능 고려사항

### 최적화된 설정
- **비동기 로깅**: CSV 쓰기는 버퍼링되어 주기적으로 flush
- **Flush 주기**: 기본적으로 100 iterations마다 또는 60초마다 flush
- **메모리 효율**: 메트릭을 주기적으로 파일에 쓰고 메모리에서 제거

### 성능 영향
로깅 시스템은 스케줄러 성능에 미치는 영향을 최소화하도록 설계되었습니다:
- 단순한 메트릭 수집 (복잡한 계산 없음)
- 버퍼링된 I/O
- 조건부 활성화 (환경변수로 제어)

## 구현 세부사항

### 파일 구조
```
vllm/v1/core/sched/
├── logger.py           # SchedulerLogger 클래스 구현
├── scheduler.py        # 로깅 포인트가 통합된 스케줄러
└── SCHEDULER_LOGGING.md  # 이 문서
```

### 주요 클래스
- `SchedulerLogger`: 메트릭 수집 및 CSV 출력을 담당하는 메인 클래스
- `RequestMetrics`: Request별 메트릭을 추적하는 데이터 클래스
- `IterationMetrics`: Iteration별 메트릭을 추적하는 데이터 클래스
- `WaitingReason`: Request가 waiting 상태인 이유를 나타내는 enum

### 시간 측정
모든 시간은 `time.perf_counter_ns()`를 사용하여 nanosecond 단위로 측정됩니다:
- 스케줄러 시작 시점을 t=0으로 하는 상대 시간 사용
- 높은 정밀도와 일관성 보장

## 문제 해결

### 로그 파일이 생성되지 않는 경우
1. 환경변수가 올바르게 설정되었는지 확인:
   ```bash
   echo $VLLM_SCHEDULER_LOGGING
   ```
2. 로그 디렉토리에 쓰기 권한이 있는지 확인
3. vLLM이 실제로 요청을 처리했는지 확인

### 로그 파일이 비어있는 경우
- 로깅이 활성화되었지만 아직 flush되지 않았을 수 있습니다
- 서버를 정상적으로 종료하여 마지막 데이터를 flush하세요

### 메모리 사용량이 높은 경우
- `VLLM_SCHEDULER_LOG_DIR`을 충분한 공간이 있는 디렉토리로 설정
- Flush 주기를 조정하려면 `SchedulerLogger` 초기화 시 `flush_interval_iterations` 또는 `flush_interval_seconds` 파라미터 수정

## 향후 개선 사항

- [ ] Waiting reason 자동 분류 정확도 향상
- [ ] 더 상세한 prefill/decode 토큰 분리
- [ ] GPU 메모리 사용량 추적
- [ ] 실시간 대시보드 지원
- [ ] Distributed 환경 지원

## 참고

이 로깅 시스템은 연구 및 성능 분석 목적으로 설계되었습니다. 프로덕션 환경에서는 필요한 경우에만 활성화하는 것을 권장합니다.
