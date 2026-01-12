## 목적
vLLM v1 스케줄러의 동작을 연구 목적으로 상세히 분석하기 위한 로깅 시스템 구축

## 요구사항 개요
스케줄러가 실행되는 동안 각 request와 iteration에 대한 상세 정보를 수집하고, 이를 CSV 형식으로 출력하여 성능 분석 및 스케줄링 정책 연구에 활용

---

## 1. Request-level 메트릭

각 request에 대해 다음 정보를 추적:

### 1.1 기본 타이밍 정보
- **request_id**: Request의 고유 식별자 (vLLM의 request_id 사용)
- **arrival_time**: Request가 스케줄러에 도착한 시간 (nanosecond 단위)
- **first_scheduled_time**: Request가 처음 running queue에 들어간 시간
- **completion_time**: Request가 완전히 종료된 시간
- **waiting_time**: Waiting queue에서 대기한 총 시간 (여러 번 preemption 당한 경우 누적)
- **execution_time**: 실제 running 상태로 처리된 총 시간
- **e2e_latency**: Arrival부터 completion까지 전체 시간

### 1.2 토큰 정보
- **input_token_count**: Input prompt의 토큰 수
- **output_token_count**: 현재까지 생성된 output 토큰 수
- **total_token_count**: Input + output 토큰 합계
- **current_iteration**: 현재 몇 번째 iteration을 처리 중인지

### 1.3 성능 지표
- **TTFT** (Time To First Token): 첫 번째 토큰이 생성되기까지 걸린 시간
- **TPOT** (Time Per Output Token): Output 토큰 하나당 평균 생성 시간
- **throughput**: 단위 시간당 생성된 토큰 수 (tokens/sec)

### 1.4 메모리 및 스케줄링 정보
- **kv_cache_blocks**: Request가 사용하는 KV cache block 수
- **kv_cache_memory_bytes**: Request가 사용하는 KV cache 메모리 양 (bytes)
- **preemption_count**: Request가 preemption 당한 총 횟수
- **eviction_count**: Request가 eviction 당한 총 횟수
- **priority_score**: 스케줄러가 사용하는 우선순위 점수 (있다면)

### 1.5 상태 전이 히스토리
- **state_transitions**: Request의 상태 전이 히스토리 (JSON 또는 별도 파일)
  - 예: `[{time: 1000, state: "waiting"}, {time: 2000, state: "running"}, {time: 3000, state: "preempted"}, ...]`

### 1.6 Waiting 원인 분류
Request가 waiting 상태일 때 그 원인을 기록:
- `max_num_seqs`: Running request 개수가 max_num_seqs 제한에 도달
- `max_num_batched_tokens`: Batch 내 총 토큰 수가 max_num_batched_tokens 제한에 도달
- `memory_insufficient`: KV cache 메모리 부족
- `context_length_limit`: Max context length 제약
- `decode_priority`: Decode phase 우선 정책으로 인한 prefill 대기
- `chunked_prefill`: Prefill이 chunk로 나뉘어 처리 중
- `preempted`: 다른 request에 의해 preemption 당함
- `evicted`: Eviction 당함
- `recomputation`: Eviction 후 KV cache 재계산 대기
- `initial_waiting`: 아직 처음 스케줄링되지 않음

---

## 2. Iteration-level 메트릭

스케줄러의 각 iteration마다 다음 정보를 기록:

### 2.1 타이밍 정보
- **iteration_id**: Iteration 번호 (0부터 시작)
- **iteration_start_time**: Iteration 시작 시간 (nanosecond)
- **iteration_end_time**: Iteration 종료 시간 (nanosecond)
- **iteration_duration**: Iteration 소요 시간
- **scheduling_overhead**: Scheduling 결정에 걸린 시간
- **forward_pass_duration**: 실제 모델 forward pass 시간

### 2.2 Batch 정보
- **batch_size**: 현재 iteration에서 처리된 request 개수
- **total_tokens_in_batch**: Batch 내 처리한 총 토큰 수
- **prefill_tokens**: Prefill phase 토큰 수
- **decode_tokens**: Decode phase 토큰 수

### 2.3 Queue 상태
- **running_requests**: Running queue에 있는 request ID 리스트
- **waiting_requests**: Waiting queue에 있는 request ID 리스트
- **preempted_requests**: 이번 iteration에서 preemption 당한 request ID 리스트
- **admitted_requests**: 이번 iteration에서 새로 admit된 request ID 리스트
- **completed_requests**: 이번 iteration에서 완료된 request ID 리스트

### 2.4 시스템 리소스
- **kv_cache_usage**: 현재 사용 중인 KV cache blocks
- **kv_cache_total**: 전체 사용 가능한 KV cache blocks
- **kv_cache_utilization**: KV cache 사용률 (%)
- **gpu_memory_used**: GPU 메모리 사용량 (bytes, 가능하면)
- **gpu_utilization**: GPU 사용률 (%, 가능하면)

### 2.5 Scheduler Decision
- **scheduler_action**: 이번 iteration의 주요 결정
  - 예: `admit_new`, `preempt`, `evict`, `continue`, `swap_in`, `swap_out`

---

## 3. Per-Iteration Request 상태

각 iteration마다 각 request의 현재 상태를 기록:

- **iteration_id**: Iteration 번호
- **request_id**: Request ID
- **state**: 현재 상태 (`waiting`, `running`, `preempted`, `completed`)
- **queue_position**: Queue에서의 위치 (waiting queue에 있을 경우)
- **waiting_reason**: Waiting 상태일 경우 그 원인 (섹션 1.6 참조)
- **tokens_generated_so_far**: 현재까지 생성된 토큰 수
- **kv_cache_blocks**: 현재 사용 중인 KV cache blocks
- **time_in_current_state**: 현재 상태에 머문 시간

---

## 4. Event Log

주요 이벤트를 시간순으로 기록:

- **timestamp**: 이벤트 발생 시간 (nanosecond)
- **event_type**: 이벤트 타입
  - `request_arrived`
  - `request_admitted`
  - `request_scheduled`
  - `request_preempted`
  - `request_evicted`
  - `request_completed`
  - `first_token_generated`
  - `kv_cache_allocated`
  - `kv_cache_freed`
- **request_id**: 관련 request ID
- **details**: 추가 정보 (JSON 형식)

---

## 5. 출력 형식

다음 4개의 CSV 파일로 출력:

### 5.1 `per_request.csv`
Request별 요약 통계
```csv
request_id,arrival_time,first_scheduled_time,completion_time,waiting_time,execution_time,e2e_latency,input_tokens,output_tokens,ttft,tpot,throughput,preemption_count,eviction_count,kv_cache_blocks
```

### 5.2 `per_iteration.csv`
각 iteration의 시스템 상태
```csv
iteration_id,start_time,end_time,duration,scheduling_overhead,forward_duration,batch_size,total_tokens,prefill_tokens,decode_tokens,kv_cache_used,kv_cache_total,running_count,waiting_count,preempted_count,scheduler_action
```

### 5.3 `per_iteration_requests.csv`
각 iteration에서 각 request의 상태
```csv
iteration_id,request_id,state,queue_position,waiting_reason,tokens_generated,kv_cache_blocks,time_in_state
```

### 5.4 `events.csv`
모든 이벤트 로그
```csv
timestamp,event_type,request_id,details
```

---

## 6. 구현 요구사항

### 6.1 시간 단위
- Python `time.perf_counter_ns()` 사용 (nanosecond 단위)
- 스케줄러 시작 시점을 `t=0`으로 하는 상대 시간 사용
- 절대 timestamp도 함께 기록

### 6.2 성능 고려사항
- 로깅이 스케줄러 성능에 영향을 최소화해야 함
- 가능하면 비동기 로깅 구현
- CSV 버퍼링 및 주기적 flush (예: 100 iterations마다 & 60초 마다, 선 도달 시 flush)
- 메모리 사용량 고려 (큰 배치 실험 시 메모리 부족 방지)

### 6.3 설정 옵션
- 환경변수 또는 config 파일로 로깅 활성화/비활성화
  - 예: `VLLM_SCHEDULER_LOGGING=1`
- 로그 출력 경로 지정 가능
  - 예: `VLLM_SCHEDULER_LOG_DIR=/path/to/logs`
- 로깅 레벨 설정 (전체/요약만/비활성화)

### 6.4 코드 수정 위치
vLLM v1 스케줄러 코드베이스 내에서 다음 위치들을 수정:
- Request 도착 시점: `arrival_time` 기록
- Scheduler 메인 루프: Iteration 시작/종료, queue 상태 기록
- Request state transition: 상태 변경 시 transition 기록
- Scheduling decision: Preemption/eviction/admission 결정 시 이유 기록
- Model execution: Forward pass 타이밍 측정
- Request completion: 최종 통계 계산 및 기록

### 6.5 데이터 구조
- Request별 메트릭을 저장할 딕셔너리 또는 클래스 생성
- Iteration별 메트릭을 저장할 구조 생성
- CSV writer를 관리하는 Logger 클래스 생성

---

## 7. 추가 고려사항

### 7.1 Fairness 메트릭
- Queue position 변화 추적으로 starvation 분석 가능
- Request별 service rate 계산

### 7.2 디버깅 정보
- Scheduler의 결정 이유를 상세히 기록 (디버그 모드)
- 예상치 못한 대기 시간 발생 시 원인 추적 가능하도록

### 7.3 확장성
- 추후 새로운 메트릭 추가 시 기존 로그와 호환되도록 설계
- CSV 컬럼 순서 고정, 버전 정보 기록

---

## 8. 예시 사용 시나리오

```bash
# 로깅 활성화하여 vLLM 실행
export VLLM_SCHEDULER_LOGGING=1
export VLLM_SCHEDULER_LOG_DIR=./scheduler_logs
python -m vllm.entrypoints.api_server --model meta-llama/Llama-2-7b

# 실험 종료 후 로그 분석
ls scheduler_logs/
# per_request.csv
# per_iteration.csv
# per_iteration_requests.csv
# events.csv
```

분석 예시:
```python
import pandas as pd

# Request 요약 통계
requests = pd.read_csv('scheduler_logs/per_request.csv')
print(f"Average TTFT: {requests['ttft'].mean()}")
print(f"Average TPOT: {requests['tpot'].mean()}")
print(f"Preemption rate: {(requests['preemption_count'] > 0).mean()}")

# Iteration별 리소스 사용률
iterations = pd.read_csv('scheduler_logs/per_iteration.csv')
iterations['kv_cache_util'] = iterations['kv_cache_used'] / iterations['kv_cache_total']
print(f"Average KV cache utilization: {iterations['kv_cache_util'].mean()}")

# 이벤트 로그 분석
events = pd.read_csv('scheduler_logs/events.csv')
preemption_events = events[events['event_type'] == 'request_preempted']
print(f"Total preemptions: {len(preemption_events)}")
```

---

## 9. 구현 체크리스트

- [ ] Request 도착 시 arrival_time 기록
- [ ] Iteration 시작/종료 시 타이밍 측정
- [ ] Queue 상태 추적 (running/waiting/preempted)
- [ ] Waiting 원인 분류 로직 구현
- [ ] State transition 히스토리 기록
- [ ] TTFT, TPOT 계산
- [ ] KV cache 사용량 추적
- [ ] CSV 파일 생성 및 writer 구현
- [ ] 환경변수 기반 설정 구현
- [ ] 비동기 로깅 또는 버퍼링 구현
- [ ] 성능 오버헤드 측정 및 최적화
- [ ] 예시 분석 스크립트 작성
