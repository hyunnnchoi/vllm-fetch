연구 목적으로, vLLM 스케줄러가 돌아갈 때 각 request에 대해 여러 정보를 추출하고 싶음. 
내가 필요한건,
1. 각 request 별: arrival time, waiting time, execution time. 최종적으로 csv 형태로 출력해야 함. 단, 시간 단위에서 오차가 발생하지 않도록 적절한 시간 단위를 사용할 것. 
2. 각 request 별: 현재 몇 번째 iteration을 돌고 있는 것인지, input prompt가 몇 토큰인지, output token이 몇 토큰인지
3. 각 request 별: TTFT, TPOT? 등, Throughput 관련 지표들
4. waiting time이 있다면 그 원인, 예를 들어 max_num_seqs 에 걸렸거나, max_num_batched_token 에 걸렸거나, eviction 당했거나, preemption 당했거나 (혹시 내가 미처 분류하지 못한 원인이 있다면 알려주고) 
5. iteration 루프가 돌아갈 때, 현재 running queue에 있는 request 및 request의 정보, waiting queue에 있는 request 및 request의 정보, preemption 당한 request가 있다면 그 request. 

일단, 추가로 뽑아보면 좋을 사항들이 있으면 제시해줘. 
이 마크다운 문서 아래에 네 생각을 써줘라.

---

## 추가 제안 사항

### 1. Request-level 추가 메트릭
- **Queue transition history**: Request가 waiting → running → preempted → running 등으로 상태 전이한 전체 히스토리 (timestamp 포함)
- **Total queue time**: Waiting queue에서 보낸 총 시간 (여러 번 preemption 당한 경우 누적)
- **Preemption count**: Request가 몇 번 preemption 당했는지
- **Memory footprint**: Request가 사용하는 KV cache 메모리 양 (bytes 또는 blocks)
- **Priority/scheduling score**: 스케줄러가 사용하는 우선순위 점수 (FCFS, priority-based 등)
- **Completion time**: Request가 완전히 종료된 시간
- **E2E latency**: Arrival부터 completion까지 전체 시간

### 2. Iteration-level 메트릭
- **Timestamp**: 각 iteration의 시작/종료 시간 (nanosecond 또는 microsecond 단위)
- **Batch size**: 현재 iteration에서 함께 처리된 request 개수
- **Total tokens in batch**: Iteration에서 처리한 총 토큰 수
- **Decode step duration**: 실제 모델 forward pass에 걸린 시간
- **Scheduling overhead**: Scheduling 결정에 걸린 시간
- **GPU utilization**: GPU 사용률 (가능하다면)
- **KV cache usage**: 현재 사용 중인 KV cache blocks / 전체 available blocks

### 3. Waiting 원인 상세 분류
이미 언급한 것 외에 추가 원인:
- **Memory insufficient**: KV cache 메모리 부족으로 대기
- **Context length limit**: Max context length 제약으로 대기
- **Decode phase priority**: Prefill보다 decode를 우선하는 정책으로 대기
- **Chunked prefill**: Prefill이 여러 chunk로 나뉘어 처리되는 경우
- **Recomputation**: Eviction 후 KV cache 재계산이 필요한 경우

### 4. System-level 메트릭 (각 iteration마다)
- **Total running requests**: Running queue 크기
- **Total waiting requests**: Waiting queue 크기
- **Total preempted requests**: Preempted/swapped requests 개수
- **Available KV cache blocks**: 남은 KV cache 메모리
- **Scheduler decision**: 이번 iteration에서 어떤 결정을 내렸는지 (admit new requests, preempt, evict 등)

### 5. Fairness 및 분석 메트릭
- **Position in queue**: Waiting queue에서의 위치 변화 추적
- **Starvation time**: Request가 waiting queue에서 대기한 최대 연속 시간
- **Service rate**: 단위 시간당 처리된 토큰 수

### 6. 출력 형식 제안
- **per_request.csv**: Request별 요약 통계 (arrival, completion, TTFT, TPOT, total latency 등)
- **per_iteration.csv**: 각 iteration에서 각 request의 상태 (running/waiting/preempted, queue position, token counts)
- **system_state.csv**: 각 iteration의 시스템 전체 상태 (queue sizes, memory usage, batch composition)
- **events.csv**: 모든 이벤트 로그 (request arrival, admission, preemption, eviction, completion)

### 7. 시간 단위
- Python `time.perf_counter()` 또는 `time.perf_counter_ns()` 사용 (nanosecond 단위 권장)
- 상대 시간 (스케줄러 시작 시점을 0으로) 및 절대 timestamp 모두 기록

### 8. 구현 팁
- 로깅이 스케줄러 성능에 영향을 주지 않도록 비동기 로깅 고려
- CSV 파일 크기가 커질 수 있으므로 버퍼링 및 주기적 flush
- Request ID를 명확히 tracking (vLLM의 request_id 활용) 