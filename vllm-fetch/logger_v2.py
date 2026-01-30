# /vllm/vllm/v1/core/sched/logger.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# NOTE, hyunnnchoi, 2026.01.08
# vLLM v1 스케줄러의 상세 로깅 시스템 구현
# 연구 목적으로 request-level, iteration-level 메트릭 수집

import csv
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


class WaitingReason(Enum):
    """Request가 waiting 상태일 때의 원인"""

    MAX_NUM_SEQS = "max_num_seqs"
    MAX_NUM_BATCHED_TOKENS = "max_num_batched_tokens"
    MEMORY_INSUFFICIENT = "memory_insufficient"
    CONTEXT_LENGTH_LIMIT = "context_length_limit"
    DECODE_PRIORITY = "decode_priority"
    CHUNKED_PREFILL = "chunked_prefill"
    PREEMPTED = "preempted"
    EVICTED = "evicted"
    RECOMPUTATION = "recomputation"
    INITIAL_WAITING = "initial_waiting"
    UNKNOWN = "unknown"


@dataclass
class RequestMetrics:
    """Request별 메트릭을 추적하는 데이터 클래스"""

    request_id: str
    arrival_time: int  # nanosecond
    input_token_count: int

    # Timing information
    first_scheduled_time: int = 0
    completion_time: int = 0
    waiting_time: int = 0  # 누적
    execution_time: int = 0  # 누적

    # Token information
    output_token_count: int = 0
    current_iteration: int = 0

    # Performance metrics
    ttft: int = 0  # Time to first token (nanoseconds)

    # Memory and scheduling
    kv_cache_blocks: int = 0
    kv_cache_memory_bytes: int = 0  # KV cache 메모리 양 (bytes)
    preemption_count: int = 0
    eviction_count: int = 0
    priority_score: int = 0  # 우선순위 점수

    # State tracking
    state_transitions: list[dict[str, Any]] = field(default_factory=list)
    current_state: str = "waiting"
    state_start_time: int = 0

    # Waiting reason tracking
    waiting_reasons: list[str] = field(default_factory=list)
    current_waiting_reason: str = ""


@dataclass
class IterationMetrics:
    """Iteration별 메트릭"""

    iteration_id: int
    iteration_start_time: int
    iteration_end_time: int = 0
    iteration_duration: int = 0
    scheduling_overhead: int = 0

    # NOTE, hyunnnchoi, 2026.01.26
    # Forward pass와 output processing 시간 breakdown
    # forward_pass_duration: schedule() 종료 ~ update_from_output() 시작
    # output_processing_duration: update_from_output() 시작 ~ 다음 begin_iteration()
    forward_pass_duration: int = 0
    output_processing_duration: int = 0

    # Batch information
    batch_size: int = 0
    total_tokens_in_batch: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0

    # Queue state
    running_requests: list[str] = field(default_factory=list)
    waiting_requests: list[str] = field(default_factory=list)
    preempted_requests: list[str] = field(default_factory=list)
    admitted_requests: list[str] = field(default_factory=list)
    completed_requests: list[str] = field(default_factory=list)

    # System resources
    kv_cache_used: int = 0
    kv_cache_total: int = 0

    # Scheduler decision
    scheduler_action: str = "continue"


@dataclass
class RequestStateInIteration:
    """각 iteration에서 request의 상태"""

    iteration_id: int
    request_id: str
    state: str
    queue_position: int
    waiting_reason: str
    tokens_generated_so_far: int
    kv_cache_blocks: int
    time_in_current_state: int


@dataclass
class SchedulerEvent:
    """스케줄러 이벤트"""

    timestamp: int
    event_type: str
    request_id: str
    details: str = ""


class SchedulerLogger:
    """
    vLLM v1 스케줄러의 상세 로깅 시스템

    Request-level, iteration-level 메트릭을 수집하고 CSV 형식으로 출력
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        enabled: bool = False,
        flush_interval_iterations: int = 100,
        flush_interval_seconds: float = 60.0,
    ):
        self.enabled = enabled
        if not self.enabled:
            return

        # 로그 디렉토리 설정
        if log_dir is None:
            log_dir = os.getenv("VLLM_SCHEDULER_LOG_DIR", "./scheduler_logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Scheduler logging enabled. Log directory: {self.log_dir}")

        # 시작 시간 (상대 시간 계산용)
        self.start_time_ns = time.perf_counter_ns()
        self.start_time_abs = time.time()

        # Flush 설정
        self.flush_interval_iterations = flush_interval_iterations
        self.flush_interval_seconds = flush_interval_seconds
        self.last_flush_time = time.time()

        # 메트릭 저장소
        self.request_metrics: dict[str, RequestMetrics] = {}
        self.iteration_metrics: list[IterationMetrics] = []
        self.iteration_request_states: list[RequestStateInIteration] = []
        self.events: list[SchedulerEvent] = []

        # 현재 iteration 추적
        self.current_iteration_id = 0
        self.current_iteration_metrics: Optional[IterationMetrics] = None

        # NOTE, hyunnnchoi, 2026.01.26
        # Forward pass / Output processing 시간 측정을 위한 타임스탬프
        # _prev_iteration_end_time: 이전 iteration의 end_iteration() 호출 시점
        # _output_begin_time: 현재 iteration의 update_from_output() 시작 시점
        self._prev_iteration_end_time: int = 0
        self._output_begin_time: int = 0

        # CSV 파일 초기화
        self._init_csv_files()

        logger.info("SchedulerLogger initialized successfully")

    def _init_csv_files(self) -> None:
        """CSV 파일 헤더 초기화"""
        # per_request.csv
        self.per_request_file = open(self.log_dir / "per_request.csv", "w", newline="")
        self.per_request_writer = csv.writer(self.per_request_file)
        self.per_request_writer.writerow(
            [
                "request_id",
                "arrival_time",
                "first_scheduled_time",
                "completion_time",
                "waiting_time",
                "execution_time",
                "e2e_latency",
                "input_tokens",
                "output_tokens",
                "ttft",
                "tpot",
                "throughput",
                "preemption_count",
                "eviction_count",
                "kv_cache_blocks",
                "kv_cache_memory_bytes",
                "priority_score",
            ]
        )

        # state_transitions.csv (별도 파일로 출력)
        self.state_transitions_file = open(
            self.log_dir / "state_transitions.csv", "w", newline=""
        )
        self.state_transitions_writer = csv.writer(self.state_transitions_file)
        self.state_transitions_writer.writerow(
            ["request_id", "timestamp", "state", "duration_in_prev_state"]
        )

        # per_iteration.csv
        # NOTE, hyunnnchoi, 2026.01.26
        # forward_pass_duration, output_processing_duration 컬럼 추가
        self.per_iteration_file = open(
            self.log_dir / "per_iteration.csv", "w", newline=""
        )
        self.per_iteration_writer = csv.writer(self.per_iteration_file)
        self.per_iteration_writer.writerow(
            [
                "iteration_id",
                "start_time",
                "end_time",
                "duration",
                "scheduling_overhead",
                "forward_pass_duration",
                "output_processing_duration",
                "batch_size",
                "total_tokens",
                "prefill_tokens",
                "decode_tokens",
                "kv_cache_used",
                "kv_cache_total",
                "running_count",
                "waiting_count",
                "preempted_count",
                "scheduler_action",
            ]
        )

        # per_iteration_requests.csv
        self.per_iteration_requests_file = open(
            self.log_dir / "per_iteration_requests.csv", "w", newline=""
        )
        self.per_iteration_requests_writer = csv.writer(
            self.per_iteration_requests_file
        )
        self.per_iteration_requests_writer.writerow(
            [
                "iteration_id",
                "request_id",
                "state",
                "queue_position",
                "waiting_reason",
                "tokens_generated",
                "kv_cache_blocks",
                "time_in_state",
            ]
        )

        # events.csv
        self.events_file = open(self.log_dir / "events.csv", "w", newline="")
        self.events_writer = csv.writer(self.events_file)
        self.events_writer.writerow(
            ["timestamp", "event_type", "request_id", "details"]
        )

    def _get_relative_time_ns(self) -> int:
        """상대 시간 (nanosecond)"""
        return time.perf_counter_ns() - self.start_time_ns

    def record_request_arrival(self, request: Request) -> None:
        """Request 도착 기록"""
        if not self.enabled:
            return

        arrival_time = self._get_relative_time_ns()
        metrics = RequestMetrics(
            request_id=request.request_id,
            arrival_time=arrival_time,
            input_token_count=request.num_prompt_tokens,
            state_start_time=arrival_time,
        )
        self.request_metrics[request.request_id] = metrics

        # 이벤트 기록
        self._record_event("request_arrived", request.request_id)

    def record_request_scheduled(self, request_id: str) -> None:
        """Request가 scheduled된 시점 기록

        NOTE, hyunnnchoi, 2026.01.26
        first_scheduled_time 업데이트와 상태 전이(running)를 분리하여,
        Preemption 후 재개(Resume)된 요청도 상태 전이가 정확히 기록되도록 수정.
        이전에는 first_scheduled_time 조건문 내부에서만 _update_state가 호출되어
        재개된 요청의 execution_time이 정확하게 계산되지 않는 버그가 있었음.
        """
        if not self.enabled:
            return

        metrics = self.request_metrics.get(request_id)
        if not metrics:
            return

        # NOTE, hyunnnchoi, 2026.01.26
        # first_scheduled_time은 최초 한 번만 기록
        if metrics.first_scheduled_time == 0:
            metrics.first_scheduled_time = self._get_relative_time_ns()
            self._record_event("request_scheduled", request_id)

        # NOTE, hyunnnchoi, 2026.01.26
        # 상태 전이는 매번 호출되어야 함 (Preemption 후 Resume 포함)
        # 이미 running 상태인 경우는 중복 전이를 방지
        if metrics.current_state != "running":
            self._record_event("request_resumed", request_id)
            self._update_state(request_id, "running")

    def record_first_token_generated(self, request_id: str) -> None:
        """첫 번째 토큰 생성 시점 기록 (TTFT 계산)"""
        if not self.enabled:
            return

        metrics = self.request_metrics.get(request_id)
        if metrics and metrics.ttft == 0:
            current_time = self._get_relative_time_ns()
            metrics.ttft = current_time - metrics.arrival_time
            self._record_event("first_token_generated", request_id)

    def record_request_preempted(self, request_id: str) -> None:
        """Request preemption 기록"""
        if not self.enabled:
            return

        metrics = self.request_metrics.get(request_id)
        if metrics:
            metrics.preemption_count += 1
            self._update_state(request_id, "preempted")
            self._record_event("request_preempted", request_id)

    def record_request_completed(self, request_id: str, output_tokens: int) -> None:
        """Request 완료 기록"""
        if not self.enabled:
            return

        metrics = self.request_metrics.get(request_id)
        if metrics:
            metrics.completion_time = self._get_relative_time_ns()
            metrics.output_token_count = output_tokens
            self._update_state(request_id, "completed")
            self._record_event("request_completed", request_id)

    def record_output_begin(self) -> None:
        """Output processing 시작 시점 기록

        NOTE, hyunnnchoi, 2026.01.26
        update_from_output() 시작 시점에 호출하여 forward_pass_duration 계산에 사용.
        forward_pass_duration = output_begin_time - prev_iteration_end_time
        """
        if not self.enabled:
            return

        current_time = self._get_relative_time_ns()
        self._output_begin_time = current_time

        # Forward pass duration 계산 (이전 iteration_end ~ 현재 output_begin)
        if self._prev_iteration_end_time > 0 and len(self.iteration_metrics) > 0:
            prev_metrics = self.iteration_metrics[-1]
            prev_metrics.forward_pass_duration = (
                current_time - self._prev_iteration_end_time
            )

    def begin_iteration(self) -> None:
        """Iteration 시작"""
        if not self.enabled:
            return

        current_time = self._get_relative_time_ns()

        # NOTE, hyunnnchoi, 2026.01.26
        # 이전 iteration의 output_processing_duration 계산
        # (이전 output_begin ~ 현재 iteration_begin 사이의 시간)
        if self._output_begin_time > 0 and len(self.iteration_metrics) > 0:
            prev_metrics = self.iteration_metrics[-1]
            prev_metrics.output_processing_duration = (
                current_time - self._output_begin_time
            )

        self.current_iteration_metrics = IterationMetrics(
            iteration_id=self.current_iteration_id,
            iteration_start_time=current_time,
        )

    def end_iteration(
        self,
        running_requests: list[str],
        waiting_requests: list[str],
        kv_cache_used: int,
        kv_cache_total: int,
        num_scheduled_tokens: dict[str, int],
        preempted_requests: list[str] | None = None,
        admitted_requests: list[str] | None = None,
        completed_requests: list[str] | None = None,
        scheduler_action: str = "continue",
        prefill_decode_info: dict[str, tuple[int, int]] | None = None,
    ) -> None:
        """Iteration 종료 및 메트릭 기록

        Args:
            running_requests: 현재 running 상태인 request ID 리스트
            waiting_requests: 현재 waiting 상태인 request ID 리스트
            kv_cache_used: 사용 중인 KV cache blocks 수
            kv_cache_total: 전체 KV cache blocks 수
            num_scheduled_tokens: request별 scheduled 토큰 수
            preempted_requests: 이번 iteration에서 preemption된 request ID 리스트
            admitted_requests: 이번 iteration에서 새로 admitted된 request ID 리스트
            completed_requests: 이번 iteration에서 완료된 request ID 리스트
            scheduler_action: 이번 iteration의 주요 스케줄러 결정
            prefill_decode_info: request별 (prefill_tokens, decode_tokens) 튜플
        """
        if not self.enabled or self.current_iteration_metrics is None:
            return

        current_time = self._get_relative_time_ns()
        metrics = self.current_iteration_metrics
        metrics.iteration_end_time = current_time
        metrics.iteration_duration = current_time - metrics.iteration_start_time

        # Queue 상태
        metrics.running_requests = running_requests.copy()
        metrics.waiting_requests = waiting_requests.copy()
        metrics.batch_size = len(running_requests)
        metrics.preempted_requests = (
            preempted_requests.copy() if preempted_requests else []
        )
        metrics.admitted_requests = (
            admitted_requests.copy() if admitted_requests else []
        )
        metrics.completed_requests = (
            completed_requests.copy() if completed_requests else []
        )
        metrics.scheduler_action = scheduler_action

        # KV cache 상태
        metrics.kv_cache_used = kv_cache_used
        metrics.kv_cache_total = kv_cache_total

        # 토큰 수 계산
        total_tokens = sum(num_scheduled_tokens.values())
        metrics.total_tokens_in_batch = total_tokens

        # Prefill/Decode 토큰 분리
        if prefill_decode_info:
            for req_id in num_scheduled_tokens.keys():
                if req_id in prefill_decode_info:
                    prefill, decode = prefill_decode_info[req_id]
                    metrics.prefill_tokens += prefill
                    metrics.decode_tokens += decode

        # NOTE, hyunnnchoi, 2026.01.26
        # iteration 종료 시점 기록 (forward pass 시간 계산에 사용)
        self._prev_iteration_end_time = current_time

        # 메트릭 저장
        self.iteration_metrics.append(metrics)

        # Request 상태 기록
        self._record_request_states_in_iteration(
            running_requests, waiting_requests, num_scheduled_tokens
        )

        # 다음 iteration 준비
        self.current_iteration_id += 1
        self.current_iteration_metrics = None

        # 주기적으로 flush
        self._check_and_flush()

    def _record_request_states_in_iteration(
        self,
        running_requests: list[str],
        waiting_requests: list[str],
        num_scheduled_tokens: dict[str, int],
    ) -> None:
        """각 iteration에서 request 상태 기록"""
        current_time = self._get_relative_time_ns()

        # Running requests
        for idx, req_id in enumerate(running_requests):
            metrics = self.request_metrics.get(req_id)
            if metrics:
                state = RequestStateInIteration(
                    iteration_id=self.current_iteration_id,
                    request_id=req_id,
                    state="running",
                    queue_position=idx,
                    waiting_reason="",
                    tokens_generated_so_far=metrics.output_token_count,
                    kv_cache_blocks=metrics.kv_cache_blocks,
                    time_in_current_state=current_time - metrics.state_start_time,
                )
                self.iteration_request_states.append(state)

        # Waiting requests
        for idx, req_id in enumerate(waiting_requests):
            metrics = self.request_metrics.get(req_id)
            if metrics:
                # 현재 waiting reason 사용 (가장 최근 것)
                waiting_reason = metrics.current_waiting_reason or (
                    metrics.waiting_reasons[-1]
                    if metrics.waiting_reasons
                    else "unknown"
                )
                state = RequestStateInIteration(
                    iteration_id=self.current_iteration_id,
                    request_id=req_id,
                    state="waiting",
                    queue_position=idx,
                    waiting_reason=waiting_reason,
                    tokens_generated_so_far=metrics.output_token_count,
                    kv_cache_blocks=metrics.kv_cache_blocks,
                    time_in_current_state=current_time - metrics.state_start_time,
                )
                self.iteration_request_states.append(state)

    def set_waiting_reason(self, request_id: str, reason: WaitingReason) -> None:
        """Waiting 원인 설정"""
        if not self.enabled:
            return

        metrics = self.request_metrics.get(request_id)
        if metrics:
            metrics.waiting_reasons.append(reason.value)
            metrics.current_waiting_reason = reason.value

    def update_kv_cache_blocks(
        self, request_id: str, num_blocks: int, block_size: int = 0
    ) -> None:
        """KV cache block 수 및 메모리 업데이트"""
        if not self.enabled:
            return

        metrics = self.request_metrics.get(request_id)
        if metrics:
            metrics.kv_cache_blocks = num_blocks
            # KV cache 메모리 계산 (block_size와 hidden_size 기반)
            # 간단한 추정: num_blocks * block_size * hidden_dim * 2 (K, V) * dtype_size
            # 실제로는 더 복잡하지만, 기본 추정치로 사용
            if block_size > 0:
                # Assume FP16 (2 bytes), typical hidden_dim = 4096, K and V
                # This is a rough estimate
                metrics.kv_cache_memory_bytes = num_blocks * block_size * 4096 * 2 * 2

    def update_priority(self, request_id: str, priority: int) -> None:
        """Request 우선순위 업데이트"""
        if not self.enabled:
            return

        metrics = self.request_metrics.get(request_id)
        if metrics:
            metrics.priority_score = priority

    def _update_state(self, request_id: str, new_state: str) -> None:
        """Request 상태 전이 기록"""
        metrics = self.request_metrics.get(request_id)
        if not metrics:
            return

        current_time = self._get_relative_time_ns()

        # 이전 상태에서 보낸 시간 계산
        time_in_state = current_time - metrics.state_start_time

        # 상태 전이 기록
        transition = {
            "time": current_time,
            "state": new_state,
            "duration_in_prev_state": time_in_state,
        }
        metrics.state_transitions.append(transition)

        # NOTE, hyunnnchoi, 2026.01.26
        # Waiting/execution time 누적
        # 'preempted' 상태도 'waiting'과 마찬가지로 대기 시간에 포함되어야 함.
        # Preemption 후 재개되기까지의 시간도 요청 관점에서는 대기 시간이기 때문.
        if metrics.current_state in ("waiting", "preempted"):
            metrics.waiting_time += time_in_state
        elif metrics.current_state == "running":
            metrics.execution_time += time_in_state

        # 새 상태로 업데이트
        metrics.current_state = new_state
        metrics.state_start_time = current_time

    def _record_event(
        self, event_type: str, request_id: str, details: str = ""
    ) -> None:
        """이벤트 기록"""
        event = SchedulerEvent(
            timestamp=self._get_relative_time_ns(),
            event_type=event_type,
            request_id=request_id,
            details=details,
        )
        self.events.append(event)

    def _check_and_flush(self) -> None:
        """주기적으로 CSV 파일에 flush"""
        should_flush = False

        # Iteration 기반 flush
        if self.current_iteration_id % self.flush_interval_iterations == 0:
            should_flush = True

        # 시간 기반 flush
        current_time = time.time()
        if current_time - self.last_flush_time >= self.flush_interval_seconds:
            should_flush = True

        if should_flush:
            self.flush()
            self.last_flush_time = current_time

    def flush(self) -> None:
        """CSV 파일에 데이터 쓰기"""
        if not self.enabled:
            return

        # NOTE, hyunnnchoi, 2026.01.26
        # Iteration 메트릭 쓰기 (forward_pass_duration, output_processing_duration 추가)
        for metrics in self.iteration_metrics:
            self.per_iteration_writer.writerow(
                [
                    metrics.iteration_id,
                    metrics.iteration_start_time,
                    metrics.iteration_end_time,
                    metrics.iteration_duration,
                    metrics.scheduling_overhead,
                    metrics.forward_pass_duration,
                    metrics.output_processing_duration,
                    metrics.batch_size,
                    metrics.total_tokens_in_batch,
                    metrics.prefill_tokens,
                    metrics.decode_tokens,
                    metrics.kv_cache_used,
                    metrics.kv_cache_total,
                    len(metrics.running_requests),
                    len(metrics.waiting_requests),
                    len(metrics.preempted_requests),
                    metrics.scheduler_action,
                ]
            )
        self.iteration_metrics.clear()

        # Iteration별 request 상태 쓰기
        for state in self.iteration_request_states:
            self.per_iteration_requests_writer.writerow(
                [
                    state.iteration_id,
                    state.request_id,
                    state.state,
                    state.queue_position,
                    state.waiting_reason,
                    state.tokens_generated_so_far,
                    state.kv_cache_blocks,
                    state.time_in_current_state,
                ]
            )
        self.iteration_request_states.clear()

        # 이벤트 쓰기
        for event in self.events:
            self.events_writer.writerow(
                [
                    event.timestamp,
                    event.event_type,
                    event.request_id,
                    event.details,
                ]
            )
        self.events.clear()

        # 파일 flush
        self.per_iteration_file.flush()
        self.per_iteration_requests_file.flush()
        self.events_file.flush()

    def finalize_request(self, request_id: str) -> None:
        """Request 완료 시 최종 통계 계산 및 기록"""
        if not self.enabled:
            return

        metrics = self.request_metrics.get(request_id)
        if not metrics:
            return

        # E2E latency 계산
        e2e_latency = metrics.completion_time - metrics.arrival_time

        # TPOT 계산 (Time Per Output Token)
        tpot = 0
        if metrics.output_token_count > 1:
            time_after_first_token = e2e_latency - metrics.ttft
            tpot = time_after_first_token // (metrics.output_token_count - 1)

        # Throughput 계산 (tokens/sec)
        throughput = 0.0
        if e2e_latency > 0:
            throughput = (metrics.output_token_count * 1_000_000_000) / e2e_latency

        # CSV에 쓰기 (새로운 필드 포함)
        self.per_request_writer.writerow(
            [
                metrics.request_id,
                metrics.arrival_time,
                metrics.first_scheduled_time,
                metrics.completion_time,
                metrics.waiting_time,
                metrics.execution_time,
                e2e_latency,
                metrics.input_token_count,
                metrics.output_token_count,
                metrics.ttft,
                tpot,
                f"{throughput:.2f}",
                metrics.preemption_count,
                metrics.eviction_count,
                metrics.kv_cache_blocks,
                metrics.kv_cache_memory_bytes,
                metrics.priority_score,
            ]
        )
        self.per_request_file.flush()

        # State transitions를 별도 파일에 기록
        for transition in metrics.state_transitions:
            self.state_transitions_writer.writerow(
                [
                    metrics.request_id,
                    transition["time"],
                    transition["state"],
                    transition["duration_in_prev_state"],
                ]
            )
        self.state_transitions_file.flush()

    def shutdown(self) -> None:
        """로거 종료 및 파일 닫기"""
        if not self.enabled:
            return

        logger.info("Shutting down SchedulerLogger and flushing remaining data")

        # 남은 데이터 flush
        self.flush()

        # 파일 닫기
        self.per_request_file.close()
        self.per_iteration_file.close()
        self.per_iteration_requests_file.close()
        self.events_file.close()
        self.state_transitions_file.close()

        logger.info(f"Scheduler logs saved to {self.log_dir}")
