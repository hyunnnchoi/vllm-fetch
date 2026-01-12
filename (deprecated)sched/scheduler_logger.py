# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Scheduler Logging Module for vLLM v1

This module provides comprehensive logging for the vLLM v1 scheduler,
capturing request-level, iteration-level, and event-level metrics
for performance analysis and research purposes.

Usage:
    export VLLM_SCHEDULER_LOGGING=1
    export VLLM_SCHEDULER_LOG_DIR=./scheduler_logs
"""

from __future__ import annotations

import atexit
import csv
import json
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class WaitingReason(str, Enum):
    """Reasons why a request is in waiting state."""
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
    WAITING_FOR_FSM = "waiting_for_fsm"
    WAITING_FOR_REMOTE_KVS = "waiting_for_remote_kvs"
    LORA_LIMIT = "lora_limit"
    ENCODER_BUDGET = "encoder_budget"
    UNKNOWN = "unknown"


class SchedulerEventType(str, Enum):
    """Types of scheduler events."""
    REQUEST_ARRIVED = "request_arrived"
    REQUEST_ADMITTED = "request_admitted"
    REQUEST_SCHEDULED = "request_scheduled"
    REQUEST_PREEMPTED = "request_preempted"
    REQUEST_EVICTED = "request_evicted"
    REQUEST_COMPLETED = "request_completed"
    FIRST_TOKEN_GENERATED = "first_token_generated"
    KV_CACHE_ALLOCATED = "kv_cache_allocated"
    KV_CACHE_FREED = "kv_cache_freed"


class RequestState(str, Enum):
    """State of a request in the scheduler."""
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    COMPLETED = "completed"


@dataclass
class StateTransition:
    """Records a state transition for a request."""
    timestamp_ns: int
    from_state: str
    to_state: str
    reason: Optional[str] = None


@dataclass
class RequestMetrics:
    """Metrics tracked for each request."""
    request_id: str
    arrival_time_ns: int
    first_scheduled_time_ns: Optional[int] = None
    completion_time_ns: Optional[int] = None

    # Token information
    input_token_count: int = 0
    output_token_count: int = 0

    # Performance metrics
    ttft_ns: Optional[int] = None  # Time to first token

    # Memory and scheduling info
    kv_cache_blocks: int = 0
    preemption_count: int = 0
    eviction_count: int = 0

    # Waiting time tracking
    waiting_start_time_ns: Optional[int] = None
    total_waiting_time_ns: int = 0

    # Running time tracking
    running_start_time_ns: Optional[int] = None
    total_execution_time_ns: int = 0

    # State transitions
    state_transitions: list[StateTransition] = field(default_factory=list)
    current_state: str = "waiting"
    current_state_start_time_ns: Optional[int] = None

    # Current waiting reason
    waiting_reason: WaitingReason = WaitingReason.INITIAL_WAITING

    # Iteration tracking
    current_iteration: int = 0
    first_token_iteration: Optional[int] = None

    def record_state_transition(self, new_state: str, timestamp_ns: int,
                                reason: Optional[str] = None) -> None:
        """Record a state transition."""
        transition = StateTransition(
            timestamp_ns=timestamp_ns,
            from_state=self.current_state,
            to_state=new_state,
            reason=reason
        )
        self.state_transitions.append(transition)

        # Update timing based on state change
        if self.current_state == RequestState.WAITING.value:
            if self.waiting_start_time_ns is not None:
                self.total_waiting_time_ns += (
                    timestamp_ns - self.waiting_start_time_ns)
                self.waiting_start_time_ns = None
        elif self.current_state == RequestState.RUNNING.value:
            if self.running_start_time_ns is not None:
                self.total_execution_time_ns += (
                    timestamp_ns - self.running_start_time_ns)
                self.running_start_time_ns = None

        # Start timing for new state
        if new_state == RequestState.WAITING.value:
            self.waiting_start_time_ns = timestamp_ns
        elif new_state == RequestState.RUNNING.value:
            self.running_start_time_ns = timestamp_ns
            if self.first_scheduled_time_ns is None:
                self.first_scheduled_time_ns = timestamp_ns

        self.current_state = new_state
        self.current_state_start_time_ns = timestamp_ns

    def calculate_e2e_latency_ns(self) -> Optional[int]:
        """Calculate end-to-end latency in nanoseconds."""
        if self.completion_time_ns is not None:
            return self.completion_time_ns - self.arrival_time_ns
        return None

    def calculate_tpot_ns(self) -> Optional[float]:
        """Calculate time per output token in nanoseconds."""
        if self.output_token_count > 1 and self.ttft_ns is not None:
            e2e = self.calculate_e2e_latency_ns()
            if e2e is not None:
                decode_time = e2e - self.ttft_ns
                # Exclude first token from TPOT calculation
                return decode_time / (self.output_token_count - 1)
        return None

    def calculate_throughput(self) -> Optional[float]:
        """Calculate throughput in tokens/sec."""
        e2e = self.calculate_e2e_latency_ns()
        if e2e is not None and e2e > 0 and self.output_token_count > 0:
            return self.output_token_count / (e2e / 1e9)
        return None


@dataclass
class IterationMetrics:
    """Metrics for each scheduler iteration."""
    iteration_id: int
    start_time_ns: int
    end_time_ns: Optional[int] = None
    scheduling_overhead_ns: Optional[int] = None
    forward_pass_duration_ns: Optional[int] = None

    # Batch information
    batch_size: int = 0
    total_tokens_in_batch: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0

    # Queue state
    running_request_ids: list[str] = field(default_factory=list)
    waiting_request_ids: list[str] = field(default_factory=list)
    preempted_request_ids: list[str] = field(default_factory=list)
    admitted_request_ids: list[str] = field(default_factory=list)
    completed_request_ids: list[str] = field(default_factory=list)

    # System resources
    kv_cache_used_blocks: int = 0
    kv_cache_total_blocks: int = 0

    # Scheduler decision
    scheduler_actions: list[str] = field(default_factory=list)

    @property
    def duration_ns(self) -> Optional[int]:
        if self.end_time_ns is not None:
            return self.end_time_ns - self.start_time_ns
        return None

    @property
    def kv_cache_utilization(self) -> float:
        if self.kv_cache_total_blocks > 0:
            return self.kv_cache_used_blocks / self.kv_cache_total_blocks
        return 0.0


@dataclass
class PerIterationRequestState:
    """State of a request at a specific iteration."""
    iteration_id: int
    request_id: str
    state: str
    queue_position: int = -1
    waiting_reason: str = ""
    tokens_generated: int = 0
    kv_cache_blocks: int = 0
    time_in_state_ns: int = 0


@dataclass
class SchedulerEvent:
    """An event in the scheduler."""
    timestamp_ns: int
    event_type: SchedulerEventType
    request_id: str
    details: dict[str, Any] = field(default_factory=dict)


class SchedulerLogger:
    """
    Logger for vLLM v1 scheduler metrics.

    Collects request-level, iteration-level, and event-level metrics
    and writes them to CSV files for analysis.
    """

    # CSV headers
    REQUEST_CSV_HEADERS = [
        "request_id", "arrival_time", "first_scheduled_time", "completion_time",
        "waiting_time", "execution_time", "e2e_latency", "input_tokens",
        "output_tokens", "ttft", "tpot", "throughput", "preemption_count",
        "eviction_count", "kv_cache_blocks"
    ]

    ITERATION_CSV_HEADERS = [
        "iteration_id", "start_time", "end_time", "duration",
        "scheduling_overhead", "forward_duration", "batch_size", "total_tokens",
        "prefill_tokens", "decode_tokens", "kv_cache_used", "kv_cache_total",
        "running_count", "waiting_count", "preempted_count", "scheduler_action"
    ]

    PER_ITERATION_REQUEST_CSV_HEADERS = [
        "iteration_id", "request_id", "state", "queue_position",
        "waiting_reason", "tokens_generated", "kv_cache_blocks", "time_in_state"
    ]

    EVENT_CSV_HEADERS = [
        "timestamp", "event_type", "request_id", "details"
    ]

    def __init__(
        self,
        log_dir: Optional[str] = None,
        buffer_size: int = 100,
        enabled: bool = True,
    ) -> None:
        self.enabled = enabled
        if not self.enabled:
            return

        # Set up log directory
        if log_dir is None:
            log_dir = os.environ.get(
                "VLLM_SCHEDULER_LOG_DIR", "./scheduler_logs")
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Buffer settings
        self.buffer_size = buffer_size

        # Start time for relative timestamps
        self.start_time_ns = time.perf_counter_ns()

        # Request metrics storage
        self.request_metrics: dict[str, RequestMetrics] = {}

        # Iteration counter
        self.iteration_id = 0

        # Current iteration metrics
        self.current_iteration: Optional[IterationMetrics] = None

        # Buffers for batch writing
        self.request_buffer: deque[RequestMetrics] = deque()
        self.iteration_buffer: deque[IterationMetrics] = deque()
        self.per_iteration_request_buffer: deque[PerIterationRequestState] = (
            deque())
        self.event_buffer: deque[SchedulerEvent] = deque()

        # Thread lock for concurrent access
        self._lock = threading.Lock()

        # Initialize CSV files
        self._init_csv_files()

        # Register cleanup on exit
        atexit.register(self.flush_all)

        logger.info("SchedulerLogger initialized. Log directory: %s",
                    self.log_dir)

    def _init_csv_files(self) -> None:
        """Initialize CSV files with headers."""
        self._write_csv_header(
            self.log_dir / "per_request.csv",
            self.REQUEST_CSV_HEADERS
        )
        self._write_csv_header(
            self.log_dir / "per_iteration.csv",
            self.ITERATION_CSV_HEADERS
        )
        self._write_csv_header(
            self.log_dir / "per_iteration_requests.csv",
            self.PER_ITERATION_REQUEST_CSV_HEADERS
        )
        self._write_csv_header(
            self.log_dir / "events.csv",
            self.EVENT_CSV_HEADERS
        )

    def _write_csv_header(self, filepath: Path, headers: list[str]) -> None:
        """Write CSV header to file."""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def _get_relative_time_ns(self) -> int:
        """Get current time relative to start time in nanoseconds."""
        return time.perf_counter_ns() - self.start_time_ns

    # =========================================================================
    # Request-level logging
    # =========================================================================

    def on_request_arrived(self, request: "Request") -> None:
        """Called when a new request arrives at the scheduler."""
        if not self.enabled:
            return

        timestamp_ns = self._get_relative_time_ns()

        with self._lock:
            metrics = RequestMetrics(
                request_id=request.request_id,
                arrival_time_ns=timestamp_ns,
                input_token_count=request.num_prompt_tokens,
                waiting_start_time_ns=timestamp_ns,
                current_state_start_time_ns=timestamp_ns,
            )
            self.request_metrics[request.request_id] = metrics

        self._log_event(
            SchedulerEventType.REQUEST_ARRIVED,
            request.request_id,
            {"input_tokens": request.num_prompt_tokens}
        )

    def on_request_scheduled(
        self,
        request: "Request",
        is_first_schedule: bool = False,
        kv_cache_blocks: int = 0,
    ) -> None:
        """Called when a request is scheduled to run."""
        if not self.enabled:
            return

        timestamp_ns = self._get_relative_time_ns()

        with self._lock:
            metrics = self.request_metrics.get(request.request_id)
            if metrics is None:
                return

            metrics.kv_cache_blocks = kv_cache_blocks
            metrics.record_state_transition(
                RequestState.RUNNING.value,
                timestamp_ns,
                reason="scheduled"
            )

            if is_first_schedule:
                metrics.first_scheduled_time_ns = timestamp_ns

        self._log_event(
            SchedulerEventType.REQUEST_SCHEDULED,
            request.request_id,
            {
                "is_first_schedule": is_first_schedule,
                "kv_cache_blocks": kv_cache_blocks
            }
        )

    def on_request_preempted(self, request: "Request") -> None:
        """Called when a request is preempted."""
        if not self.enabled:
            return

        timestamp_ns = self._get_relative_time_ns()

        with self._lock:
            metrics = self.request_metrics.get(request.request_id)
            if metrics is None:
                return

            metrics.preemption_count += 1
            metrics.waiting_reason = WaitingReason.PREEMPTED
            metrics.record_state_transition(
                RequestState.PREEMPTED.value,
                timestamp_ns,
                reason="preempted"
            )

        self._log_event(
            SchedulerEventType.REQUEST_PREEMPTED,
            request.request_id,
            {"preemption_count": metrics.preemption_count}
        )

    def on_first_token_generated(
        self,
        request_id: str,
        iteration_id: int,
    ) -> None:
        """Called when the first token is generated for a request."""
        if not self.enabled:
            return

        timestamp_ns = self._get_relative_time_ns()

        with self._lock:
            metrics = self.request_metrics.get(request_id)
            if metrics is None:
                return

            if metrics.ttft_ns is None:
                metrics.ttft_ns = timestamp_ns - metrics.arrival_time_ns
                metrics.first_token_iteration = iteration_id

        self._log_event(
            SchedulerEventType.FIRST_TOKEN_GENERATED,
            request_id,
            {
                "ttft_ns": metrics.ttft_ns,
                "iteration_id": iteration_id
            }
        )

    def on_token_generated(self, request_id: str) -> None:
        """Called when a token is generated for a request."""
        if not self.enabled:
            return

        with self._lock:
            metrics = self.request_metrics.get(request_id)
            if metrics is not None:
                metrics.output_token_count += 1
                metrics.current_iteration = self.iteration_id

    def on_request_completed(
        self,
        request: "Request",
        kv_cache_blocks: int = 0,
    ) -> None:
        """Called when a request is completed."""
        if not self.enabled:
            return

        timestamp_ns = self._get_relative_time_ns()

        with self._lock:
            metrics = self.request_metrics.get(request.request_id)
            if metrics is None:
                return

            metrics.completion_time_ns = timestamp_ns
            metrics.kv_cache_blocks = kv_cache_blocks
            metrics.output_token_count = request.num_output_tokens
            metrics.record_state_transition(
                RequestState.COMPLETED.value,
                timestamp_ns,
                reason="completed"
            )

            # Move to buffer for writing
            self.request_buffer.append(metrics)
            del self.request_metrics[request.request_id]

        self._log_event(
            SchedulerEventType.REQUEST_COMPLETED,
            request.request_id,
            {
                "output_tokens": request.num_output_tokens,
                "e2e_latency_ns": metrics.calculate_e2e_latency_ns()
            }
        )

        self._maybe_flush()

    def update_waiting_reason(
        self,
        request_id: str,
        reason: WaitingReason,
    ) -> None:
        """Update the waiting reason for a request."""
        if not self.enabled:
            return

        with self._lock:
            metrics = self.request_metrics.get(request_id)
            if metrics is not None:
                metrics.waiting_reason = reason

    def on_request_back_to_waiting(
        self,
        request: "Request",
        reason: WaitingReason = WaitingReason.UNKNOWN,
    ) -> None:
        """Called when a request goes back to waiting state."""
        if not self.enabled:
            return

        timestamp_ns = self._get_relative_time_ns()

        with self._lock:
            metrics = self.request_metrics.get(request.request_id)
            if metrics is None:
                return

            metrics.waiting_reason = reason
            metrics.record_state_transition(
                RequestState.WAITING.value,
                timestamp_ns,
                reason=reason.value
            )

    # =========================================================================
    # Iteration-level logging
    # =========================================================================

    def begin_iteration(self) -> int:
        """Called at the beginning of a scheduler iteration."""
        if not self.enabled:
            return self.iteration_id

        timestamp_ns = self._get_relative_time_ns()

        with self._lock:
            self.current_iteration = IterationMetrics(
                iteration_id=self.iteration_id,
                start_time_ns=timestamp_ns,
            )

        return self.iteration_id

    def end_scheduling_phase(self) -> None:
        """Called at the end of the scheduling decision phase."""
        if not self.enabled:
            return

        timestamp_ns = self._get_relative_time_ns()

        with self._lock:
            if self.current_iteration is not None:
                self.current_iteration.scheduling_overhead_ns = (
                    timestamp_ns - self.current_iteration.start_time_ns
                )

    def record_forward_pass_duration(self, duration_ns: int) -> None:
        """Record the forward pass duration."""
        if not self.enabled:
            return

        with self._lock:
            if self.current_iteration is not None:
                self.current_iteration.forward_pass_duration_ns = duration_ns

    def end_iteration(
        self,
        running_requests: list["Request"],
        waiting_request_ids: list[str],
        num_scheduled_tokens: dict[str, int],
        preempted_request_ids: list[str],
        admitted_request_ids: list[str],
        completed_request_ids: list[str],
        kv_cache_used_blocks: int,
        kv_cache_total_blocks: int,
        scheduler_actions: list[str],
    ) -> None:
        """Called at the end of a scheduler iteration."""
        if not self.enabled:
            return

        timestamp_ns = self._get_relative_time_ns()

        with self._lock:
            if self.current_iteration is None:
                return

            iteration = self.current_iteration
            iteration.end_time_ns = timestamp_ns

            # Batch information
            iteration.batch_size = len(running_requests)
            iteration.total_tokens_in_batch = sum(num_scheduled_tokens.values())

            # Calculate prefill vs decode tokens
            for req in running_requests:
                req_id = req.request_id
                if req_id in num_scheduled_tokens:
                    num_tokens = num_scheduled_tokens[req_id]
                    metrics = self.request_metrics.get(req_id)
                    if metrics is not None:
                        if metrics.output_token_count == 0:
                            iteration.prefill_tokens += num_tokens
                        else:
                            iteration.decode_tokens += num_tokens

            # Queue state
            iteration.running_request_ids = [r.request_id
                                             for r in running_requests]
            iteration.waiting_request_ids = waiting_request_ids
            iteration.preempted_request_ids = preempted_request_ids
            iteration.admitted_request_ids = admitted_request_ids
            iteration.completed_request_ids = completed_request_ids

            # System resources
            iteration.kv_cache_used_blocks = kv_cache_used_blocks
            iteration.kv_cache_total_blocks = kv_cache_total_blocks

            # Scheduler actions
            iteration.scheduler_actions = scheduler_actions

            # Record per-iteration request states
            self._record_per_iteration_request_states(
                iteration, running_requests, waiting_request_ids)

            # Move to buffer
            self.iteration_buffer.append(iteration)
            self.iteration_id += 1
            self.current_iteration = None

        self._maybe_flush()

    def _record_per_iteration_request_states(
        self,
        iteration: IterationMetrics,
        running_requests: list["Request"],
        waiting_request_ids: list[str],
    ) -> None:
        """Record the state of each request at this iteration."""
        timestamp_ns = self._get_relative_time_ns()

        # Running requests
        for req in running_requests:
            metrics = self.request_metrics.get(req.request_id)
            if metrics is None:
                continue

            time_in_state = 0
            if metrics.current_state_start_time_ns is not None:
                time_in_state = timestamp_ns - metrics.current_state_start_time_ns

            state = PerIterationRequestState(
                iteration_id=iteration.iteration_id,
                request_id=req.request_id,
                state=RequestState.RUNNING.value,
                queue_position=-1,
                waiting_reason="",
                tokens_generated=metrics.output_token_count,
                kv_cache_blocks=metrics.kv_cache_blocks,
                time_in_state_ns=time_in_state,
            )
            self.per_iteration_request_buffer.append(state)

        # Waiting requests
        for idx, req_id in enumerate(waiting_request_ids):
            metrics = self.request_metrics.get(req_id)
            if metrics is None:
                continue

            time_in_state = 0
            if metrics.current_state_start_time_ns is not None:
                time_in_state = timestamp_ns - metrics.current_state_start_time_ns

            state = PerIterationRequestState(
                iteration_id=iteration.iteration_id,
                request_id=req_id,
                state=RequestState.WAITING.value,
                queue_position=idx,
                waiting_reason=metrics.waiting_reason.value,
                tokens_generated=metrics.output_token_count,
                kv_cache_blocks=metrics.kv_cache_blocks,
                time_in_state_ns=time_in_state,
            )
            self.per_iteration_request_buffer.append(state)

    # =========================================================================
    # Event logging
    # =========================================================================

    def _log_event(
        self,
        event_type: SchedulerEventType,
        request_id: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a scheduler event."""
        if not self.enabled:
            return

        timestamp_ns = self._get_relative_time_ns()

        event = SchedulerEvent(
            timestamp_ns=timestamp_ns,
            event_type=event_type,
            request_id=request_id,
            details=details or {},
        )

        with self._lock:
            self.event_buffer.append(event)

    def log_kv_cache_allocated(
        self,
        request_id: str,
        num_blocks: int,
    ) -> None:
        """Log KV cache allocation."""
        self._log_event(
            SchedulerEventType.KV_CACHE_ALLOCATED,
            request_id,
            {"num_blocks": num_blocks}
        )

    def log_kv_cache_freed(
        self,
        request_id: str,
        num_blocks: int,
    ) -> None:
        """Log KV cache freed."""
        self._log_event(
            SchedulerEventType.KV_CACHE_FREED,
            request_id,
            {"num_blocks": num_blocks}
        )

    # =========================================================================
    # Buffer management and file writing
    # =========================================================================

    def _maybe_flush(self) -> None:
        """Flush buffers if they exceed the threshold."""
        total_buffer_size = (
            len(self.request_buffer) +
            len(self.iteration_buffer) +
            len(self.event_buffer)
        )

        if total_buffer_size >= self.buffer_size:
            self.flush_all()

    def flush_all(self) -> None:
        """Flush all buffers to disk."""
        if not self.enabled:
            return

        with self._lock:
            self._flush_request_buffer()
            self._flush_iteration_buffer()
            self._flush_per_iteration_request_buffer()
            self._flush_event_buffer()

    def _flush_request_buffer(self) -> None:
        """Flush request metrics buffer to CSV."""
        if not self.request_buffer:
            return

        filepath = self.log_dir / "per_request.csv"
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            while self.request_buffer:
                metrics = self.request_buffer.popleft()
                row = [
                    metrics.request_id,
                    metrics.arrival_time_ns,
                    metrics.first_scheduled_time_ns or "",
                    metrics.completion_time_ns or "",
                    metrics.total_waiting_time_ns,
                    metrics.total_execution_time_ns,
                    metrics.calculate_e2e_latency_ns() or "",
                    metrics.input_token_count,
                    metrics.output_token_count,
                    metrics.ttft_ns or "",
                    metrics.calculate_tpot_ns() or "",
                    metrics.calculate_throughput() or "",
                    metrics.preemption_count,
                    metrics.eviction_count,
                    metrics.kv_cache_blocks,
                ]
                writer.writerow(row)

    def _flush_iteration_buffer(self) -> None:
        """Flush iteration metrics buffer to CSV."""
        if not self.iteration_buffer:
            return

        filepath = self.log_dir / "per_iteration.csv"
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            while self.iteration_buffer:
                metrics = self.iteration_buffer.popleft()
                actions = ";".join(metrics.scheduler_actions) if \
                    metrics.scheduler_actions else ""
                row = [
                    metrics.iteration_id,
                    metrics.start_time_ns,
                    metrics.end_time_ns or "",
                    metrics.duration_ns or "",
                    metrics.scheduling_overhead_ns or "",
                    metrics.forward_pass_duration_ns or "",
                    metrics.batch_size,
                    metrics.total_tokens_in_batch,
                    metrics.prefill_tokens,
                    metrics.decode_tokens,
                    metrics.kv_cache_used_blocks,
                    metrics.kv_cache_total_blocks,
                    len(metrics.running_request_ids),
                    len(metrics.waiting_request_ids),
                    len(metrics.preempted_request_ids),
                    actions,
                ]
                writer.writerow(row)

    def _flush_per_iteration_request_buffer(self) -> None:
        """Flush per-iteration request state buffer to CSV."""
        if not self.per_iteration_request_buffer:
            return

        filepath = self.log_dir / "per_iteration_requests.csv"
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            while self.per_iteration_request_buffer:
                state = self.per_iteration_request_buffer.popleft()
                row = [
                    state.iteration_id,
                    state.request_id,
                    state.state,
                    state.queue_position,
                    state.waiting_reason,
                    state.tokens_generated,
                    state.kv_cache_blocks,
                    state.time_in_state_ns,
                ]
                writer.writerow(row)

    def _flush_event_buffer(self) -> None:
        """Flush event buffer to CSV."""
        if not self.event_buffer:
            return

        filepath = self.log_dir / "events.csv"
        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            while self.event_buffer:
                event = self.event_buffer.popleft()
                row = [
                    event.timestamp_ns,
                    event.event_type.value,
                    event.request_id,
                    json.dumps(event.details),
                ]
                writer.writerow(row)

    def shutdown(self) -> None:
        """Shutdown the logger and flush all remaining data."""
        if not self.enabled:
            return

        self.flush_all()
        logger.info("SchedulerLogger shutdown complete. Logs saved to: %s",
                    self.log_dir)


def is_scheduler_logging_enabled() -> bool:
    """Check if scheduler logging is enabled via environment variable."""
    return os.environ.get("VLLM_SCHEDULER_LOGGING", "0") == "1"


def create_scheduler_logger() -> Optional[SchedulerLogger]:
    """Create a scheduler logger if logging is enabled."""
    if not is_scheduler_logging_enabled():
        return None

    log_dir = os.environ.get("VLLM_SCHEDULER_LOG_DIR", "./scheduler_logs")
    return SchedulerLogger(log_dir=log_dir, enabled=True)
