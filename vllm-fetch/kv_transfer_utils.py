# /vllm/vllm/attention/utils/kv_transfer_utils.py
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import csv
import inspect
import os
import time

# NOTE, hyunnnchoi, 2026.02.03
# 다중 프로세스 CSV 동시 쓰기 충돌 방지
# NOTE, hyunnnchoi, 2026.02.03
# 프로세스별 CSV 분리로 락 제거
from collections.abc import Callable
from functools import wraps
from pathlib import Path

from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.logger import init_logger

# NOTE, hyunnnchoi, 2026.01.29
# KV transfer layer-wise 대기 시간 측정을 위한 logger 추가
logger = init_logger(__name__)

# NOTE, hyunnnchoi, 2026.01.29
# KV transfer 타이밍을 CSV로 저장하기 위한 파일 핸들
_kv_transfer_csv_file = None
_kv_transfer_csv_writer = None
_kv_transfer_log_enabled = False
# NOTE, hyunnnchoi, 2026.02.03
# 로깅 초기화 시도 여부 플래그 추가
_kv_transfer_log_attempted = False
_kv_transfer_iteration_counter = 0
_kv_transfer_last_flush_time = 0.0


def _init_kv_transfer_csv():
    """KV transfer 타이밍 CSV 파일 초기화"""
    global \
        _kv_transfer_csv_file, \
        _kv_transfer_csv_writer, \
        _kv_transfer_log_enabled, \
        _kv_transfer_log_attempted, \
        _kv_transfer_last_flush_time
    _kv_transfer_log_attempted = True

    # 환경변수로 로깅 활성화 여부 확인 (두 변수 모두 지원)
    log_dir = os.environ.get("VLLM_SCHED_LOG_DIR") or os.environ.get(
        "VLLM_SCHEDULER_LOG_DIR"
    )
    if not log_dir:
        _kv_transfer_log_enabled = False
        return

    _kv_transfer_log_enabled = True
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # NOTE, hyunnnchoi, 2026.02.03
    # 프로세스별 CSV 파일 생성 및 헤더 작성
    pid = os.getpid()
    csv_path = log_path / f"kv_transfer_timing_pid_{pid}.csv"
    _kv_transfer_csv_file = open(csv_path, "w", newline="", buffering=8192)
    _kv_transfer_csv_writer = csv.writer(_kv_transfer_csv_file)
    _kv_transfer_csv_writer.writerow(
        [
            "timestamp_ns",
            "layer_name",
            "wait_time_us",
            "attn_time_us",
            "save_time_us",
            "total_time_us",
        ]
    )

    # NOTE, hyunnnchoi, 2026.02.03
    # Flush 간격 설정값 읽기 (샘플링/임계값은 비활성화됨)
    flush_interval_sec = float(
        os.environ.get("VLLM_KV_TRANSFER_LOG_FLUSH_INTERVAL", "5.0")
    )

    # flush 시간 초기화
    _kv_transfer_last_flush_time = time.time()

    logger.info(
        f"KV transfer timing CSV initialized at {csv_path} "
        f"(sampling=disabled, threshold=disabled, "
        f"flush_interval={flush_interval_sec}s)"
    )


def _log_kv_transfer_timing(
    layer_name: str, wait_time: float, attn_time: float, save_time: float
):
    """KV transfer 타이밍을 CSV에 기록 (모든 레이어 기록)"""
    global \
        _kv_transfer_csv_writer, \
        _kv_transfer_csv_file, \
        _kv_transfer_iteration_counter, \
        _kv_transfer_log_attempted, \
        _kv_transfer_last_flush_time

    # NOTE, hyunnnchoi, 2026.02.03
    # 환경변수 기반으로 첫 호출 시에만 CSV 초기화
    if _kv_transfer_csv_writer is None and not _kv_transfer_log_attempted:
        _init_kv_transfer_csv()

    if (not _kv_transfer_log_enabled) or (_kv_transfer_csv_writer is None):
        return

    wait_us = wait_time * 1e6
    attn_us = attn_time * 1e6
    save_us = save_time * 1e6
    total_us = (wait_time + attn_time + save_time) * 1e6

    _kv_transfer_iteration_counter += 1

    # NOTE, hyunnnchoi, 2026.02.03
    # 모든 레이어 기록 (필터링 비활성화)
    should_log = True

    if not should_log:
        return

    timestamp_ns = time.time_ns()
    _kv_transfer_csv_writer.writerow(
        [
            timestamp_ns,
            layer_name,
            f"{wait_us:.1f}",
            f"{attn_us:.1f}",
            f"{save_us:.1f}",
            f"{total_us:.1f}",
        ]
    )

    # NOTE, hyunnnchoi, 2026.01.29
    # Flush 조건: 100회마다 OR 5초마다
    flush_interval_sec = float(
        os.environ.get("VLLM_KV_TRANSFER_LOG_FLUSH_INTERVAL", "5.0")
    )
    current_time = time.time()

    should_flush = (_kv_transfer_iteration_counter % 100 == 0) or (
        (current_time - _kv_transfer_last_flush_time) >= flush_interval_sec
    )

    if should_flush:
        _kv_transfer_csv_file.flush()
        _kv_transfer_last_flush_time = current_time


def maybe_transfer_kv_layer(func: Callable) -> Callable:
    """Decorator that handles KV layer transfer prior and after execution of
    an attention layer, if enabled. Otherwise, the wrapper is a no-op.

    On entry: waits for the KV layer from the connector.
    On exit: saves the KV layer to the connector.
    """
    # Import at runtime to avoid circular dependency
    from vllm.attention.layer import get_attention_context

    # Inspect the signature ONCE when the decorator is applied.
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())

    # Find the index of 'layer_name' parameter.
    try:
        layer_name_index = param_names.index("layer_name")
    except ValueError as e:
        raise TypeError(
            f"Function {func.__name__} must have a 'layer_name' parameter"
        ) from e

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return func(*args, **kwargs)

        layer_name: str = args[layer_name_index]

        # Extract attention context (layer-specific metadata, layer, and kv_cache)
        attn_metadata, attn_layer, kv_cache = get_attention_context(layer_name)
        connector = get_kv_transfer_group()
        if attn_metadata is None or not connector.has_connector_metadata():
            return func(*args, **kwargs)

        # NOTE, hyunnnchoi, 2026.01.29
        # KV layer load 대기 시간 측정 시작
        wait_start = time.perf_counter()

        # Wait for KV layer on entry
        connector.wait_for_layer_load(layer_name)

        wait_time = time.perf_counter() - wait_start

        # Execute the function
        attn_start = time.perf_counter()
        result = func(*args, **kwargs)
        attn_time = time.perf_counter() - attn_start

        # Save KV cache layer on exit
        save_start = time.perf_counter()
        connector.save_kv_layer(layer_name, kv_cache, attn_metadata)
        save_time = time.perf_counter() - save_start

        # NOTE, hyunnnchoi, 2026.01.29
        # CSV에 타이밍 기록 (로깅 오버헤드 최소화)
        _log_kv_transfer_timing(layer_name, wait_time, attn_time, save_time)

        return result

    return wrapper
