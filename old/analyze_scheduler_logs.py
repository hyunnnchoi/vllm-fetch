#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# NOTE, hyunnnchoi, 2026.01.08
# 스케줄러 로그 분석 예시 스크립트

"""
스케줄러 로그를 분석하고 주요 메트릭을 출력하는 스크립트

Usage:
    python analyze_scheduler_logs.py ./scheduler_logs
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def analyze_per_request(log_dir: Path) -> None:
    """Request별 요약 통계 분석"""
    csv_path = log_dir / "per_request.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return

    print("\n" + "=" * 80)
    print("REQUEST-LEVEL METRICS")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    if df.empty:
        print("No request data found")
        return

    # TTFT를 초로 변환 (nanoseconds -> seconds)
    df['ttft_sec'] = df['ttft'] / 1e9
    df['tpot_sec'] = df['tpot'] / 1e9
    df['e2e_latency_sec'] = df['e2e_latency'] / 1e9

    print(f"\nTotal requests: {len(df)}")
    print(f"\nTTFT (Time To First Token):")
    print(f"  Mean: {df['ttft_sec'].mean():.4f} sec")
    print(f"  Median: {df['ttft_sec'].median():.4f} sec")
    print(f"  P50: {df['ttft_sec'].quantile(0.50):.4f} sec")
    print(f"  P90: {df['ttft_sec'].quantile(0.90):.4f} sec")
    print(f"  P99: {df['ttft_sec'].quantile(0.99):.4f} sec")

    print(f"\nTPOT (Time Per Output Token):")
    print(f"  Mean: {df['tpot_sec'].mean():.4f} sec")
    print(f"  Median: {df['tpot_sec'].median():.4f} sec")

    print(f"\nE2E Latency:")
    print(f"  Mean: {df['e2e_latency_sec'].mean():.4f} sec")
    print(f"  Median: {df['e2e_latency_sec'].median():.4f} sec")

    print(f"\nThroughput:")
    print(f"  Mean: {df['throughput'].mean():.2f} tokens/sec")
    print(f"  Median: {df['throughput'].median():.2f} tokens/sec")

    print(f"\nPreemption:")
    preemption_rate = (df['preemption_count'] > 0).mean() * 100
    print(f"  Preemption rate: {preemption_rate:.2f}%")
    print(f"  Average preemptions per request: {df['preemption_count'].mean():.2f}")

    print(f"\nTokens:")
    print(f"  Average input tokens: {df['input_tokens'].mean():.2f}")
    print(f"  Average output tokens: {df['output_tokens'].mean():.2f}")

    # 새로운 필드들
    if 'kv_cache_memory_bytes' in df.columns:
        print(f"\nKV Cache Memory:")
        print(f"  Average: {df['kv_cache_memory_bytes'].mean() / 1024 / 1024:.2f} MB")
        print(f"  Max: {df['kv_cache_memory_bytes'].max() / 1024 / 1024:.2f} MB")

    if 'priority_score' in df.columns:
        print(f"\nPriority:")
        print(f"  Average priority score: {df['priority_score'].mean():.2f}")


def analyze_per_iteration(log_dir: Path) -> None:
    """Iteration별 시스템 상태 분석"""
    csv_path = log_dir / "per_iteration.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return

    print("\n" + "=" * 80)
    print("ITERATION-LEVEL METRICS")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    if df.empty:
        print("No iteration data found")
        return

    # 시간을 밀리초로 변환
    df['duration_ms'] = df['duration'] / 1e6

    print(f"\nTotal iterations: {len(df)}")
    print(f"\nIteration duration:")
    print(f"  Mean: {df['duration_ms'].mean():.2f} ms")
    print(f"  Median: {df['duration_ms'].median():.2f} ms")
    print(f"  P90: {df['duration_ms'].quantile(0.90):.2f} ms")
    print(f"  P99: {df['duration_ms'].quantile(0.99):.2f} ms")

    print(f"\nBatch size:")
    print(f"  Mean: {df['batch_size'].mean():.2f}")
    print(f"  Median: {df['batch_size'].median():.2f}")
    print(f"  Max: {df['batch_size'].max():.0f}")

    print(f"\nTokens per iteration:")
    print(f"  Mean: {df['total_tokens'].mean():.2f}")
    print(f"  Median: {df['total_tokens'].median():.2f}")
    print(f"  Max: {df['total_tokens'].max():.0f}")

    # KV cache 사용률 계산
    df['kv_cache_util'] = df['kv_cache_used'] / df['kv_cache_total'] * 100
    print(f"\nKV Cache utilization:")
    print(f"  Mean: {df['kv_cache_util'].mean():.2f}%")
    print(f"  Median: {df['kv_cache_util'].median():.2f}%")
    print(f"  Max: {df['kv_cache_util'].max():.2f}%")

    print(f"\nQueue lengths:")
    print(f"  Mean running: {df['running_count'].mean():.2f}")
    print(f"  Mean waiting: {df['waiting_count'].mean():.2f}")
    print(f"  Max waiting: {df['waiting_count'].max():.0f}")


def analyze_events(log_dir: Path) -> None:
    """이벤트 로그 분석"""
    csv_path = log_dir / "events.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return

    print("\n" + "=" * 80)
    print("EVENT STATISTICS")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    if df.empty:
        print("No event data found")
        return

    print(f"\nTotal events: {len(df)}")
    print(f"\nEvent type distribution:")
    event_counts = df['event_type'].value_counts()
    for event_type, count in event_counts.items():
        print(f"  {event_type}: {count}")

    # Preemption 이벤트 분석
    preemption_events = df[df['event_type'] == 'request_preempted']
    if not preemption_events.empty:
        print(f"\nTotal preemptions: {len(preemption_events)}")


def analyze_request_states(log_dir: Path) -> None:
    """Request 상태 분석"""
    csv_path = log_dir / "per_iteration_requests.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return

    print("\n" + "=" * 80)
    print("REQUEST STATE ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    if df.empty:
        print("No request state data found")
        return

    print(f"\nTotal request state records: {len(df)}")
    print(f"\nState distribution:")
    state_counts = df['state'].value_counts()
    for state, count in state_counts.items():
        print(f"  {state}: {count}")

    # Waiting 원인 분석
    waiting_df = df[df['state'] == 'waiting']
    if not waiting_df.empty and 'waiting_reason' in waiting_df.columns:
        print(f"\nWaiting reasons:")
        reason_counts = waiting_df['waiting_reason'].value_counts()
        for reason, count in reason_counts.items():
            if reason and reason != 'nan':
                print(f"  {reason}: {count}")


def analyze_state_transitions(log_dir: Path) -> None:
    """State transitions 분석"""
    csv_path = log_dir / "state_transitions.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found")
        return

    print("\n" + "=" * 80)
    print("STATE TRANSITIONS")
    print("=" * 80)

    df = pd.read_csv(csv_path)

    if df.empty:
        print("No state transition data found")
        return

    print(f"\nTotal state transitions: {len(df)}")
    print(f"\nState distribution:")
    state_counts = df['state'].value_counts()
    for state, count in state_counts.items():
        print(f"  {state}: {count}")

    # 평균 state duration
    print(f"\nAverage duration in each state:")
    df['duration_sec'] = df['duration_in_prev_state'] / 1e9
    avg_durations = df.groupby('state')['duration_sec'].mean()
    for state, duration in avg_durations.items():
        print(f"  {state}: {duration:.4f} sec")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze vLLM scheduler logs"
    )
    parser.add_argument(
        "log_dir",
        type=str,
        help="Path to the scheduler logs directory",
    )

    args = parser.parse_args()
    log_dir = Path(args.log_dir)

    if not log_dir.exists():
        print(f"Error: Log directory '{log_dir}' does not exist")
        sys.exit(1)

    if not log_dir.is_dir():
        print(f"Error: '{log_dir}' is not a directory")
        sys.exit(1)

    print(f"Analyzing scheduler logs from: {log_dir}")

    # 각 CSV 파일 분석
    analyze_per_request(log_dir)
    analyze_per_iteration(log_dir)
    analyze_events(log_dir)
    analyze_request_states(log_dir)
    analyze_state_transitions(log_dir)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
