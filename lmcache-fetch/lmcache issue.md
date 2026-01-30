# vLLM Server Crash with LMCache `use_layerwise: true`

## Summary
vLLM server crashes with `StopIteration` error when running inference workloads with LMCache's `use_layerwise` option enabled. The issue does not occur when `use_layerwise: false`.

## Environment
- **Model**: DeepSeek-R1-70B (80 layers)
- **vLLM Version**: v1 (latest)
- **LMCache Version**: latest
- **Python Version**: 3.12
- **Configuration**: `use_layerwise: true` in `lmcache_config.yaml`

## Error Stack Trace
```
ERROR 01-30 15:56:28 [multiproc_executor.py:824] Traceback (most recent call last):
  File "/vllm/vllm/v1/worker/kv_connector_model_runner_mixin.py", line 133, in _get_kv_connector_output
    kv_connector.wait_for_save()
  File "/vllm/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py", line 171, in wait_for_save
    self._lmcache_engine.wait_for_save()
  File "/lmcache/lmcache/integration/vllm/vllm_v1_adapter.py", line 1347, in wait_for_save
    next(layerwise_storer)
StopIteration

The above exception was the direct cause of the following exception:

RuntimeError: generator raised StopIteration
```

## Root Cause Analysis

### Issue Location
The error originates from `lmcache/integration/vllm/vllm_v1_adapter.py` in the `wait_for_save()` method when calling `next(layerwise_storer)`.

### Generator Lifecycle Mismatch
The `store_layer()` generator in `lmcache/v1/cache_engine.py`:
- Yields exactly **`num_layers + 1`** times (81 times for 80-layer model)
- Layers 0-79: yielded during layer-wise storage
- Final yield: after completing storage (line 621 in `cache_engine.py`)

The `vllm_v1_adapter.py` calls `next()`:
- 80 times during forward pass (layers 0-79)
- 1 time in `wait_for_save()` for final cleanup
- **Total: 81 times** ✅

### The Problem: State Not Reset Between Batches

**In `vllm_v1_adapter.py`:**

1. **`layerwise_storers` list initialization** (line ~1245):
```python
if self.current_layer == 0:
    self.layerwise_storers = []
    # Create new generators...
```

2. **Layer-wise `next()` calls** (line ~1306):
```python
# Called 80 times during forward pass (layers 0-79)
for layerwise_storer in self.layerwise_storers:
    next(layerwise_storer)
self.current_layer += 1
```

3. **Final `next()` in `wait_for_save()`** (line ~1347):
```python
if self.use_layerwise:
    for layerwise_storer in self.layerwise_storers:
        next(layerwise_storer)  # 81st call - exhausts generators
    # Missing: self.current_layer = 0
    # Missing: self.layerwise_storers = []
```

### The Bug
After processing the first batch:
- `current_layer` increments from 0 → 80 during forward pass
- `wait_for_save()` is called with `current_layer = 80`
- **Missing reset**: `current_layer` stays at 80
- **Missing cleanup**: `layerwise_storers` list retains exhausted generators

When processing the second batch:
- `if self.current_layer == 0:` check fails (current_layer = 80)
- `layerwise_storers` list is NOT re-initialized
- The list still contains **exhausted generators from the first batch**
- Calling `next()` on exhausted generators → `StopIteration`

## Evidence from Debug Logs

### Observation 1: `num_storers: 0` in Early Layers
```json
{"location": "vllm_v1_adapter.py:1245", "message": "Layer 0 - initializing layerwise_storers", "data": {"num_requests": 1}}
{"location": "vllm_v1_adapter.py:1302", "message": "Layer 0 - created storers", "data": {"num_storers": 0}}
```
This suggests the `if self.current_layer == 0:` condition is not being met for subsequent batches because `current_layer` was never reset to 0.

### Observation 2: `current_layer` Progression
```
Max current_layer before error: 79
```
The layer counter progresses correctly up to layer 79, but the problem occurs when `wait_for_save()` attempts the 81st `next()` call on generators that were already exhausted in a previous batch.

### Observation 3: Multiple `wait_for_save()` Calls
```json
{"location": "vllm_v1_adapter.py:1323", "message": "wait_for_save - calling final next() on all storers", "data": {"num_storers": 0}}
{"location": "vllm_v1_adapter.py:1323", "message": "wait_for_save - calling final next() on all storers", "data": {"num_storers": 10}}
```
Some `wait_for_save()` calls have `num_storers: 0` (likely from batches where storers weren't re-initialized), while others have `num_storers: 10` (successful batches).

## Hypothesis

**Primary Hypothesis**: Missing state cleanup in `wait_for_save()` causes stale generator references to persist across batches.

The `wait_for_save()` method should:
1. Reset `self.current_layer = 0` to allow re-initialization for the next batch
2. Clear `self.layerwise_storers = []` to remove exhausted generator references

### Expected Behavior vs Actual Behavior

**Expected (per batch):**
```
Batch 1: Initialize storers → 80 layer calls → 1 final call → Reset → Clear
Batch 2: Initialize storers → 80 layer calls → 1 final call → Reset → Clear
...
```

**Actual (buggy):**
```
Batch 1: Initialize storers → 80 layer calls → 1 final call → [No reset/clear]
Batch 2: [Skip initialization - current_layer != 0] → Use stale storers → StopIteration
```

## Attempted Fix (Still Not Working)

Added state cleanup in `wait_for_save()` after the final `next()` calls:

```python
if self.use_layerwise:
    for layerwise_storer in self.layerwise_storers:
        next(layerwise_storer)
    
    # Reset state for next batch
    self.current_layer = 0
    self.layerwise_storers = []
```

**Result**: The error still persists, suggesting there might be additional issues:

### Additional Possible Issues

1. **Concurrency/Threading**: 
   - vLLM uses multi-process/multi-threaded workers (TP0, TP1, TP2, TP3 in logs)
   - `current_layer` and `layerwise_storers` might not be thread-safe
   - Race conditions could occur if multiple workers share state

2. **Generator Reuse Across Workers**:
   - Each worker might have its own `current_layer` counter
   - Generators might be shared across workers incorrectly
   - Worker-specific state management might be needed

3. **Early Generator Exhaustion**:
   - Some code path might be calling `next()` more than 81 times
   - Generator might be exhausting earlier than expected in certain conditions
   - Need to verify all code paths that call `next()` on `layerwise_storers`

4. **Incorrect Initialization Condition**:
   - The `if self.current_layer == 0:` check might not be the correct condition
   - Should it check something else (e.g., new request batch, scheduler state)?

## Minimal Reproduction

1. Configure LMCache with `use_layerwise: true`
2. Run vLLM server with a multi-layer model (e.g., DeepSeek-R1-70B)
3. Send multiple inference requests (batch size > 1)
4. Observe `StopIteration` error after first batch completes

## Expected Behavior
The server should handle multiple batches without crashing when `use_layerwise: true`.

## Additional Notes

- **Works fine with `use_layerwise: false`**: This confirms the issue is specific to layer-wise storage mode
- **Error timing**: Occurs consistently when processing subsequent batches, not on the first batch
- **Multiple workers affected**: Errors appear across multiple worker processes (TP0, TP1, TP2, TP3)

## Questions for Maintainers

1. Is `current_layer` and `layerwise_storers` expected to be worker-specific or shared across workers?
2. Should the state reset logic be in `wait_for_save()` or elsewhere (e.g., before starting a new batch)?
3. Are there any threading/multiprocessing concerns with the current implementation?
4. Is the `if self.current_layer == 0:` condition the correct way to detect a new batch?
