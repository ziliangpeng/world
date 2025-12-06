# vLLM API Layer - Request Intake & Output

## Overview

The API Layer (Request Intake & Output Layer) sits between external users and the Engine Core. It handles converting user requests into tokenized inputs, and converting engine outputs back into user-friendly responses.

**Key Responsibility**: Transform text ↔ tokens, handle streaming, manage async coordination

---

## Architecture Diagram

![vLLM API Layer Architecture](intake-and-output.png)

---

## Layer Positioning

```
User/Client (HTTP, Python)
    ↓
┌─────────────────────────────────────┐
│   API Layer (This Document)         │
│   - Request Intake                  │
│   - Background Step Loop            │
│   - Output Processing               │
└─────────────────────────────────────┘
    ↓
Engine Core Layer
    - EngineCore
    - Scheduler
    - Workers
    - BlockManager
```

---

## Main Components

### Request Intake Components

**1. API Server** (`vllm/entrypoints/openai/`)

- **Purpose**: HTTP endpoints (OpenAI-compatible API)
- **Endpoints**: `/v1/completions`, `/v1/chat/completions`
- **Receives**: JSON with prompt, sampling params
- **Returns**: Streaming or non-streaming responses

**2. AsyncLLM** (`vllm/v1/engine/async_llm.py`)

- **Purpose**: Async wrapper with background step loop
- **Key Feature**: Runs inference loop in background task
- **Use Case**: API servers, concurrent requests
- **Key Methods**:
  - `add_request()` - Queue new request (non-blocking)
  - `generate()` - High-level async generation
  - `abort()` - Cancel request

**3. LLMEngine** (`vllm/v1/engine/llm_engine.py`)

- **Purpose**: Synchronous wrapper for manual control
- **Key Feature**: Manual `step()` calls
- **Use Case**: Batch processing, scripts
- **Key Methods**:
  - `add_request()` - Add request
  - `step()` - Execute one inference iteration (manual)
  - `abort_request()` - Cancel request

**4. Processor** (`vllm/v1/engine/processor.py`)

- **Purpose**: Tokenize and validate inputs
- **Key Method**: `process_inputs(prompt, params) → EngineCoreRequest`
- **Responsibilities**:
  - Tokenize text prompts
  - Handle multimodal inputs (images)
  - Validate parameters
  - Create EngineCoreRequest objects

**5. EngineCoreClient** (`vllm/v1/engine/core_client.py`)

- **Purpose**: Abstraction over in-process vs multi-process
- **Implementations**:
  - `InprocClient` - Same process (direct calls)
  - `MPClient` - Multi-process via ZMQ sockets
- **Why**: Enables flexible deployment (single process, distributed)

### Output Processing Components

**6. OutputProcessor** (`vllm/v1/engine/output_processor.py`)

- **Purpose**: Convert engine outputs to user-facing results
- **Key Method**: `process_outputs(EngineCoreOutputs) → OutputProcessorOutput`
- **Responsibilities**:
  - Detokenize tokens → text
  - Compute log probabilities
  - Check stop strings
  - Handle streaming
  - Track request state

**7. RequestState** (`vllm/v1/engine/output_processor.py`)

- **Purpose**: Track state of a single request
- **Key Fields**:
  - `request_id` - Unique identifier
  - `prompt_token_ids` - Original prompt tokens
  - `detokenizer` - Incremental detokenizer instance
  - `logprobs_processor` - Log probability processor
  - `queue` - Output queue (for AsyncLLM)
  - `sent_tokens_offset` - Streaming position
  - `stream_interval` - Send output every N tokens

**8. RequestOutputCollector** (`vllm/v1/engine/output_processor.py`)

- **Purpose**: Per-request async queue for streaming
- **Key Methods**:
  - `put(output)` - Producer adds output (non-blocking)
  - `async get()` - Consumer retrieves output (async)
- **Pattern**: Async producer-consumer queue
- **Usage**: One queue per request in AsyncLLM

**9. Detokenizer** (`vllm/v1/engine/detokenizer.py`)

- **Purpose**: Convert token IDs → text incrementally
- **Implementations**:
  - `FastIncrementalDetokenizer` - Fast path
  - `SlowIncrementalDetokenizer` - Fallback
- **Key Feature**: Detects stop strings during decoding

---

## Data Structures

### Input: EngineCoreRequest

**Location**: `vllm/v1/engine/__init__.py`

**Fields**:

- `request_id: str` - Unique identifier
- `prompt_token_ids: list[int]` - Tokenized prompt (main payload)
- `mm_features: list[MultiModalFeatureSpec]` - Multimodal data (images, audio)
- `sampling_params: SamplingParams` - Temperature, top_p, max_tokens, etc.
- `pooling_params: PoolingParams` - For embedding models
- `eos_token_id: int` - End-of-sequence token
- `arrival_time: float` - Timestamp
- `lora_request: LoRARequest` - LoRA adapter (optional)
- `priority: int` - Request priority

### Output: EngineCoreOutput

**Location**: `vllm/v1/engine/__init__.py`

**Fields**:

- `request_id: str` - Which request this output belongs to
- `new_token_ids: list[int]` - Newly generated tokens
- `new_logprobs: LogprobsLists` - Token probabilities
- `finish_reason: FinishReason` - Why generation stopped (stop, length, abort)
- `stop_reason: int | str` - Specific stop string if applicable
- `num_cached_tokens: int` - Prefix cache hits

### User-Facing: RequestOutput

**Location**: `vllm/outputs.py`

**Fields**:

- `request_id: str` - Unique identifier
- `prompt: str` - Original prompt text
- `prompt_token_ids: list[int]` - Tokenized prompt
- `outputs: list[CompletionOutput]` - Generated outputs (can be multiple for n>1)
- `finished: bool` - Is generation complete?
- `metrics: RequestMetrics` - Performance stats

**CompletionOutput**:

- `index: int` - Output index (for n>1)
- `text: str` - Generated text
- `token_ids: list[int]` - Generated token IDs
- `cumulative_logprob: float` - Total log probability
- `logprobs: list[dict]` - Per-token log probabilities
- `finish_reason: str` - Why stopped ("stop", "length", etc.)
- `stop_reason: str` - Specific stop string

---

## The Request Flow

### 1. Request Intake (Synchronous Path)

```
HTTP Request
  ↓
API Server (parse JSON)
  ↓
AsyncLLM.add_request(request_id, prompt, params)
  ↓
Processor.process_inputs(prompt, params)
  ├─ Tokenize: "Hello world" → [1, 2, 3, 4]
  ├─ Validate params
  └─ Create EngineCoreRequest
  ↓
OutputProcessor.add_request(request, queue)
  └─ Create RequestState for tracking
  ↓
EngineCoreClient.add_request_async(request)
  └─ Send to EngineCore (scheduler queue)
  ↓
Returns RequestOutputCollector (queue)
```

**Key Point**: `add_request()` returns **immediately** with a queue. It doesn't wait for inference!

---

### 2. Background Step Loop (Async Producer)

**Location**: `AsyncLLM._run_output_handler()` (line 489)

```python
async def output_handler():
    while True:  # Infinite loop!
        # 1. Get outputs from EngineCore (calls step() internally)
        outputs = await engine_core.get_output_async()

        # 2. Process outputs (detokenize, sample, check stops)
        processed = output_processor.process_outputs(outputs.outputs)

        # 3. Abort finished requests
        await engine_core.abort_requests_async(processed.reqs_to_abort)

        # 4. Log stats
        logger_manager.record(...)
```

**Key Points**:

- Runs in **background asyncio task**
- Started automatically when AsyncLLM is created
- **Always running** - doesn't wait for requests
- Processes ALL requests together (continuous batching)
- `engine_core.get_output_async()` internally calls `EngineCore.step()`

---

### 3. Output Processing (In the Loop)

**`OutputProcessor.process_outputs()`** (lines 441-546)

For each `EngineCoreOutput`:

```python
# Step 1: Update stats
self._update_stats_from_output(req_state, engine_core_output)

# Step 2: Detokenize tokens → text
if pooling_output is None:
    stop_string = req_state.detokenizer.update(
        new_token_ids,
        finish_reason == FinishReason.STOP
    )
    if stop_string:
        finish_reason = FinishReason.STOP  # Detected stop string!

    # Step 3: Compute logprobs if requested
    req_state.logprobs_processor.update_from_output(engine_core_output)

# Step 4: Create RequestOutput
if request_output := req_state.make_request_output(...):
    if req_state.queue is not None:
        # AsyncLLM: push to queue
        req_state.queue.put(request_output)
    else:
        # LLMEngine: add to return list
        request_outputs.append(request_output)

# Step 5: Cleanup finished requests
if finish_reason is not None:
    self.request_states.pop(req_id)
    if not engine_core_output.finished:
        reqs_to_abort.append(req_id)  # Tell EngineCore
```

---

### 4. Output Queuing & Streaming

**RequestOutputCollector** - One queue per request

```python
class RequestOutputCollector:
    def __init__(self):
        self.output = None
        self.ready = asyncio.Event()  # Async coordination

    def put(self, output):  # Producer (background loop)
        self.output = output
        self.ready.set()  # Signal output ready

    async def get(self):  # Consumer (API handler)
        while self.output is None:
            await self.ready.wait()  # Block until output ready
        output = self.output
        self.output = None
        self.ready.clear()
        return output
```

**Usage in API Handler**:

```python
# add_request returns queue immediately
queue = await engine.add_request("req1", "Hello", params)

# Stream outputs as they arrive
async for output in queue:
    yield f"data: {json.dumps(output)}\n\n"  # SSE format
    if output.finished:
        break
```

---

## Key Patterns

### Pattern 1: Decoupled Producer-Consumer

**Producer**: Background loop continuously calls `step()`, generates tokens

**Consumer**: API handlers pull from their queues

**Benefit**: They don't block each other - continuous batching works!

```
Request 1 arrives → add_request() → returns queue1 → handler streams from queue1
Request 2 arrives → add_request() → returns queue2 → handler streams from queue2
Request 3 arrives → add_request() → returns queue3 → handler streams from queue3

Background loop:
  step() → processes all 3 together → outputs pushed to all 3 queues
```

---

### Pattern 2: Continuous Batching

**Traditional (batching per request)**:

```
Request 1 arrives → process alone → return
Request 2 arrives → process alone → return
```

**Continuous Batching (vLLM)**:

```
Requests 1, 2, 3 arrive at different times
Background loop batches them together:
  step() → [output1, output2, output3]
Each pushed to respective queue
```

**Benefit**: Maximize GPU utilization, higher throughput

---

### Pattern 3: Streaming via Intervals

**Stream Interval**: Send output every N tokens (reduce HTTP overhead)

```python
# stream_interval = 5
if stream_interval > 1:
    # Send only when:
    # 1. Finished
    # 2. First token
    # 3. Reached interval
    if not (finished or
            sent_offset == 0 or
            len(tokens) - sent_offset >= stream_interval):
        return None  # Don't send yet
```

**Example** (`stream_interval=5`):

- Token 1: Send (first)
- Tokens 2-4: Buffer
- Token 5: Send (interval)
- Tokens 6-9: Buffer
- Token 10: Send
- Final: Send (finished)

---

## LLMEngine vs AsyncLLM

### LLMEngine (Synchronous)

**Control**: You manually call `step()`

```python
engine = LLMEngine(...)

engine.add_request("req1", "Hello", params)
engine.add_request("req2", "Hi", params)

# YOU control the loop
while engine.has_unfinished_requests():
    outputs = engine.step()  # BLOCKING
    for output in outputs:
        print(output.text)
```

**Use Cases**:

- Offline batch processing
- Scripts where you want full control
- Debugging

---

### AsyncLLM (Asynchronous)

**Control**: Background task calls `step()` automatically

```python
engine = AsyncLLM(...)

# Returns queue immediately
queue = await engine.add_request("req1", "Hello", params)

# Background loop handles step() for you
async for output in queue:
    print(output.text)
```

**Use Cases**:

- API servers (FastAPI, etc.)
- Concurrent request handling
- Streaming responses
- Long-running services

---

## The Complete Flow Example

### User sends request:

```
POST /v1/completions
{
  "prompt": "Hello world",
  "temperature": 0.7,
  "max_tokens": 100
}
```

### API Layer Processing:

```
1. API Server receives HTTP request
   ↓
2. AsyncLLM.add_request()
   - Calls Processor.process_inputs()
     - Tokenizes: "Hello world" → [1, 2, 3, 4]
     - Creates EngineCoreRequest
   - Creates RequestState
   - Creates RequestOutputCollector (queue)
   - Sends to EngineCore
   - Returns queue immediately
   ↓
3. API handler waits on queue:
   async for output in queue:
       ...

Meanwhile, in background task:

4. Background Loop (running continuously):
   while True:
       outputs = await engine_core.get_output_async()
         ↓ (internally calls EngineCore.step())

5. EngineCore.step():
   - Scheduler: form batch
   - Workers: execute forward pass
   - Returns EngineCoreOutputs
         ↓

6. OutputProcessor.process_outputs():
   - Detokenize: [42] → " How"
   - Check stop strings: not found
   - Compute logprobs: -2.3
   - Create RequestOutput
   - Push to queue: queue.put(output)
         ↓

7. API handler receives from queue:
   output = await queue.get()
   yield "data: {\"text\": \" How\"}\n\n"
         ↓

8. User receives SSE stream:
   data: {"text": " How"}
```

---

## Output Modes

### 1. FINAL_ONLY

Only return when generation is complete

```python
# No intermediate outputs
# Only final: "Hello world how are you?"
```

### 2. CUMULATIVE (default)

Each output contains ALL tokens so far

```python
# Output 1: "Hello"
# Output 2: "Hello world"
# Output 3: "Hello world how"
# Output 4: "Hello world how are you?"
```

### 3. DELTA

Each output contains only NEW tokens

```python
# Output 1: "Hello"
# Output 2: " world"
# Output 3: " how"
# Output 4: " are you?"
```

---

## Request State Tracking

**OutputProcessor maintains**:

```python
request_states: dict[str, RequestState]
```

**Key**: `request_id` (string like `"cmpl-abc123"`)

**Value**: `RequestState` object tracking:

- Detokenizer state
- Logprobs processor
- Tokens sent so far (for streaming)
- Queue (for AsyncLLM)
- Stats (latency, tokens/sec)

**Lookup flow**:

1. `add_request()` → `request_states["req-123"] = RequestState(...)`
2. EngineCore returns `EngineCoreOutput(request_id="req-123", ...)`
3. OutputProcessor looks up: `req_state = request_states["req-123"]`
4. Processes output for that specific request
5. Pushes to that request's queue

---

## Detokenization & Stop String Detection

**Incremental Detokenization**:

```python
detokenizer = IncrementalDetokenizer(...)

# Step 1: Add tokens [1, 2, 3]
detokenizer.update([1, 2, 3], is_finished=False)
# Returns: "Hello"

# Step 2: Add token [4]
detokenizer.update([4], is_finished=False)
# Returns: "Hello world"

# Step 3: Add token [5] - stop string!
stop_string = detokenizer.update([5], is_finished=False)
# Returns: "###" (detected stop string)
```

**Stop String Detection**:

If detokenizer finds a stop string:

1. Set `finish_reason = FinishReason.STOP`
2. Set `stop_reason = "###"` (the actual string)
3. Mark request as finished
4. Tell EngineCore to abort (free KV cache)

---

## Key Files Reference

### Request Intake

- `vllm/entrypoints/openai/api_server.py` - HTTP server
- `vllm/entrypoints/openai/serving_chat.py` - Chat endpoint
- `vllm/entrypoints/openai/serving_completion.py` - Completion endpoint
- `vllm/v1/engine/async_llm.py` - AsyncLLM (813 lines)
- `vllm/v1/engine/llm_engine.py` - LLMEngine (408 lines)
- `vllm/v1/engine/processor.py` - Processor (621 lines)
- `vllm/v1/engine/core_client.py` - EngineCoreClient (1400 lines)

### Output Processing

- `vllm/v1/engine/output_processor.py` - OutputProcessor (659 lines)
- `vllm/v1/engine/detokenizer.py` - Detokenizer (351 lines)
- `vllm/v1/engine/logprobs.py` - LogprobsProcessor (182 lines)

### Data Structures

- `vllm/v1/engine/__init__.py` - EngineCoreRequest, EngineCoreOutput (211 lines)
- `vllm/outputs.py` - RequestOutput, CompletionOutput
- `vllm/sampling_params.py` - SamplingParams

---

## Summary

The API Layer (Request Intake & Output Layer) is responsible for:

**Request Intake**:

1. Accept requests (HTTP or Python API)
2. Tokenize prompts
3. Validate parameters
4. Create EngineCoreRequest
5. Send to EngineCore
6. Return immediately (non-blocking)

**Background Loop** (AsyncLLM only):

1. Continuously call `step()` in background
2. Process all requests together (continuous batching)
3. Never blocks user requests

**Output Processing**:

1. Receive raw outputs from EngineCore
2. Detokenize tokens → text
3. Compute log probabilities
4. Detect stop strings
5. Push to per-request queues
6. Stream to users via HTTP/async

**Key Innovation**: Decoupled producer-consumer with continuous batching enables high throughput and low latency for concurrent requests.
