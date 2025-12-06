# vLLM Engine Core

## Code Structure

### Directory: `vllm/v1/engine/` (14 files, ~7700 lines)

#### Core Execution Files

**core.py** (1420 lines)

- `EngineCore` - Main engine class, orchestrates scheduling + execution
- `EngineCoreProc` - Runs EngineCore in separate process, handles ZMQ communication
- `DPEngineCoreProc` - Data parallel version of EngineCoreProc
- `DPEngineCoreActor` - Actor-based data parallel version

**core_client.py** (1400 lines)

- `EngineCoreClient` (ABC) - Abstract client interface
- `InprocClient` - Same-process communication (direct method calls)
- `MPClient` - Multi-process communication via ZMQ sockets
- `SyncMPClient` - Synchronous multi-process client
- `AsyncMPClient` - Asynchronous multi-process client
- `DPAsyncMPClient` - Data parallel async client
- `DPLBAsyncMPClient` - Data parallel with load balancing

**llm_engine.py** (408 lines)

- `LLMEngine` - High-level engine wrapper, provides backwards compatibility API

#### Request/Output Processing

**processor.py** (621 lines)

- `Processor` - Tokenizes input prompts, prepares EngineCoreRequest objects

**output_processor.py** (659 lines)

- `OutputProcessor` - Processes EngineCoreOutputs into RequestOutputs
- `RequestState` - Tracks state of each request
- `RequestOutputCollector` - Collects outputs for streaming

**detokenizer.py** (351 lines)

- `IncrementalDetokenizer` (ABC) - Interface for detokenization
- `FastIncrementalDetokenizer` - Fast detokenization implementation
- `SlowIncrementalDetokenizer` - Fallback detokenization

#### Data Structures & Support

**__init__.py** (211 lines)

- `EngineCoreRequest` - Input request format
- `EngineCoreOutput` - Single request output
- `EngineCoreOutputs` - Batch of outputs
- `FinishReason` - Enum for completion reasons
- `EngineCoreRequestType` - Request type enum

**utils.py** (1072 lines)

- `CoreEngineProcManager` - Manages EngineCore processes
- `CoreEngineActorManager` - Manages EngineCore actors
- Helper functions for process management

#### Other Files

**async_llm.py** (813 lines)

- `AsyncLLM` - Async wrapper around engine

**coordinator.py** (377 lines)

- `DPCoordinator` - Coordinates data parallel engines

**logprobs.py** (182 lines)

- `LogprobsProcessor` - Processes log probabilities

**parallel_sampling.py** (145 lines)

- `ParentRequest` - Handles n>1 sampling (multiple outputs per prompt)

**additional_heads.py** (46 lines)

- `AdditionalHeadsProcessor` - Handles additional model heads

**exceptions.py** (18 lines)

- `EngineGenerateError` - Generation errors
- `EngineDeadError` - Engine crash errors

---

## Call Chain: API Server → Engine Core

### Full Request Path

```
HTTP Client
    ↓ (HTTP request)
API Server (vllm/entrypoints/openai/api_server.py)
    ↓ (creates)
LLMEngine (vllm/v1/engine/llm_engine.py)
    ↓ (delegates to)
EngineCoreClient (vllm/v1/engine/core_client.py)
    ├─ InprocClient (same process) → direct call
    └─ MPClient (separate process) → ZMQ socket
        ↓ (serialized request)
EngineCoreProc (vllm/v1/engine/core.py)
    ↓ (wraps)
EngineCore (vllm/v1/engine/core.py)
    ↓ (coordinates)
Scheduler + Workers + Block Manager
```

### Detailed Flow

#### 1. API Server Layer

- **Location**: `vllm/entrypoints/openai/api_server.py`
- **Responsibility**: HTTP endpoint handling
- **Actions**:
  - Receives HTTP POST to `/v1/completions` or `/v1/chat/completions`
  - Parses JSON request
  - Creates `LLMEngine` instance (if not already created)

#### 2. LLMEngine Layer

- **Location**: `vllm/v1/engine/llm_engine.py`
- **Responsibility**: High-level request orchestration
- **Key Methods**:
  - `add_request()` - Accepts new prompts
  - `step()` - Main inference loop iteration
- **Actions**:
  - Uses `Processor` to tokenize prompt
  - Creates `EngineCoreRequest` object
  - Delegates to `self.engine_core` (EngineCoreClient)

#### 3. EngineCoreClient Layer

- **Location**: `vllm/v1/engine/core_client.py`
- **Responsibility**: Abstraction over in-process vs multi-process
- **Two Modes**:

**InprocClient** (same process):

- Direct method calls to `EngineCore`
- No serialization overhead
- Used for simple single-node deployments

**MPClient** (multi-process via ZMQ):

- Serializes request using msgspec
- Sends over ZMQ socket
- Enables process isolation and fault tolerance
- Used for production deployments

#### 4. EngineCoreProc Layer

- **Location**: `vllm/v1/engine/core.py`
- **Responsibility**: Process boundary and message handling
- **Actions**:
  - Runs `run_busy_loop()` in separate process
  - Listens on ZMQ sockets for requests
  - Deserializes incoming messages
  - Dispatches to `EngineCore` methods
  - Serializes and sends back outputs

#### 5. EngineCore Layer

- **Location**: `vllm/v1/engine/core.py`
- **Responsibility**: Core inference orchestration
- **Key Methods**:
  - `add_request()` - Queue request to scheduler
  - `step()` - Execute one inference iteration
  - `abort_requests()` - Cancel requests
- **Actions**:
  - Calls `scheduler.schedule()` to get batch
  - Calls `model_executor.execute()` to run inference
  - Returns `EngineCoreOutputs`

### Return Path

```
EngineCore
    ↓ (EngineCoreOutputs)
EngineCoreProc
    ↓ (serialized via ZMQ)
MPClient
    ↓ (deserializes)
LLMEngine
    ↓ (uses OutputProcessor)
Detokenizer + Sampler
    ↓ (RequestOutput)
API Server
    ↓ (JSON response)
HTTP Client
```

---

## Key Classes and Their Responsibilities

### EngineCore (core.py)

**Purpose**: The actual inference engine

- Initialize model executor, scheduler, KV cache
- Coordinate scheduling and execution
- Manage request lifecycle

**Key Methods**:

- `__init__()` - Setup executor, scheduler, block manager
- `add_request(request: Request)` - Add to scheduler queue
- `step() -> EngineCoreOutputs` - Run one inference iteration
- `abort_requests(request_ids)` - Cancel requests

### EngineCoreClient (core_client.py)

**Purpose**: Client interface for talking to EngineCore

- Abstract away in-process vs multi-process communication
- Provide consistent API regardless of deployment mode

**Why it exists**:

- Same code works whether EngineCore is local or remote
- Enables flexible deployment (single process, multi-process, distributed)

### LLMEngine (llm_engine.py)

**Purpose**: High-level wrapper and backwards compatibility

- Provide user-facing API
- Handle preprocessing (tokenization)
- Handle postprocessing (detokenization, sampling)
- Manage output streaming

**Key insight**: LLMEngine is NOT the engine - it's a wrapper around EngineCore!

### Processor (processor.py)

**Purpose**: Convert raw prompts into tokenized requests

- Tokenize text prompts
- Handle multimodal inputs
- Create `EngineCoreRequest` objects

### OutputProcessor (output_processor.py)

**Purpose**: Convert engine outputs into user-facing results

- Detokenize tokens to text
- Apply sampling to logits
- Check stopping conditions
- Handle streaming outputs
- Track request state
