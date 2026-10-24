# Conversational Agent — Design Document

> Feed this file to Claude Code as context before making any changes.
> It describes the full architecture, design decisions, and constraints.

---

## Purpose

A memory-enabled conversational agent that remembers:
- **Short term** — what was said in the current session (Redis)
- **Long term** — what was said across all past sessions, searchable by meaning (Neo4j vector)

---

## Tech Stack

| Concern | Technology |
|---|---|
| LLM | Ollama `llama3.1` via LangChain `ChatOllama` |
| Embeddings | `sentence-transformers` (local, no API cost) |
| Short term memory | Redis — session scoped, TTL based |
| Long term memory | Neo4j — vector similarity search |
| Session store | Redis — session metadata, TTL based |
| Async runtime | Python `asyncio` throughout |
| Existing driver | `AsyncGraphDatabase.driver` from Neo4j |

---

## Project Structure

```
agentic_ai/
|--common
├  |── environment.py          # device detection, Neo4j driver, Redis URL — EXISTING, extended
|  ├── config.py               # LLM + EmbeddingService singletons — EXISTING, extended
├  |── llm_models.yaml         # model names config — EXISTING, unchanged
├── memory/
│   ├── __init__.py
│   ├── base.py             # BaseMemory ABC — contract for all memory backends
│   ├── redis_memory.py     # short term memory — last N turns per session
│   └── neo4j_memory.py     # long term memory — vector similarity search
├── session/
│   ├── __init__.py
│   └── session.py          # Session dataclass + SessionStore
├── agent/
│   ├── __init__.py
│   └── agent.py            # MemoryManager + ConversationalAgent
└── main.py                 # wiring + async REPL loop
```

---

## Class Design

### `BaseMemory` (ABC) — `memory/base.py`

```
BaseMemory
    store(session_id, role, content) → None   [abstractmethod]
    retrieve(session_id, query)      → list[dict]  [abstractmethod]
    clear(session_id)                → None   [abstractmethod]
    _validate_role(role)             → None   [concrete — shared validation]

Constants:
    ROLE_HUMAN = "human"
    ROLE_AI    = "ai"
    VALID_ROLES = {"human", "ai"}
```

**Why ABC not Protocol:**
- We own both implementations (Redis, Neo4j)
- Hard enforcement — missing method = `TypeError` at instantiation, not silently at runtime
- Shared constants (`ROLE_HUMAN`, `VALID_ROLES`) belong here

---

### `RedisMemory(BaseMemory)` — `memory/redis_memory.py`

```
RedisMemory
    __init__(redis_client, ttl_seconds=7200, max_turns=10)
    store(session_id, role, content)  → append turn to Redis list, reset TTL
    retrieve(session_id, query=None)  → last max_turns turns (query ignored)
    clear(session_id)                 → delete Redis key
```

**Redis key schema:**
```
session:{session_id}:history  →  JSON list of {role, content, timestamp}
```

**Design decisions:**
- `redis_client` injected — not created inside class (testability)
- Uses pipeline for atomic `RPUSH` + `EXPIRE` on every write
- TTL is sliding window — resets on every message (2hr inactivity = expire)
- `query` param accepted but ignored — Redis uses recency not semantics

---

### `Neo4jMemory(BaseMemory)` — `memory/neo4j_memory.py`

```
Neo4jMemory
    __init__(driver, embedding_service, database, top_k=5, similarity_threshold=0.7)
    create_index_if_not_exists()      → creates vector index on startup (async)
    store(session_id, role, content)  → embed content, create Memory node, link to Session
    retrieve(session_id, query)       → vector similarity search, returns top_k results
    clear(session_id)                 → DETACH DELETE all Memory nodes for session
    _retrieve_recent(session_id)      → fallback when no query — most recent 5 memories
```

**Graph schema:**
```
(:User {id, created_at})
    -[:HAS_SESSION]->
(:Session {id, user_id, created_at})
    -[:HAS_MEMORY]->
(:Memory {
    id,           uuid
    session_id,   string
    role,         "human" | "ai"
    content,      string — the message text
    timestamp,    ISO datetime string
    embedding     list[float] — 384 dims (sentence-transformers default)
})
```

**Vector index:**
```cypher
CREATE VECTOR INDEX memory_embedding IF NOT EXISTS
FOR (m:Memory) ON m.embedding
OPTIONS {indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
}}
```

**Design decisions:**
- `driver` and `embedding_service` both injected — not created inside class
- `create_index_if_not_exists()` separate from `__init__` because `__init__` cannot be async
- Neo4j write failure in `MemoryManager` is non-fatal — conversation continues
- `MERGE` on Session/User (no duplicates), `CREATE` on Memory (every turn is new)
- Similarity threshold 0.7 — below this, memory not relevant enough to include

**Separate database for conversation memory:**
```
Neo4j instance
    ├── "neo4j" / "prodgraph"  ← existing GraphRepository (document chunks + entities)
    └── "convmemory"           ← Neo4jMemory (User, Session, Memory nodes)
```

Create with: `CREATE DATABASE convmemory IF NOT EXISTS` in Neo4j Browser.

---

### `EmbeddingService` (singleton) — `config.py`

```
EmbeddingService
    get_instance()        → classmethod, returns singleton
    encode(text)          → list[float]  (384 dims)
    dimension             → int property

Internal:
    _model: SentenceTransformer   loaded once on first get_instance() call
    _dim:   int                   from model.get_sentence_embedding_dimension()
```

**Why singleton:**
- `SentenceTransformer` loads ~500MB model from disk
- Loading once per process is critical — reloading per request would be seconds of latency
- Device (MPS/CUDA/CPU) detected once from `environment.get_device()`

**Model name source:**
- Comes from `config.get_embeddings_model()` → reads from `llm_models.yaml`
- `EmbeddingService` does NOT hardcode the model name — reads from config

---

### `Session` (dataclass) + `SessionStore` — `session/session.py`

```
@dataclass
Session
    session_id: str
    user_id:    str
    created_at: str   (ISO datetime, default = now)
    to_dict()   → dict
    from_dict() → Session  (classmethod)

SessionStore
    __init__(redis_client)
    get_or_create(user_id, session_id=None) → Session
    expire(session_id)                      → None
```

**Redis key schema:**
```
session:{session_id}:meta  →  JSON of Session fields
```

**Why Session is a dataclass not a dict:**
- `session.user_id` vs `session['user_id']` — readable, explicit
- Type safety — fields declared, IDE autocomplete works
- `from_dict` / `to_dict` for Redis serialization

**Why SessionStore uses Redis not Neo4j:**
- Sessions are ephemeral — TTL based, no need for permanent persistence
- Sub-millisecond reads — needed before every chat turn
- Neo4j `Session` nodes (in the graph) track memory history — different concept from active chat session

---

### `MemoryManager` — `agent/agent.py`

```
MemoryManager
    __init__(short_term: BaseMemory, long_term: BaseMemory)
    add_message(session_id, role, content) → writes to both Redis + Neo4j
    get_context(session_id, query)         → combines recent + semantic, deduplicates
    clear(session_id)                      → clears both memories
```

**Context building strategy:**
1. `short_term.retrieve()` → last N turns (recent conversation flow)
2. `long_term.retrieve(query=message)` → top K semantically similar past memories
3. Deduplicate — remove semantic results already in recent turns
4. Return combined: `recent_turns + unique_semantic_memories`

**Fault tolerance:**
- Neo4j write failure → logged as WARNING, not raised — conversation continues
- Neo4j retrieve failure → logged as WARNING, returns empty — Redis context still used

**Why `MemoryManager` is not an ABC:**
- Single implementation — no variation expected
- Orchestration logic only — no need for abstraction here

---

### `ConversationalAgent` — `agent/agent.py`

```
ConversationalAgent
    __init__(memory_manager, session_store, llm)
    chat(user_message, user_id, session_id=None) → (response_text, session_id)
    _build_messages(context_turns, current_message) → list[LangChain messages]
```

**Chat flow per turn:**
```
1. SessionStore.get_or_create(user_id, session_id)
        ↓
2. MemoryManager.get_context(session_id, query=user_message)
        ↓ returns combined recent + semantic turns
3. _build_messages(context_turns, user_message)
        ↓ [SystemMessage] + [HumanMessage/AIMessage from context] + [HumanMessage current]
4. llm.ainvoke(messages)
        ↓ ChatOllama async call
5. MemoryManager.add_message(session_id, HUMAN, user_message)
   MemoryManager.add_message(session_id, AI, response_text)
        ↓ write both turns to Redis + Neo4j
6. return (response_text, session_id)
```

**Returns `session_id`:**
- Caller passes `session_id=None` on first turn → agent creates new session
- Agent returns `session_id` → caller passes it back next turn → session continues
- Caller can persist `session_id` to resume across process restarts

---

## Dependency Graph

```
llm_models.yaml
      ↓
  Config (singleton)
      ├── get_llm_model()        → ChatOllama(llama3.1)
      └── get_embeddings_model() → "model-name-string"

Environment (singleton)
      ├── get_async_driver()     → AsyncGraphDatabase.driver
      ├── get_redis_url()        → "redis://localhost:6379"
      └── get_device()           → "mps" | "cuda" | "cpu"

EmbeddingService (singleton)
      ├── uses: Config.get_embeddings_model() for model name
      └── uses: Environment.get_device() for device
            ↓
      SentenceTransformer(model_name, device=device)

redis.asyncio.from_url(get_redis_url())
      ├── RedisMemory(redis_client)
      └── SessionStore(redis_client)

Neo4jMemory(driver=get_async_driver(), embedding_service=EmbeddingService.get_instance())

MemoryManager(short_term=RedisMemory, long_term=Neo4jMemory)

ConversationalAgent(memory_manager, session_store, llm=get_llm_model())
```

All wiring happens in `main.py:build_agent()`. Classes do not create their own dependencies.

---

## Environment Variables

```bash
# Neo4j — existing
PRD_GRF_DB_URI=bolt://localhost:7687
PRD_GRF_DB_USER=neo4j
PRD_GRF_DB_PWD=your_password
DEFAULT_DATABASE=neo4j

# Neo4j — new (conversation memory)
CONVERSATIONAL_DATABASE=convmemory

# Redis — new
REDIS_URL=redis://localhost:6379

# Ollama — check your existing yaml
# model name lives in llm_models.yaml, not .env
```

---

## llm_models.yaml expected shape

```yaml
llm_model:
  default: llama3.1

embeddings:
  default: all-MiniLM-L6-v2    # 384 dims — fast, good quality
  # alternatives:
  # default: all-mpnet-base-v2  # 768 dims — better quality, slower
```

If using a different sentence-transformers model, update `vector.dimensions` in the vector index creation to match.

---

## Existing GraphRepository — separate concern

`GraphRepository` (your existing class) is a **Knowledge Graph for documents**:
- Stores: `Chunk`, `Entity` nodes — document text + extracted entities
- Purpose: RAG — "what does my document say about X?"
- Database: `prodgraph` / `neo4j`

`Neo4jMemory` (new) is a **Conversation Memory store**:
- Stores: `User`, `Session`, `Memory` nodes — conversation turns
- Purpose: Memory — "what did this user say/ask before?"
- Database: `convmemory`

Same Neo4j instance, different databases, different schemas, different purposes.
They can be combined in future — e.g. link a `Memory` node to `Chunk` nodes it referenced.

---

## Key Design Principles Applied

| Principle | Where |
|---|---|
| Dependency Injection | `redis_client`, `driver`, `embedding_service` all injected, never created inside class |
| Singleton | `Environment`, `Config`, `EmbeddingService` — expensive to create, shared |
| ABC for owned hierarchy | `BaseMemory` — own both impls, want hard enforcement |
| Dataclass for pure data | `Session` — no behavior, just fields |
| Fault tolerance | Neo4j failures non-fatal in `MemoryManager` |
| Async throughout | All IO methods are `async` — matches existing Neo4j driver |
| Separation of concerns | `SessionStore` ≠ Neo4j Session node — active session vs memory graph |

---

## What Claude Code Should Know

1. **Do not instantiate `SentenceTransformer` directly** — always use `EmbeddingService.get_instance()`
2. **Do not create Redis client inside memory classes** — inject from `main.py`
3. **Do not create Neo4j driver inside memory classes** — inject from `environment.get_async_driver()`
4. **`create_index_if_not_exists()` must be awaited on startup** — called in `main.py:build_agent()`, not in `__init__`
5. **Neo4j write/read failures in `MemoryManager` are non-fatal** — log WARNING, do not raise
6. **All new IO methods must be `async def`** — this is an async codebase throughout
7. **Session in Redis** (`SessionStore`) and **Session node in Neo4j** (`Neo4jMemory`) are different things — do not confuse
8. **Vector index dimension must match embedding model** — default 384 for `all-MiniLM-L6-v2`
