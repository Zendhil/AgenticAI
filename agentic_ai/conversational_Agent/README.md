# Conversational Agent

A memory-enabled conversational agent with dual memory systems:
- **Short-term memory** (Redis) - Recent conversation turns within current session
- **Long-term memory** (Neo4j) - Semantic search across all past conversations

## Architecture

See [AGENT.md](./AGENT.md) for complete design documentation.

## Setup

### 1. Environment Variables

Create a `.env` file with:

```bash
# Neo4j - Existing
PRD_GRF_DB_URI=bolt://localhost:7687
PRD_GRF_DB_USER=neo4j
PRD_GRF_DB_PWD=your_password
DEFAULT_DATABASE=neo4j

# Neo4j - Conversation Memory (NEW)
CONVERSATIONAL_DATABASE=convmemory

# Redis (NEW)
REDIS_URL=redis://localhost:6379
```

### 2. Create Neo4j Database

In Neo4j Browser, run:

```cypher
CREATE DATABASE convmemory IF NOT EXISTS
```

### 3. Install Dependencies

```bash
pip install redis sentence-transformers langchain-ollama
```

### 4. Start Services

```bash
# Start Redis
redis-server

# Start Neo4j (if not already running)
neo4j start

# Start Ollama with llama3.1
ollama run llama3.1
```

## Usage

### Run Interactive REPL

```bash
python -m agentic_ai.conversational_Agent.main
```

Commands:
- `/new` - Start a new session
- `/clear` - Clear current session memory
- `/quit` or `/exit` - Exit

### Programmatic Usage

```python
import asyncio
from agentic_ai.conversational_Agent.main import build_agent

async def example():
    # Build agent
    agent = await build_agent()

    # First message (creates new session)
    response, session_id = await agent.chat(
        user_message="Hello! My name is Alice.",
        user_id="user_123",
        session_id=None  # None = new session
    )
    print(f"Agent: {response}")

    # Continue conversation (uses same session)
    response, session_id = await agent.chat(
        user_message="What's my name?",
        user_id="user_123",
        session_id=session_id  # Pass session_id to continue
    )
    print(f"Agent: {response}")

asyncio.run(example())
```

## Memory Behavior

### Short-term Memory (Redis)
- Stores last 10 turns per session
- 2-hour TTL (sliding window)
- Used for immediate conversation context

### Long-term Memory (Neo4j)
- Stores all conversation turns with embeddings
- Retrieves top 5 semantically similar memories
- Similarity threshold: 0.7
- Separate database: `convmemory`

### Combined Context
The agent combines:
1. Recent turns from Redis (recency)
2. Semantically similar past memories from Neo4j (relevance)
3. Deduplication to avoid repeating recent context

## Project Structure

```
conversational_Agent/
├── AGENT.md              # Complete design document
├── README.md             # This file
├── main.py               # Entry point + wiring
├── memory/
│   ├── base.py           # BaseMemory ABC
│   ├── redis_memory.py   # Short-term memory
│   └── neo4j_memory.py   # Long-term memory
├── session/
│   └── session.py        # Session dataclass + SessionStore
└── agent/
    └── agent.py          # MemoryManager + ConversationalAgent
```

## Configuration

Edit `agentic_ai/common/llm_models.yaml`:

```yaml
llm_model:
  default: llama3.1:latest

embeddings:
  default: "all-MiniLM-L6-v2"  # 384 dims - fast, good quality
  # Alternative: "all-mpnet-base-v2"  # 768 dims - better, slower
```

**Note:** If changing embedding model, update vector index dimensions in Neo4j accordingly.

## Troubleshooting

### Redis Connection Error
```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG
```

### Neo4j Database Not Found
```cypher
// In Neo4j Browser
SHOW DATABASES;
CREATE DATABASE convmemory IF NOT EXISTS;
```

### Embedding Model Download
First run downloads ~80MB model. Subsequent runs use cached version.

### Neo4j Write Failures
Agent continues with Redis-only mode. Check Neo4j logs for connection issues.