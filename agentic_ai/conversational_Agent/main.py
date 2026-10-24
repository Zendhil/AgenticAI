"""
Conversational Agent - Main Entry Point

Wires together all components and provides an async REPL loop for testing.
"""

import asyncio
import logging
import os
import redis.asyncio as redis
from agentic_ai.common.environment import get_redis_url
from agentic_ai.common.graph_setup import get_graph_driver
from agentic_ai.common.config import get_llm_model, EmbeddingService
from agentic_ai.conversational_Agent.memory.redis_memory import RedisMemory
from agentic_ai.conversational_Agent.memory.neo4j_memory import Neo4jMemory
from agentic_ai.conversational_Agent.session.session import SessionStore
from agentic_ai.conversational_Agent.agent.agent import MemoryManager, ConversationalAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def build_agent() -> ConversationalAgent:
    """Build and initialize the conversational agent with all dependencies.

    Returns:
        ConversationalAgent instance ready to use
    """
    logger.info("Building conversational agent...")

    # Get Redis client
    redis_url = get_redis_url()
    redis_client = await redis.from_url(redis_url, decode_responses=True)
    logger.info(f"Connected to Redis: {redis_url}")

    # Get Neo4j driver
    neo4j_driver = get_graph_driver()
    logger.info("Neo4j driver initialized")

    # Get conversation database name from environment
    conv_database = os.getenv("CONVERSATIONAL_DATABASE", "convmemory")
    logger.info(f"Using Neo4j database: {conv_database}")

    # Initialize embedding service (singleton)
    embedding_service = EmbeddingService.get_instance()
    logger.info(f"Embedding service initialized with dimension: {embedding_service.dimension}")

    # Initialize memory backends
    redis_memory = RedisMemory(redis_client, ttl_seconds=7200, max_turns=10)
    neo4j_memory = Neo4jMemory(
        driver=neo4j_driver,
        embedding_service=embedding_service,
        database=conv_database,
        top_k=5,
        similarity_threshold=0.7
    )

    # Create vector index if not exists
    logger.info("Creating/verifying vector index...")
    await neo4j_memory.create_index_if_not_exists()

    # Initialize session store
    session_store = SessionStore(redis_client, ttl_seconds=7200)

    # Initialize memory manager
    memory_manager = MemoryManager(short_term=redis_memory, long_term=neo4j_memory)

    # Get LLM
    llm = get_llm_model()
    logger.info("LLM initialized")

    # Create agent
    agent = ConversationalAgent(
        memory_manager=memory_manager,
        session_store=session_store,
        llm=llm
    )

    logger.info("Conversational agent ready!")
    return agent


async def run_repl(agent: ConversationalAgent, user_id: str = "test_user"):
    """Run an interactive REPL loop for testing the agent.

    Args:
        agent: ConversationalAgent instance
        user_id: User identifier for the session
    """
    print("\n" + "="*60)
    print("Conversational Agent - Interactive REPL")
    print("="*60)
    print("Commands:")
    print("  /quit or /exit - Exit the REPL")
    print("  /new - Start a new session")
    print("  /clear - Clear current session memory")
    print("="*60 + "\n")

    session_id = None

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ["/quit", "/exit"]:
                print("Goodbye!")
                break

            if user_input.lower() == "/new":
                session_id = None
                print("Started new session")
                continue

            if user_input.lower() == "/clear":
                if session_id:
                    await agent.memory_manager.clear(session_id)
                    print("Cleared session memory")
                else:
                    print("No active session to clear")
                continue

            # Process message
            response, session_id = await agent.chat(
                user_message=user_input,
                user_id=user_id,
                session_id=session_id
            )

            print(f"Assistant: {response}\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in REPL: {e}", exc_info=True)
            print(f"Error: {e}\n")


async def main():
    """Main entry point."""
    try:
        # Build agent
        agent = await build_agent()

        # Run REPL
        await run_repl(agent)

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())