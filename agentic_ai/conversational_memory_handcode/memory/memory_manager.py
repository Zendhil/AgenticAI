#standard library imports
import datetime
from abc import ABC,abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
import json
import logging
import asyncio
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#app library imports
from agentic_ai.conversational_memory_handcode.session import cache_client
from agentic_ai.conversational_memory_handcode.memory.session_manager import Message, SessionManager, Session, SessionKey
from agentic_ai.common import EmbeddingService


logger = logging.getLogger(__name__)


@dataclass
class UserMemory:
    """
    Memory-enabled user session manager.

    Manages user-specific memory including short-term message history.
    Long-term memory to be defined in future versions.
    """
    user_id: str
    short_term_memory:list[Message]
    #To be defined: Long term memory


class MemoryManager:
    """
    Memory-enabled conversational agent.

    Manages storage, retrieval, and compaction of conversation memory.
    """

    def __init__(self,session_manager:SessionManager=None):
        """
        Initializes the MemoryManager with an optional session manager.

        Args:
            session_manager: Optional SessionManager instance for handling session operations
        """
        logger.info("Inside class MemoryManager, function __init__")
        self._session_manager = session_manager


    async def _store(self, session_message:Session, final=False):
        """
        Stores conversation data using the provided session manager.

        Args:
            session_manager: SessionManager instance to handle storage operations

        Raises:
            Exception: If storage operation fails
        """
        logger.info("Inside class MemoryManager, function _store")
        try:
            if not final:
                await self._session_manager.save_session(session_message)
            #else:
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            raise


    async def _retrieve(self)->list[Message]:
        """
        Retrieves conversation data for the specified agent, user, and session.

        Args:
            agent_id: Unique identifier for the agent
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session

        Returns:
            Retrieved session data

        Raises:
            Exception: If retrieval operation fails
        """
        logger.info("Inside class MemoryManager, function _retrieve")
        try:
            # check relevant info in session context
            session_info = await self._session_manager.retrieve_info()

        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            raise
        return session_info

    async def fetch_relevant_info(self, user_query:str, threshold:float=0.65)->list[Message]:

        logger.info("Inside class MemoryManager, function fetch_relevant_info")
        try:
            # convert user-query into embeddings.
            # retrieve context from local cache. using retrieve_info()
            # convert each message in list into embddings.
            # find embeddings >0.65 cosine distance.
            # return that info as context

            # Initialize embedding service once
            embedding_service = EmbeddingService()

            # Get query embedding and reshape to 2D array for cosine_similarity
            query_embed = np.array(embedding_service.encode(user_query)).reshape(1, -1)

            # Retrieve session and get messages list
            messages = await self._retrieve()

            logger.info(f"Found {len(messages)} messages")
            if not messages:
                logger.warning("No messages found in session")
                return []

            # Encode all message contents and convert to numpy arrays
            msg_vectors = [np.array(embedding_service.encode(msg.content)).reshape(1, -1) for msg in messages]

            # Find messages with similarity above threshold
            context_messages = []
            for i, msg_vector in enumerate(msg_vectors):
                similarity_score = cosine_similarity(query_embed, msg_vector)[0][0]
                logger.info(f"similarity score of {messages[i]} {i}: {similarity_score}")
                if similarity_score >= threshold:
                    context_messages.append(messages[i])
                    logger.debug(f"Message {i} similarity: {similarity_score:.3f}")

            logger.info(f"Found {len(context_messages)} relevant messages out of {len(messages)}")
            return context_messages

        except Exception as e:
            logger.error(f"Error fetching relevant info: {e}")
            raise

    async def _compact(self,agent_id:str,user_id:str,session_id:str):
        """
        Compacts conversation memory for the specified agent, user, and session.

        Args:
            agent_id: Unique identifier for the agent
            user_id: Unique identifier for the user
            session_id: Unique identifier for the session

        Raises:
            Exception: If compaction operation fails
        """
        logger.info("Inside class MemoryManager, function _compact")
        try:
            # Implementation to be added
            pass
        except Exception as e:
            logger.error(f"Error compacting data: {e}")
            raise


# Sample test code for storing and retrieving from Redis
async def test_redis_operations():
    """
    Sample test code demonstrating storing and retrieving session data from Redis.

    This test:
    1. Creates a session key
    2. Creates messages
    3. Saves messages to Redis
    4. Retrieves messages from Redis
    5. Cleans up the test data
    """
    logger.info("Starting Redis operations test")

    try:
        # Step 1: Create a session key
        session_key = SessionKey(
            session_id="test_session_relevance",
            user_id="test_user_relevance",
            agent_id="test_agent_relevance"
        )
        logger.info(f"Created session key: {session_key.to_redis_key()}")

        # Step 2: Create sample messages
        message1 = Message(
            role="user",
            content="Hello, how are you?",
            timestamp=datetime.now()
        )

        message2 = Message(
            role="assistant",
            content="I'm doing well, thank you! How can I help you today?",
            timestamp=datetime.now()
        )

        logger.info(f"Created {2} test messages")

        # Step 3: Create a session with messages
        session = Session(
            key_=session_key,
            values_=[message1, message2]
        )

        # Step 4: Save session to Redis
        session_manager = SessionManager(session_key=session_key)
        await session_manager.save_session(session)
        logger.info("Successfully saved session to Redis")

        # Step 5: Retrieve session from Redis
        retrieved_session = await session_manager.retrieve_info()
        logger.info(f"Successfully retrieved session from Redis: {retrieved_session}")

        # Step 6: Verify data
        if retrieved_session:
            logger.info("Test PASSED: Data was successfully stored and retrieved from Redis")
            mem_man = MemoryManager(session_manager)
            await mem_man.fetch_relevant_info(user_query="what is Python")
        else:
            logger.warning("Test WARNING: No data retrieved from Redis")

        return retrieved_session

    except Exception as e:
        logger.error(f"Test FAILED with error: {e}")
        raise
    finally:
        # Cleanup: Close Redis connection
        try:
            await cache_client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")


if __name__ == "__main__":
    """
    Run the test when this file is executed directly.
    Make sure Redis is running on localhost:6379 before running this test.
    """
    print("=" * 60)
    print("Memory Manager - Redis Storage Test")
    print("=" * 60)
    print("Prerequisites: Redis server must be running on localhost:6379")
    print("=" * 60)

    # Run the async test
    asyncio.run(test_redis_operations())
    # asyncio.run(MemoryManager.fetch_relevant_info(user_query="What is python?"))
    print("=" * 60)
    print("Test completed. Check logs above for results.")
    print("=" * 60)
