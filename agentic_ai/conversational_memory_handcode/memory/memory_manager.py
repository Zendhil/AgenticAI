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
from agentic_ai.common import EmbeddingService

logger = logging.getLogger(__name__)

@dataclass
class SessionKey:
    """Represents a unique session identifier composed of user, session, and agent IDs."""
    session_id: str
    user_id: str
    agent_id: str

    def to_redis_key(self) -> str:
        """
        Converts the session key to a Redis key format.

        Returns:
            str: Formatted Redis key in the pattern 'session:user_id:session_id:agent_id'
        """
        logger.info("Inside class SessionKey, function to_redis_key")
        return f"session:{self.user_id}:{self.session_id}:{self.agent_id}"

@dataclass
class Message:
    """Represents a conversation message with role, content, and timestamp."""
    role:str
    content: str
    timestamp: datetime

    def to_redis_value(self) -> str:
        """
        Serializes the message to a JSON string for Redis storage.

        Returns:
            str: JSON string containing role, content, and ISO-formatted timestamp
        """
        logger.info("Inside class Message, function to_redis_value")
        return json.dumps({
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat()
        })

@dataclass
class Session:
    """Represents a conversation session containing a key and list of messages."""
    key_: SessionKey
    values_: list[Message] =field(default_factory=list)
    # def to_dict(self) -> dict:
    #     return asdict(self)

# @dataclass
# class ShortTermMemory:



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

class SessionManager:
    """
    Memory-enabled session manager.

    Handles saving and retrieving session information from Redis cache.
    """
    def __init__(self,session_key:SessionKey):
        """
        Initializes the SessionManager with a session key.

        Args:
            session_key: SessionKey object containing user, session, and agent identifiers
        """
        logger.info("Inside class SessionManager, function __init__")
        self._session_key = session_key

    async def save_session(self, session_info:Session):
        """
        Saves session information to Redis cache.

        Args:
            session_info: Session object containing key and message values

        Raises:
            Exception: If Redis save operation fails
        """
        logger.info("Inside class SessionManager, function _save_session")
        try:
            key_=session_info.key_.to_redis_key()
            values_list =[value_.to_redis_value() for value_ in session_info.values_]
            await cache_client.save_session(key_, values_list)
        except Exception as e:
            logger.error(f"Error saving session to Redis: {e}")
            raise

    async def retrieve_info(self)->list[Message]:
        """
        Retrieves session information from Redis cache.

        Returns:
            Session: Retrieved session object

        Raises:
            Exception: If Redis retrieve operation fails
        """
        logger.info("Inside class SessionManager, function _retrieve_info")
        try:
            key_=self._session_key.to_redis_key()
            print("*************Key for the retrieval***************", key_)
            messages = await cache_client.retrieve_session(key_=key_)
            return messages
        except Exception as e:
            logger.error(f"Error retrieving session from Redis: {e}")
            raise

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
            session = await self._retrieve()
            messages = session.values_ if hasattr(session, 'values_') else []

            if not messages:
                logger.warning("No messages found in session")
                return []

            # Encode all message contents and convert to numpy arrays
            msg_vectors = [np.array(embedding_service.encode(msg.content)).reshape(1, -1) for msg in messages]

            # Find messages with similarity above threshold
            context_messages = []
            for i, msg_vector in enumerate(msg_vectors):
                similarity_score = cosine_similarity(query_embed, msg_vector)[0][0]
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
            session_id="test_session_456",
            user_id="test_user_456",
            agent_id="test_agent_789"
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

    print("=" * 60)
    print("Test completed. Check logs above for results.")
    print("=" * 60)
