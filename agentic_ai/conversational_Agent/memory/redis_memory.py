import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from redis.asyncio import Redis
from agentic_ai.conversational_Agent.memory.base import BaseMemory

logger = logging.getLogger(__name__)


class RedisMemory(BaseMemory):
    """Short-term memory using Redis for recent conversation turns."""

    def __init__(self, redis_client: Redis, ttl_seconds: int = 7200, max_turns: int = 10):
        """Initialize Redis memory.

        Args:
            redis_client: Async Redis client instance
            ttl_seconds: Time-to-live for session data (default: 2 hours)
            max_turns: Maximum number of turns to keep (default: 10)
        """
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns

    def _get_history_key(self, session_id: str) -> str:
        """Get Redis key for session history."""
        return f"session:{session_id}:history"

    async def store(self, session_id: str, role: str, content: str) -> None:
        """Store a message in Redis.

        Args:
            session_id: Session identifier
            role: Message role (human or ai)
            content: Message content
        """
        self._validate_role(role)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }

        key = self._get_history_key(session_id)

        # Use pipeline for atomic operations
        async with self.redis_client.pipeline() as pipe:
            # Add message to list
            await pipe.rpush(key, json.dumps(message))
            # Reset TTL (sliding window)
            await pipe.expire(key, self.ttl_seconds)
            # Trim to max_turns (keep most recent)
            await pipe.ltrim(key, -self.max_turns, -1)
            await pipe.execute()

        logger.debug(f"Stored {role} message in Redis for session {session_id}")

    async def retrieve(self, session_id: str, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve recent messages from Redis.

        Args:
            session_id: Session identifier
            query: Ignored for Redis (uses recency not semantics)

        Returns:
            List of message dictionaries with role, content, timestamp
        """
        key = self._get_history_key(session_id)

        # Get all messages from the list
        messages_json = await self.redis_client.lrange(key, 0, -1)

        messages = []
        for msg_json in messages_json:
            try:
                message = json.loads(msg_json)
                messages.append(message)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message from Redis: {e}")
                continue

        logger.debug(f"Retrieved {len(messages)} messages from Redis for session {session_id}")
        return messages

    async def clear(self, session_id: str) -> None:
        """Clear all messages for a session.

        Args:
            session_id: Session identifier
        """
        key = self._get_history_key(session_id)
        await self.redis_client.delete(key)
        logger.info(f"Cleared Redis memory for session {session_id}")