import json
import uuid
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a chat session."""

    session_id: str
    user_id: str
    created_at: str = None

    def __post_init__(self):
        """Set default created_at if not provided."""
        if self.created_at is None:
            self.created_at = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        """Convert session to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        """Create session from dictionary."""
        return cls(**data)


class SessionStore:
    """Manages session metadata in Redis."""

    def __init__(self, redis_client: Redis, ttl_seconds: int = 7200):
        """Initialize session store.

        Args:
            redis_client: Async Redis client instance
            ttl_seconds: Time-to-live for session metadata (default: 2 hours)
        """
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds

    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session metadata."""
        return f"session:{session_id}:meta"

    async def get_or_create(
        self,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Session:
        """Get existing session or create a new one.

        Args:
            user_id: User identifier
            session_id: Optional session identifier (creates new if None)

        Returns:
            Session instance
        """
        # Create new session if no session_id provided
        if session_id is None:
            session_id = str(uuid.uuid4())
            session = Session(session_id=session_id, user_id=user_id)
            await self._save(session)
            logger.info(f"Created new session: {session_id} for user: {user_id}")
            return session

        # Try to get existing session
        key = self._get_session_key(session_id)
        session_json = await self.redis_client.get(key)

        if session_json:
            # Session exists - load and refresh TTL
            session_data = json.loads(session_json)
            session = Session.from_dict(session_data)
            await self.redis_client.expire(key, self.ttl_seconds)
            logger.debug(f"Retrieved existing session: {session_id}")
            return session
        else:
            # Session doesn't exist - create new with provided session_id
            session = Session(session_id=session_id, user_id=user_id)
            await self._save(session)
            logger.info(f"Created session with provided ID: {session_id}")
            return session

    async def _save(self, session: Session) -> None:
        """Save session to Redis.

        Args:
            session: Session to save
        """
        key = self._get_session_key(session.session_id)
        session_json = json.dumps(session.to_dict())
        await self.redis_client.setex(key, self.ttl_seconds, session_json)

    async def expire(self, session_id: str) -> None:
        """Manually expire a session.

        Args:
            session_id: Session identifier
        """
        key = self._get_session_key(session_id)
        await self.redis_client.delete(key)
        logger.info(f"Expired session: {session_id}")