#standard library imports
import logging
from datetime import datetime
import json
from dataclasses import dataclass, field, fields
from uuid import uuid4
#Third party library imports

#App library imports
from agentic_ai.conversational_memory_handcode.session import cache_client

logger = logging.getLogger(__name__)

@dataclass
class SessionKey:
    """Represents a unique session identifier composed of user, session, and agent IDs."""
    session_id: str | None
    user_id: str
    agent_id: str

    def to_redis_key(self) -> str:
        """
        Converts the session key to a Redis key format.

        Returns:
            str: Formatted Redis key in the pattern 'session:user_id:session_id:agent_id'
        """
        logger.info("Inside class SessionKey, function to_redis_key")
        return f"{self.session_id}:{self.agent_id}:{self.user_id}"

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

    @classmethod
    def from_dict(cls, data: dict):
        valid_fields = {f.name for f in fields(cls)}

        missing_fields = valid_fields - data.keys()

        if missing_fields:
            raise ValueError(
                f"Missing required fields for {cls.__name__}: {missing_fields}"
            )

        return cls(**{k: v for k, v in data.items() if k in valid_fields})


@dataclass
class Session:
    """Represents a conversation session containing a key and list of messages."""
    key_: SessionKey
    values_: list[Message] =field(default_factory=list)
    # def to_dict(self) -> dict:
    #     return asdict(self)

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
        if self._session_key.session_id is None:
            self._session_key.session_id = str(uuid4())
        if self._session_key.agent_id is None:
            logger.error("In Session Manager:init:Agent ID not provided")
            raise ValueError("Agent ID not provided")
        if self._session_key.user_id is None:
            logger.error("In Session Manager:init:User ID not provided")
            raise ValueError("User ID not provided")

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
            messages_dict = await cache_client.retrieve_session(key_=key_)
            messages = [Message.from_dict(data=message) for message in messages_dict]
            return messages
        except Exception as e:
            logger.error(f"Error retrieving session from Redis: {e}")
            raise
