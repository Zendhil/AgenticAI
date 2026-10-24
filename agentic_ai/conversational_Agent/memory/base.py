from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseMemory(ABC):
    """Abstract base class for memory backends."""

    # Role constants
    ROLE_HUMAN = "human"
    ROLE_AI = "ai"
    VALID_ROLES = {ROLE_HUMAN, ROLE_AI}

    @abstractmethod
    async def store(self, session_id: str, role: str, content: str) -> None:
        """Store a message in memory.

        Args:
            session_id: Session identifier
            role: Message role (human or ai)
            content: Message content
        """
        pass

    @abstractmethod
    async def retrieve(self, session_id: str, query: str = None) -> List[Dict[str, Any]]:
        """Retrieve messages from memory.

        Args:
            session_id: Session identifier
            query: Optional query for semantic search

        Returns:
            List of message dictionaries with role, content, timestamp
        """
        pass

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Clear all messages for a session.

        Args:
            session_id: Session identifier
        """
        pass

    def _validate_role(self, role: str) -> None:
        """Validate message role.

        Args:
            role: Role to validate

        Raises:
            ValueError: If role is not valid
        """
        if role not in self.VALID_ROLES:
            raise ValueError(f"Invalid role: {role}. Must be one of {self.VALID_ROLES}")