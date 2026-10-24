import logging
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agentic_ai.conversational_Agent.memory.base import BaseMemory
from agentic_ai.conversational_Agent.session.session import SessionStore

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages both short-term and long-term memory."""

    def __init__(self, short_term: BaseMemory, long_term: BaseMemory):
        """Initialize memory manager.

        Args:
            short_term: Short-term memory backend (Redis)
            long_term: Long-term memory backend (Neo4j)
        """
        self.short_term = short_term
        self.long_term = long_term

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add message to both short-term and long-term memory.

        Args:
            session_id: Session identifier
            role: Message role (human or ai)
            content: Message content
        """
        # Always write to Redis (short-term)
        await self.short_term.store(session_id, role, content)

        # Try to write to Neo4j (long-term) - non-fatal if it fails
        try:
            await self.long_term.store(session_id, role, content)
        except Exception as e:
            logger.warning(f"Failed to write to long-term memory (Neo4j): {e}")
            # Continue - conversation can proceed with Redis only

    async def get_context(self, session_id: str, query: str) -> List[Dict[str, Any]]:
        """Get combined context from recent and semantic memories.

        Args:
            session_id: Session identifier
            query: Current user message for semantic search

        Returns:
            Combined list of recent + unique semantic memories
        """
        # Get recent turns from Redis
        recent_turns = await self.short_term.retrieve(session_id)

        # Get semantically similar memories from Neo4j
        semantic_memories = []
        try:
            semantic_memories = await self.long_term.retrieve(session_id, query=query)
        except Exception as e:
            logger.warning(f"Failed to retrieve from long-term memory (Neo4j): {e}")
            # Continue with Redis context only

        # Deduplicate - remove semantic results already in recent turns
        recent_contents = {msg.get("content") for msg in recent_turns}
        unique_semantic = [
            msg for msg in semantic_memories
            if msg.get("content") not in recent_contents
        ]

        # Combine: recent turns + unique semantic memories
        combined_context = recent_turns + unique_semantic

        logger.debug(
            f"Context for session {session_id}: "
            f"{len(recent_turns)} recent, "
            f"{len(unique_semantic)} semantic"
        )

        return combined_context

    async def clear(self, session_id: str) -> None:
        """Clear both short-term and long-term memory.

        Args:
            session_id: Session identifier
        """
        await self.short_term.clear(session_id)
        try:
            await self.long_term.clear(session_id)
        except Exception as e:
            logger.warning(f"Failed to clear long-term memory (Neo4j): {e}")


class ConversationalAgent:
    """Memory-enabled conversational agent."""

    SYSTEM_PROMPT = """You are a helpful AI assistant with memory of past conversations.
You can remember what users have discussed with you before and use that context to provide better responses.
Be helpful, accurate, and conversational."""

    def __init__(self, memory_manager: MemoryManager, session_store: SessionStore, llm):
        """Initialize conversational agent.

        Args:
            memory_manager: MemoryManager instance
            session_store: SessionStore instance
            llm: LangChain LLM instance (ChatOllama)
        """
        self.memory_manager = memory_manager
        self.session_store = session_store
        self.llm = llm

    async def chat(
        self,
        user_message: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Tuple[str, str]:
        """Process a chat turn.

        Args:
            user_message: User's message
            user_id: User identifier
            session_id: Optional session identifier (creates new if None)

        Returns:
            Tuple of (response_text, session_id)
        """
        # Get or create session
        session = await self.session_store.get_or_create(user_id, session_id)
        session_id = session.session_id

        # Get context from memory
        context_turns = await self.memory_manager.get_context(session_id, user_message)

        # Build messages for LLM
        messages = self._build_messages(context_turns, user_message)

        # Get LLM response
        try:
            response = await self.llm.ainvoke(messages)
            response_text = response.content
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            response_text = "I'm sorry, I encountered an error processing your request."

        # Store both turns in memory
        await self.memory_manager.add_message(
            session_id,
            BaseMemory.ROLE_HUMAN,
            user_message
        )
        await self.memory_manager.add_message(
            session_id,
            BaseMemory.ROLE_AI,
            response_text
        )

        return response_text, session_id

    def _build_messages(
        self,
        context_turns: List[Dict[str, Any]],
        current_message: str
    ) -> List:
        """Build LangChain messages from context and current message.

        Args:
            context_turns: List of context message dictionaries
            current_message: Current user message

        Returns:
            List of LangChain message objects
        """
        messages = [SystemMessage(content=self.SYSTEM_PROMPT)]

        # Add context turns
        for turn in context_turns:
            role = turn.get("role")
            content = turn.get("content")

            if role == BaseMemory.ROLE_HUMAN:
                messages.append(HumanMessage(content=content))
            elif role == BaseMemory.ROLE_AI:
                messages.append(AIMessage(content=content))

        # Add current message
        messages.append(HumanMessage(content=current_message))

        return messages