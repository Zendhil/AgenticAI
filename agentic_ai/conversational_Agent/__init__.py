"""
Conversational Agent Package

A memory-enabled conversational agent with short-term (Redis) and
long-term (Neo4j vector) memory capabilities.
"""

from agentic_ai.conversational_Agent.agent.agent import ConversationalAgent, MemoryManager
from agentic_ai.conversational_Agent.memory.base import BaseMemory
from agentic_ai.conversational_Agent.memory.redis_memory import RedisMemory
from agentic_ai.conversational_Agent.memory.neo4j_memory import Neo4jMemory
from agentic_ai.conversational_Agent.session.session import Session, SessionStore

__all__ = [
    "ConversationalAgent",
    "MemoryManager",
    "BaseMemory",
    "RedisMemory",
    "Neo4jMemory",
    "Session",
    "SessionStore",
]