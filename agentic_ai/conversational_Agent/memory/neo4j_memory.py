import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from neo4j import AsyncDriver
from agentic_ai.conversational_Agent.memory.base import BaseMemory

logger = logging.getLogger(__name__)


class Neo4jMemory(BaseMemory):
    """Long-term memory using Neo4j with vector similarity search."""

    def __init__(
        self,
        driver: AsyncDriver,
        embedding_service,
        database: str = "convmemory",
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """Initialize Neo4j memory.

        Args:
            driver: Async Neo4j driver instance
            embedding_service: EmbeddingService instance for text embeddings
            database: Neo4j database name (default: convmemory)
            top_k: Maximum number of similar memories to retrieve (default: 5)
            similarity_threshold: Minimum similarity score (default: 0.7)
        """
        self.driver = driver
        self.embedding_service = embedding_service
        self.database = database
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    async def create_index_if_not_exists(self) -> None:
        """Create vector index on Memory nodes if it doesn't exist."""
        cypher = """
        CREATE VECTOR INDEX memory_embedding IF NOT EXISTS
        FOR (m:Memory) ON m.embedding
        OPTIONS {indexConfig: {
            `vector.dimensions`: $dimensions,
            `vector.similarity_function`: 'cosine'
        }}
        """

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(cypher, dimensions=self.embedding_service.dimension)
            logger.info(f"Vector index created/verified for database {self.database}")
        except Exception as e:
            logger.error(f"Failed to create vector index: {e}")
            raise

    async def store(self, session_id: str, role: str, content: str) -> None:
        """Store a message in Neo4j with embedding.

        Args:
            session_id: Session identifier
            role: Message role (human or ai)
            content: Message content
        """
        self._validate_role(role)

        # Generate embedding
        try:
            embedding = self.embedding_service.encode(content)
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

        memory_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        cypher = """
        // Ensure Session node exists
        MERGE (s:Session {id: $session_id})
        ON CREATE SET s.created_at = $timestamp

        // Create Memory node
        CREATE (m:Memory {
            id: $memory_id,
            session_id: $session_id,
            role: $role,
            content: $content,
            timestamp: $timestamp,
            embedding: $embedding
        })

        // Link Memory to Session
        CREATE (s)-[:HAS_MEMORY]->(m)

        RETURN m.id as memory_id
        """

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    cypher,
                    session_id=session_id,
                    memory_id=memory_id,
                    role=role,
                    content=content,
                    timestamp=timestamp,
                    embedding=embedding
                )
                record = await result.single()
                logger.debug(f"Stored {role} message in Neo4j: {record['memory_id']}")
        except Exception as e:
            logger.warning(f"Failed to store message in Neo4j: {e}")
            # Non-fatal - raise to let MemoryManager handle

    async def retrieve(self, session_id: str, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve semantically similar messages from Neo4j.

        Args:
            session_id: Session identifier
            query: Query text for semantic search

        Returns:
            List of message dictionaries with role, content, timestamp, similarity
        """
        if query is None or query.strip() == "":
            return await self._retrieve_recent(session_id)

        # Generate query embedding
        try:
            query_embedding = self.embedding_service.encode(query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")
            return []

        cypher = """
        CALL db.index.vector.queryNodes('memory_embedding', $top_k, $query_embedding)
        YIELD node, score
        WHERE score >= $similarity_threshold
            AND node.session_id = $session_id
        RETURN
            node.role as role,
            node.content as content,
            node.timestamp as timestamp,
            score as similarity
        ORDER BY score DESC
        """

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    cypher,
                    top_k=self.top_k,
                    query_embedding=query_embedding,
                    similarity_threshold=self.similarity_threshold,
                    session_id=session_id
                )

                memories = []
                async for record in result:
                    memories.append({
                        "role": record["role"],
                        "content": record["content"],
                        "timestamp": record["timestamp"],
                        "similarity": record["similarity"]
                    })

                logger.debug(f"Retrieved {len(memories)} semantic memories for session {session_id}")
                return memories

        except Exception as e:
            logger.warning(f"Failed to retrieve from Neo4j: {e}")
            return []

    async def _retrieve_recent(self, session_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve most recent memories when no query is provided.

        Args:
            session_id: Session identifier
            limit: Maximum number of memories to retrieve

        Returns:
            List of message dictionaries
        """
        cypher = """
        MATCH (s:Session {id: $session_id})-[:HAS_MEMORY]->(m:Memory)
        RETURN
            m.role as role,
            m.content as content,
            m.timestamp as timestamp
        ORDER BY m.timestamp DESC
        LIMIT $limit
        """

        try:
            async with self.driver.session(database=self.database) as session:
                result = await session.run(
                    cypher,
                    session_id=session_id,
                    limit=limit
                )

                memories = []
                async for record in result:
                    memories.append({
                        "role": record["role"],
                        "content": record["content"],
                        "timestamp": record["timestamp"]
                    })

                return memories

        except Exception as e:
            logger.warning(f"Failed to retrieve recent memories from Neo4j: {e}")
            return []

    async def clear(self, session_id: str) -> None:
        """Clear all memories for a session.

        Args:
            session_id: Session identifier
        """
        cypher = """
        MATCH (s:Session {id: $session_id})-[:HAS_MEMORY]->(m:Memory)
        DETACH DELETE m
        """

        try:
            async with self.driver.session(database=self.database) as session:
                await session.run(cypher, session_id=session_id)
            logger.info(f"Cleared Neo4j memory for session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to clear Neo4j memory: {e}")