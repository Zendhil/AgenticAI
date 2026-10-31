# built drepository to store and retrieve
# understand graph structures and queries
# standard libraries
from typing import List, Dict, Any
import logging
# app libraies
from agentic_ai.common import get_graph_driver, get_embedding_model

logger = logging.getLogger(__name__)

class RepositoryGraph:

    def __init__(self, driver, embedding_model):
        self._driver = driver or get_graph_driver()
        self._embedding_model = embedding_model or get_embedding_model()
        self._label = "Conversation"
        self._index_name = "conversation_embedding_index"

    async def _store_conversations(
            self,
            conversations: List[Dict[str, Any]]
    ) -> None:

        if not conversations:
            return

        # Generate embeddings outside the database transaction
        texts = [item["content"] for item in conversations]

        embeddings = self._embedding_model.encode(texts)

        records = []

        for item, embedding in zip(conversations, embeddings):
            records.append({
                **item,
                "embedding": embedding
            })

        query = """
            UNWIND $conversations AS item
            MERGE (c:Conversation {message_id: item.message_id})
            SET c.user_id = item.user_id,
                c.agent_id = item.agent_id,
                c.session_id = item.session_id,
                c.role = item.role,
                c.content = item.content,
                c.timestamp = item.timestamp,
                c.embedding = item.embedding
        """

        async with self._driver.session() as session:
            await session.run(query, conversations=records)

    async def search_conversations(
            self,
            user_id: str,
            agent_id: str,
            query_text: str,
            top_k: int = 5
    ) -> List[Dict[str, Any]]:

        query_embedding = self._embedding_model.encode(query_text)

        query = """
            CALL db.index.vector.queryNodes(
                $index_name, $top_k, $embedding
            )
            YIELD node, score

            WHERE node.user_id = $user_id
              AND node.agent_id = $agent_id

            RETURN node.content AS content,
                   node.role AS role,
                   node.session_id AS session_id,
                   node.timestamp AS timestamp,
                   score

            ORDER BY score DESC
        """

        async with self._driver.session() as session:
            result = await session.run(
                query,
                index_name=self._index_name,
                top_k=top_k,
                embedding=query_embedding,
                user_id=user_id,
                agent_id=agent_id
            )

            # Convert result to list for compatibility
            records = await result.list()
            return [dict(record) for record in records]

if __name__ == '__main__':
    print("Inside repository graph.py")
    import asyncio
    rep_graph = RepositoryGraph(driver=None, embedding_model=None)
    print(asyncio.run(rep_graph.store_conversations([
        {
            "message_id": "test_msg_iso_b",
            "user_id": "test_user_a",
            "agent_id": "test_agent_a",
            "session_id": "test_session_iso_a",
            "role": "human",
            "content": "User B is related to User A",
            "timestamp": "2026-10-31T10:00:00",
            "embedding": None
        }
    ])))

