#standard library imports
import logging
from abc import ABC, abstractmethod

#app imports
from agentic_ai.common import get_graph_driver, get_embedding_model
from agentic_ai.conversational_memory_handcode.db import RepositoryGraph
logger = logging.getLogger(__name__)
g_graph_driver = get_graph_driver()

class DBStore(ABC):

    def __init__(self):
        pass
    @abstractmethod
    async def retrieve(self):
        pass

    @abstractmethod
    async def store_conversations(self):
        pass

class DbStoreFactory:
    async def create(self, storage:str="neo4j"):
        if storage == "neo4j":
            return Neo4j(driver=None, embedding_model=None)


class Neo4j(DBStore):
    def __init__(self, driver=None, embedding_model=None):
        self._driver = driver or g_graph_driver
        self._embedding_model = embedding_model or get_embedding_model()
        self._graph_connector = RepositoryGraph(self._driver, embedding_model)

    async def store_conversations(self, messages):
        await self._graph_connector._store_conversations(messages)

    async def retrieve(self):
        pass






if __name__ == "__main__":
    import asyncio
    print("Inside db")
    db = DbStoreFactory()
    graph_db = asyncio.run(db.create("neo4j"))
    conversations = [
        {
            "message_id": "msg_empty",
            "user_id": "user_1",
            "agent_id": "agent_1",
            "session_id": "session_1",
            "role": "human",
            "content": "testing db_store",  # Empty content
            "timestamp": "2026-10-31T10:00:00"
        }
    ]
    asyncio.run(graph_db.store_conversations(conversations))

