#standard library imports
import logging
from abc import ABC, abstractmethod

#app imports
from agentic_ai.common import get_graph_driver

logger = logging.getLogger(__name__)
g_graph_driver = get_graph_driver()

class DBStore(ABC):

    def __init__(self):
        pass
    @abstractmethod
    async def _retrieve(self):
        pass

    @abstractmethod
    async def _store(self):
        pass

class DbStoreFactory:
    def create(self, storage:str="neo4j"):
        if storage == "neo4j":
            return Neo4j()


class Neo4j(DBStore):
    def __init__(self, driver=None):
        self._driver = driver or g_graph_driver

    async def _store(self):
        graph_repo.store()

    async def _retrieve(self):
        pass






if __name__ == "__main__":
    print("Inside db")