#standard library imports
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DBStore(ABC):

    def __init__(self):
        pass
    @abstractmethod
    def _retrieve(self):
        pass

    @abstractmethod
    def _store(self):
        pass

class DbStoreFactory:
    def create(self, storage:str="neo4j"):
        if storage == "neo4j":
            return Neo4j()


class Neo4j(DBStore):
    def __init__(self):
        pass

    def _store(self):
        pass

    def _retrieve(self):
        pass






if __name__ == "__main__":
    print("Inside db")