#standard library imports
import logging

from pyparsing import opAssoc

from agentic_ai.conversational_memory_handcode.db.db_store import DBStore, DbStoreFactory
#App specific imports
from agentic_ai.conversational_memory_handcode.memory import Message

logger = logging.getLogger(__name__)

class LongTermMemoryManager:

    def __init__(self, db_obj):
        self._db_obj = db_obj

    @classmethod
    async def create(cls, db_store="Neo4j"):

        if db_store == "Neo4j":
            db_obj = await DbStoreFactory.create(db_store)

        else:
            raise ValueError(f"Unsupported DB store: {db_store}")

        return cls(db_obj=db_obj)


    async def retrieve_info(self, user_id = None, mode="query")->list[Message]:
        """
        Retrieves user relevant information from database.
        params: user_id for which info has to be retrieved
                mode is "query" or "summary"
        Returns:
            Conversations: Retrieved user Conversations

        """
        logger.info("Inside class LongTermMemoryManager, function _retrieve_info")

        if user_id == None:
            raise ValueError("user_id is required")

        if mode == "query":
            try:
                messages = await self._fetch_relevant_info()
                return messages
            except Exception as e:
                logger.error(f"Error fetching relevant info from GraphDB: {e}")
                raise
        elif mode == "summary":
            try:
                messages = await self._fetch_all()
                summary = await self._summarise(messages)
                return summary
            except Exception as e:
                logger.error(f"Error summarising user info from GraphDB: {e}")
                raise

    async def _fetch_relevant_info(self)->list[Message]:
        pass

    async def _fetch_all(self, **kwargs)->list[Message]:
        """
                Retrieves user relevant information from database.
                params: user_id for which info has to be retrieved
                        mode is "query" or "summary"
                Returns:
                    Conversations: Retrieved user Conversations

        """
        pass
        # logger.info("Inside class LongTermMemoryManager, function _fetch_all")
        # for key in kwargs:
        #     if key == "user_id":


    async def _summarise(self, messages)->list[Message]:
        pass

    async def save_info(self)->Message:
        pass

if __name__ == "__main__":
    import asyncio
    ltm = asyncio.run(LongTermMemoryManager.create(db_store="Neo4j"))
    print(ltm)
