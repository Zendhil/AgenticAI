#standard library imports
import logging

from pyparsing import opAssoc

from agentic_ai.conversational_memory_handcode.db.db_store import DBStore, DbStoreFactory
#App specific imports
from agentic_ai.conversational_memory_handcode.memory import Message
from agentic_ai.common import get_llm_model
from agentic_ai.conversational_memory_handcode.memory.prompt import SUMMARISE_PROMPT

logger = logging.getLogger(__name__)
g_llm_model = get_llm_model()

class LongTermMemoryManager:

    def __init__(self, db_obj, llm=None):
        self._db_obj = db_obj

        if llm == None:
            self._llm_model = g_llm_model

    @classmethod
    async def create(cls, db_store="Neo4j"):

        if db_store == "Neo4j":
            db_obj = await DbStoreFactory.create(db_store)

        else:
            raise ValueError(f"Unsupported DB store: {db_store}")

        return cls(db_obj=db_obj)
        # return db_obj


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

    async def fetch_all(self, **kwargs)->list[Message]:
        """
                Retrieves user relevant information from database.
                params: user_id for which info has to be retrieved
                        mode is "query" or "summary"
                Returns:
                    Conversations: Retrieved user Conversations

        """

        logger.info("Inside class LongTermMemoryManager, function _fetch_all")
        try:
            messages = await self._db_obj.fetch_all(kwargs)
            messages_list = []
            for message in messages:
                role = message.get('user_id')
                content = message.get('content')
                timestamp = message.get('timestamp')
                messages_list.append(Message(role, content, timestamp))
            return messages_list
        except Exception as e:
            logger.error(f"Error fetching relevant info from GraphDB: {e}")
            raise




    async def _summarise(self, messages)->str:
        """
        Summarises a list of messages using the LLM.

        Args:
            messages: List of Message objects with role, content, timestamp

        Returns:
            str: Concise summary focusing on user-relevant information
        """
        logger.info("Inside class LongTermMemoryManager, function _summarise")

        # Unpack Message objects into formatted conversation
        formatted_messages = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in messages
        ])

        # Combine prompt with conversation
        full_prompt = f"{SUMMARISE_PROMPT}\n\nConversation:\n{formatted_messages}"

        # Invoke LLM and extract content
        response = self._llm_model.invoke(full_prompt)
        summary = response.content if hasattr(response, 'content') else str(response)

        return summary

    async def save_info(self)->Message:
        pass

if __name__ == "__main__":
    import asyncio
    from datetime import datetime

    # ltm = LongTermMemoryManager()
    ltm = asyncio.run(LongTermMemoryManager.create(db_store="Neo4j"))
    # print(ltm)
    # asyncio.run(ltm.fetch_all(**{"user_id": "test_user_a"}))

    messages = [
        Message("user", "Hi, I am Senthil.", datetime.now()),
        Message("assistant", "Hello Senthil! How can I help you?", datetime.now()),
        Message("user", "I am building a conversational AI agent.", datetime.now()),
        Message("assistant", "Are you using Neo4j for long-term memory?", datetime.now()),
        Message("user", "Yes, I am using Neo4j and Redis.", datetime.now())
    ]
    print(asyncio.run(ltm._summarise(messages)))