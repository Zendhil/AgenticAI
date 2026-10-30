#standard libarary imports
import uuid
import logging
from pyexpat.errors import messages
from typing import Any, List
import asyncio
import datetime
#Thrid party imports
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
# App imports
from agentic_ai.common import get_llm_model, get_embedding_model
from agentic_ai.conversational_memory_handcode.memory import MemoryManager, SessionManager, SessionKey, Session, Message
from agentic_ai.conversational_memory_handcode.agent.prompt import PROMPT

logger = logging.getLogger(__name__)


class ChatAgent:

    prompt = PROMPT
    # default_llm = get_llm_model()
    embedding_model = get_embedding_model()

    def __init__(self,user_id:str, llm:Any,agent_id:str="Agent1234"):
        self._agent_id= agent_id or uuid.uuid4()
        if user_id is None:
            user_id = input("Enter your user_id:")
        self._user_id = user_id
        self._llm = llm or get_llm_model()
        self._message_count = 0
        self._session_key = SessionKey(session_id=None, user_id=user_id, agent_id=agent_id)
        logger.info(f"Session Key:{self._session_key}")
        self._session_manager = SessionManager(session_key=self._session_key)
        self._memory_manager = MemoryManager(session_manager=self._session_manager)


    async def chat(self, commit_threshold = 6):

        messages = []
        response = []
        # input_id = input("Enter your user_id:")
        # retrieve context for User_id
        # build SessionManager
        # check if context exists. Retrieve
        # check in Long term memory
        # try:
        #     messages = await session_manager.retrieve_info()
        # except Exception as e:
        #     logger.error(f"In ChatAgent: chat: Error {e}")
        #
        saved_messages = 0
        while True:
            input_message = input("User:")

            if input_message != "exit":
                messages.append(HumanMessage(content=input_message))
                self._message_count += 1
                try:
                    response = await self._llm.ainvoke(messages)
                    print(f"AI Response:{response.content}")
                    messages.append(AIMessage(content=response.content))
                    self._message_count += 1
                except Exception as e:
                    raise e
            else:
                # on exit, save all the pending messages
                if len(messages):
                    logger.info("Saving messages on exit")
                    session_info = await self.store_messages(messages[saved_messages:self._message_count], timestamp=datetime.datetime.now())
                    await self._session_manager.save_session(session_info)
                break

            if len(messages) == commit_threshold:
                # call session store to commit. - Done
                # continue from here... build session messages. messages has to unwrap human and AI.. so it needs a method for wrapping and unwrapping.
                # check messages dataclass.. how session, sessionkey, messages and Human and AI messaegs are orgainsed. thenc commit.
                # then write the retrieve from long DB.
                logger.info("Saving messages on threshold")

                session_info = await self.store_messages(messages[saved_messages:self._message_count], timestamp=datetime.datetime.now())
                await self._session_manager.save_session(session_info)
                #flush
                saved_messages = self._message_count

        print("Messages:",messages)

    async def store_messages(self,messages: List, timestamp:datetime.datetime)->Session:
        logger.info(f"In class ChatAgent, function: store_messages")
        unwrap_messages = []
        if timestamp is None:
            timestamp = datetime.datetime.now()
        for message in messages:
            role = None
            content = None
            logger.info(f"Type of message:{message}")

            if isinstance(message, AIMessage):
                role = self._agent_id
            elif isinstance(message, HumanMessage):
                role = self._user_id
            else:
                role = "Other"
            content = message.content
            unwrap_messages.append(Message(role=role, content=content, timestamp=timestamp))
        session_info = Session(key_=self._session_key, values_=unwrap_messages)
        return session_info

if __name__ == "__main__":
    print("Inside agent.py")
    agent_id = "Agent1234"
    user_id = "CBCID"
    chat_ag = ChatAgent(user_id = user_id,llm=None,agent_id = agent_id)
    asyncio.run(chat_ag.chat())
    # print(ChatAgent.embedding_model.encode("Whats your name"))