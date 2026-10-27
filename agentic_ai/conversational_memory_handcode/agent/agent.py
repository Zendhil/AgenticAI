#standard libarary imports
import uuid
import logging
from pyexpat.errors import messages
from typing import Any
import asyncio
#Thrid party imports
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage
# App imports
from agentic_ai.common import get_llm_model, get_embedding_model
from agentic_ai.conversational_memory_handcode.memory import MemoryManager, SessionManager, SessionKey, Session
from agentic_ai.conversational_memory_handcode.agent.prompt import PROMPT

logger = logging.getLogger(__name__)


class ChatAgent:

    prompt = PROMPT
    # default_llm = get_llm_model()
    embedding_model = get_embedding_model()

    def __init__(self,agent_id:str, user_id:str, memory_manager: MemoryManager, llm:Any):
        self._agent_id= agent_id or uuid.uuid4()
        self._user_id = user_id
        self._llm = llm or get_llm_model()
        self._memory_manager = memory_manager or MemoryManager() # check this... if its empty then initialization should happen

    async def chat(self, commit_threshold = 6):

        messages = []
        response = []
        input_id = input("Enter your user_id:")
        # retrieve context for User_id
        # build SessionManager
        session_key = SessionKey(session_id=None,user_id=input_id, agent_id="1234")
        session_manager = SessionManager(session_key=session_key)
        memory_manager = MemoryManager(session_manager=session_manager)
        # check if context exists. Retrieve
        # check in Long term memory
        # try:
        #     messages = await session_manager.retrieve_info()
        # except Exception as e:
        #     logger.error(f"In ChatAgent: chat: Error {e}")
        #

        while True:
            input_message = input("User:")
            messages.append(HumanMessage(content = input_message))
            if input_message != "exit":
                try:
                    response = await self._llm.ainvoke(messages)
                    print(f"AI Response:{response.content}")
                    messages.append(AIMessage(content=response.content))
                    if len(messages) == commit_threshold:
                        #call session store to commit.
                        # continue from here... build session messages. messages has to unwrap human and AI.. so it needs a method for wrapping and unwrapping.
                        # check messages dataclass.. how session, sessionkey, messages and Human and AI messaegs are orgainsed. thenc commit.
                        # then write the retrieve from long DB.
                        await session_manager.save_session(session_key)
                except Exception as e:
                    raise e
            else:
                break
        print("Messages:",messages)



if __name__ == "__main__":
    print("Inside agent.py")
    agent_id = "test_agent"
    user_id = "test_user"
    chat_ag = ChatAgent(agent_id = agent_id, user_id = user_id, memory_manager = MemoryManager(),llm=None)
    asyncio.run(chat_ag.chat())
    # print(ChatAgent.embedding_model.encode("Whats your name"))