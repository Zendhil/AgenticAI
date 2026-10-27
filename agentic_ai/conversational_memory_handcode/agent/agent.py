#standard libarary imports
import uuid
import logging
from pyexpat.errors import messages
from typing import Any
import asyncio
#Thrid party imports
from langchain.agents import create_agent

# App imports
from agentic_ai.common import get_llm_model, get_embedding_model
from agentic_ai.conversational_memory_handcode.memory import MemoryManager
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

    async def chat(self):

        messages = []
        user_message = ""
        response = []
            while (1):
                user_message = input("User:")
                messages.append(user_message)
                if user_message != "exit":
                    try:
                        response = await self._llm.ainvoke(messages)
                    except Exception as e:
                        raise e
                print(response)



if __name__ == "__main__":
    print("Inside agent.py")
    agent_id = "test_agent"
    user_id = "test_user"
    chat_ag = ChatAgent(agent_id = agent_id, user_id = user_id, memory_manager = MemoryManager(),llm=None)
    asyncio.run(chat_ag.chat())
    # print(ChatAgent.embedding_model.encode("Whats your name"))