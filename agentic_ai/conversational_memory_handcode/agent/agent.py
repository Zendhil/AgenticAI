#standard libarary imports
import uuid
import logging
from typing import Any
#Thrid party imports
from langchain.agents import create_agent

# App imports
from agentic_ai.common import get_llm_model, EmbeddingService
from agentic_ai.conversational_memory_handcode.agent.prompt import PROMPT
logger = logging.getLogger(__name__)


class ChatAgent:

    prompt = PROMPT
    default_llm = get_llm_model()
    embedding_model = EmbeddingService().get_embedding_model()
    def __init__(self,cls,agent_id:str, user_id:str, session_store, llm:Any):
        self._agent_id= agent_id or uuid.uuid4()
        self._user_id = user_id
        self._session_store = session_store
        self._llm = llm or cls.default_llm
        self._agent = create_agent( model=self._llm,
                                    system_prompt=PROMPT
                                    )



if __name__ == "__main__":
    print("Inside agent.py")
    print(ChatAgent.embedding_model.encode("Whats your name"))