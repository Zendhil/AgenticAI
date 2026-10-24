#standard libarary imports
import uuid
import logging
from typing import Any

# App imports
from agentic_ai.common import get_llm_model
from agentic_ai.conversational_memory_handcode.agent.prompt import PROMPT
logger = logging.getLogger(__name__)


class ChatAgent:

    prompt = PROMPT
    default_llm = get_llm_model()
    def __init__(self,cls,agent_id:str, user_id:str, session_store, llm:Any):
        self._agent_id= agent_id or uuid.uuid4()
        self._user_id = user_id
        self._session_store = session_store
        self._llm = llm or cls.default_llm



if __name__ == "__main__":
    print("Inside agent.py")