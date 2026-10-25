#standard library imports
import datetime
from abc import ABC,abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
#app library imports
#from agentic_ai.conversational_Agent import SessionStore
from agentic_ai.conversational_memory_handcode.session import redis_client

@dataclass
class SessionKey:
    session_id: str
    user_id: str
    agent_id: str

    def to_redis_key(self) -> str:
        return f"session:{self.user_id}:{self.session_id}:{self.agent_id}"

@dataclass
class Message:
    role:str
    content: str
    timestamp: datetime

    def to_redis_value(self) -> str:
        return f"Timestamp:{self.timestamp.isoformat()}:role:{self.role}:content:{self.content}"

@dataclass
class Session:
    key_: SessionKey
    values_: list[Message] =field(default_factory=list)
    # def to_dict(self) -> dict:
    #     return asdict(self)

@dataclass
class ShortTermMemory:
    user_id: str
    user_info:str

@dataclass
class UserMemory:
    """Memory-enabled user session manager."""
    short_term:ShortTermMemory
    #To be defined: Long term memory

class MemoryManager:
    """Memory-enabled conversational agent."""

    def __init__(self,session_manager:SessionManager=None):
        self._session_manager = session_manager


    def _store(self, :SessionManager):
        pass


    def _retrieve(self,agent_id:str,user_id:str,session:SessionStore):
        pass

    def _compact(self,agent_id:str,user_id:str,session:SessionStore):

class SessionManager:
    """Memory-enabled session manager."""
    def __init__(self,cls,agent_id:str,user_id:str,session_info:Session):
        self._session_info = session_info

    async def _save_session(self, session_info:Session):
        key_=session_info.key_.to_redis_key()
        values_=session_info.values_
        for value_ in values_:
            await redis_client.save_session(key_, value_.to_redis_value)

    async def _retrieve_info(self,memory_info):
        pass

