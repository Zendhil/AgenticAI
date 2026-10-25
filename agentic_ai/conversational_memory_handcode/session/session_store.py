#standard library imports
from abc import ABC,abstractmethod
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
#Third party library
import redis.asyncio as redis

@dataclass
class SessionKey:
    session_id: str
    user_id: str
    agent_id: str

    def to_redis_key(self) -> str:
        return f"session:{self.user_id}:{self.session_id}:{self.agent_id}"
@dataclass
class Session:
    key_: SessionKey
    message: str
    def to_dict(self) -> dict:
        return asdict(self)


# 1. Define the contract
class AbstractDatabaseClient(ABC):

    @abstractmethod
    async def save_session(self, session_info:Session) -> None:
        pass

    @abstractmethod
    async def retrieve_session(self, session_info:Session) -> dict | None:
        pass



# 2. Implement Redis
class RedisClient(AbstractDatabaseClient):

    def __init__(self, host="localhost", port=6379, db=0):
        self._client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    async def save_session(self,  session_info: Session) -> None:
        redis_key = session_info.key_.to_redis_key()
        session_data = session_info.to_dict()

        await self._client.set(redis_key, session_data.get("message"))

    async def retrieve_session(self, session_id: str) -> dict | None:
        data = await self._client.get(session_id)
        return json.loads(data) if data is not None else None

    async def connect(self) -> bool:
        return await self._client.ping()

    async def close(self) -> bool:
        return await self._client.aclose()


# 3. Create the factory
class SessionClientFactory:

    @staticmethod
    def create(client="redis", **kwargs) -> AbstractDatabaseClient:
        match client:
            case "redis":
                return RedisClient(**kwargs)
            case _:
                raise ValueError(f"Unsupported database client: {client}")

redis_client = SessionClientFactory.create("redis")



if __name__=="__main__":
    print(asyncio.run(redis_client.close()))