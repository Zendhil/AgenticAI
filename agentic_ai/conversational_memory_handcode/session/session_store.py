#standard library imports
from abc import ABC,abstractmethod
import json
import asyncio
from abc import ABC, abstractmethod

#Third party library
import redis.asyncio as redis




# 1. Define the contract
class AbstractDatabaseClient(ABC):

    @abstractmethod
    async def save_session(self, session_id: str, data: dict) -> None:
        pass

    @abstractmethod
    async def retrieve_session(self, session_id: str) -> dict | None:
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

    async def save_session(self, session_id: str, data: dict) -> None:
        await self._client.set(session_id, json.dumps(data))

    async def retrieve_session(self, session_id: str) -> dict | None:
        data = await self._client.get(session_id)
        return json.loads(data) if data is not None else None

    async def connect(self) -> bool:
        return await self._client.ping()


# 3. Create the factory
class SessionClientFactory:

    @staticmethod
    def create(client="redis", **kwargs) -> AbstractDatabaseClient:
        match client:
            case "redis":
                return RedisClient(**kwargs)
            case _:
                raise ValueError(f"Unsupported database client: {client}")



if __name__=="__main__":
    client_= SessionClientFactory.create(client="redis")

    print(asyncio.run(client_.connect()))