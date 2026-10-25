#standard library imports
from abc import ABC,abstractmethod
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
#Third party library
import redis.asyncio as redis




# 1. Define the contract
class AbstractDatabaseClient(ABC):

    @abstractmethod
    async def save_session(self, key:str, value:str) -> None:
        pass

    @abstractmethod
    async def retrieve_session(self, key:str, value:str) -> dict | None:
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

    async def save_session(self,  key_:str, value:str) -> None:

        await self._client.set(key_, value)

    async def retrieve_session(self, key_: str) -> dict | None:
        if key_ is None:
            raise ValueError("The value provided for Session key is None")
        data = await self._client.get(key_)
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