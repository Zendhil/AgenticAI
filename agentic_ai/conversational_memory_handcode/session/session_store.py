#standard library imports
from abc import ABC,abstractmethod
import json
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import logging
#Third party library
import redis.asyncio as redis

logger = logging.getLogger(__name__)




# 1. Define the contract
class AbstractDatabaseClient(ABC):
    """
    Abstract base class defining the contract for database client implementations.

    This interface ensures all database clients implement save and retrieve operations.
    """

    @abstractmethod
    async def save_session(self, key:str, value:str) -> None:
        """
        Saves a session value to the database.

        Args:
            key: The key under which to store the value
            value: The value to store (typically JSON string)

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass

    @abstractmethod
    async def retrieve_session(self, key:str) -> dict | None:
        """
        Retrieves a session value from the database.

        Args:
            key: The key to retrieve
            value: Additional parameter for retrieval

        Returns:
            dict | None: The retrieved data as a dictionary, or None if not found

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        pass



# 2. Implement Redis
class RedisClient(AbstractDatabaseClient):
    """
    Redis implementation of the AbstractDatabaseClient.

    Provides async Redis operations for session storage and retrieval.
    """

    def __init__(self, host="localhost", port=6379, db=0):
        """
        Initializes the Redis client with connection parameters.

        Args:
            host: Redis server host (default: localhost)
            port: Redis server port (default: 6379)
            db: Redis database number (default: 0)
        """
        logger.info("Inside class RedisClient, function __init__")
        try:
            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True
            )
            logger.info(f"Redis client initialized with host={host}, port={port}, db={db}")
        except Exception as e:
            logger.error(f"Error initializing Redis client: {e}")
            raise

    async def save_session(self,  key_:str, value:str) -> None:
        """
        Saves a session value to Redis.

        Args:
            key_: The Redis key
            value: The value to store

        Raises:
            Exception: If Redis set operation fails
        """
        logger.info("Inside class RedisClient, function save_session")
        try:
            await self._client.rpush(key_, *value)
            logger.info(f"Successfully saved session with key: {key_}")
        except Exception as e:
            logger.error(f"Error saving session to Redis with key {key_}: {e}")
            raise

    async def retrieve_session(self, key_: str) -> list[dict] | None:
        """Retrieves messages from a Redis List."""
        logger.info("Inside RedisClient.retrieve_session")

        try:
            if key_ is None:
                raise ValueError("The Session key is None")

            data = await self._client.lrange(key_, 0, -1)

            if not data:
                return None

            result = [json.loads(message) for message in data]

            logger.info("Successfully retrieved session: %s", key_)
            return result

        except (ValueError, json.JSONDecodeError):
            logger.exception("Validation or JSON parsing error for key %s", key_)
            raise
        except Exception:
            logger.exception("Error retrieving session from Redis: %s", key_)
            raise

    async def connect(self) -> bool:
        """
        Tests the connection to Redis server.

        Returns:
            bool: True if connection is successful

        Raises:
            Exception: If ping operation fails
        """
        logger.info("Inside class RedisClient, function connect")
        try:
            result = await self._client.ping()
            logger.info("Successfully connected to Redis server")
            return result
        except Exception as e:
            logger.error(f"Error connecting to Redis server: {e}")
            raise

    async def close(self) -> bool:
        """
        Closes the Redis connection.

        Returns:
            bool: True if connection closed successfully

        Raises:
            Exception: If close operation fails
        """
        logger.info("Inside class RedisClient, function close")
        try:
            result = await self._client.aclose()
            logger.info("Successfully closed Redis connection")
            return result
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")
            raise


# 3. Create the factory
class SessionClientFactory:
    """
    Factory class for creating database client instances.

    Supports creating different types of database clients based on the specified type.
    """

    @staticmethod
    def create(client="redis", **kwargs) -> AbstractDatabaseClient:
        """
        Creates and returns a database client instance.

        Args:
            client: Type of client to create (default: "redis")
            **kwargs: Additional arguments passed to the client constructor

        Returns:
            AbstractDatabaseClient: An instance of the requested database client

        Raises:
            ValueError: If an unsupported client type is specified
        """
        logger.info("Inside class SessionClientFactory, function create")
        try:
            match client:
                case "redis":
                    logger.info(f"Creating RedisClient with kwargs: {kwargs}")
                    return RedisClient(**kwargs)
                case _:
                    raise ValueError(f"Unsupported database client: {client}")
        except ValueError as ve:
            logger.error(f"Validation error in SessionClientFactory: {ve}")
            raise
        except Exception as e:
            logger.error(f"Error creating database client: {e}")
            raise

cache_client = SessionClientFactory.create("redis")



if __name__=="__main__":
    print(asyncio.run(cache_client.close()))