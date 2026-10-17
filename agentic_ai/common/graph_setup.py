from neo4j import AsyncGraphDatabase, GraphDatabase
import logging
import asyncio
from agentic_ai.common import get_prd_grf_db_details

logger = logging.getLogger(__name__)
cypher = """
CREATE (l:Learning {
    topic: $topic,
    description: $description
})
RETURN l
"""
class GraphDB:
    def __init__(self):
        self._graph_db_dtl = get_prd_grf_db_details()
        self._driver = self._get_graph_driver()

    def _get_graph_driver(self):
        uri = self._graph_db_dtl.get("uri")
        user = self._graph_db_dtl.get("user")
        password = self._graph_db_dtl.get("pwd")
        try:
            driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            logger.error(e)
            raise e
        return driver

    async def connect(self):
        await self._driver.verify_connectivity()

    async def close(self):
        await self._driver.close()

    async def query(self, cypher: str, **params):
        records, _, _ = await self._driver.execute_query(cypher, **params)
        return [r.data() for r in records]

    def get_graph_driver(self):
        if self._driver is None:
            self._driver = self._get_graph_driver()
        return self._driver

graph_db = GraphDB()
get_graph_driver = graph_db.get_graph_driver

graph_db_connect = graph_db.connect
async def main():
    await graph_db_connect()
    result = await graph_db.query(
        cypher,
        topic="Neo4j",
        description="Learning graph databases"
    )
    print(result)

if __name__ == '__main__':
    asyncio.run(main())


