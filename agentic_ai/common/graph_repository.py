##### Standard library imports #####
from typing import Any, Dict, List, Optional
import logging

##### Third party imports #####
from neo4j import AsyncSession

##### App imports #####


logger = logging.getLogger(__name__)


class GraphRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_chunks_for_entity(self, entity_name: str) -> List[Dict[str, Any]]:
        """Get all text chunks that mention a specific entity."""
        query = """
            MATCH (e {name: $entity_name})-[:EXTRACTED_FROM]->(c:Chunk)
            RETURN c.id as id, c.text as text, c.source as source,
                   c.page as page, c.index as index
            ORDER BY c.index
        """
        result = await self.session.run(query, entity_name=entity_name)
        records = [dict(record) async for record in result]
        return records

    async def get_entities_in_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Get all entities extracted from a specific chunk."""
        query = """
            MATCH (e)-[:EXTRACTED_FROM]->(c:Chunk {id: $chunk_id})
            RETURN e.name as name, labels(e)[0] as label
        """
        result = await self.session.run(query, chunk_id=chunk_id)
        records = [dict(record) async for record in result]
        return records

    async def get_entity_with_context(self, entity_name: str) -> Dict[str, Any]:
        """Get an entity with all its relationships and source chunks."""
        query = """
            MATCH (e {name: $entity_name})
            OPTIONAL MATCH (e)-[r]->(related)
            OPTIONAL MATCH (e)-[:EXTRACTED_FROM]->(c:Chunk)
            RETURN e.name as entity,
                   labels(e)[0] as label,
                   collect(DISTINCT {type: type(r), target: related.name}) as relationships,
                   collect(DISTINCT {id: c.id, text: c.text, source: c.source, page: c.page}) as chunks
        """
        result = await self.session.run(query, entity_name=entity_name)
        record = await result.single()
        return dict(record) if record else {}

    async def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk with all its properties."""
        query = """
            MATCH (c:Chunk {id: $chunk_id})
            RETURN c.id as id, c.text as text, c.source as source,
                   c.page as page, c.section as section, c.index as index
        """
        result = await self.session.run(query, chunk_id=chunk_id)
        record = await result.single()
        return dict(record) if record else None

    async def insert_graph(self, graph_json: dict):
        nodes = graph_json.get("nodes", {})
        relations = graph_json.get("relations", {})
        chunks = graph_json.get("chunks", [])
        entity_chunk_links = graph_json.get("entity_chunk_links", [])

        # 1. Insert text chunks first
        for chunk_data in chunks:
            await self._create_chunk(chunk_data)

        # 2. Insert entity nodes
        for node_key, node_data in nodes.items():
            await self._create_node(node_data, node_key)

        # 3. Insert relations between entities
        for rel_key, rel_data in relations.items():
            await self._create_relation(rel_data)

        # 4. Link entities to chunks they were extracted from
        for link_data in entity_chunk_links:
            await self._link_entity_to_chunk(link_data)

    async def _create_node(self, node_data, fallback_name):
        """Create an entity node."""
        label = node_data.get("label", "Entity")
        name = node_data.get("name", fallback_name)
        props = node_data.get("properties", {})
        source = props.get("source", "")

        query = (
            f"MERGE (n:{label} {{name: $name}}) "
            "SET n.source = $source"
        )
        await self.session.run(query, name=name, source=source)

    async def _create_relation(self, rel_data):
        """Create a relationship between two entities."""
        rel_type = rel_data.get("label", "RELATED_TO").replace(" ", "_")
        source_name = rel_data["source_id"]
        target_name = rel_data["target_id"]
        source = rel_data.get("properties", {}).get("source", "")

        query = (
            "MATCH (a {name: $source_name}) "
            "MATCH (b {name: $target_name}) "
            f"MERGE (a)-[r:`{rel_type}`]->(b) "
            "SET r.source = $source"
        )
        await self.session.run(query, source_name=source_name, target_name=target_name, source=source)

    async def _create_chunk(self, chunk_data):
        """Create a text chunk node with its content and metadata."""
        chunk_id = str(chunk_data.get("id"))
        text = chunk_data.get("text", "")
        metadata = chunk_data.get("metadata", {})

        # Build properties dynamically from metadata
        set_clauses = ["c.text = $text"]
        params = {"chunk_id": chunk_id, "text": text}

        # Add common metadata fields
        for field in ["source", "page", "section", "doc_id", "index"]:
            if field in metadata:
                set_clauses.append(f"c.{field} = ${field}")
                params[field] = metadata[field]

        query = (
            "MERGE (c:Chunk {id: $chunk_id}) "
            f"SET {', '.join(set_clauses)}"
        )
        await self.session.run(query, **params)

    async def _link_entity_to_chunk(self, link_data):
        """Link an entity to the chunk it was extracted from."""
        entity_name = link_data.get("entity_name")
        chunk_id = str(link_data.get("chunk_id"))
        rel_type = link_data.get("relationship", "EXTRACTED_FROM")

        query = (
            "MATCH (e {name: $entity_name}) "
            "MATCH (c:Chunk {id: $chunk_id}) "
            f"MERGE (e)-[r:{rel_type}]->(c)"
        )
        await self.session.run(query, entity_name=entity_name, chunk_id=chunk_id)


if __name__ == '__main__':
    print("Inside repositories_graph...")

    from agentic_ai.common import get_prd_grf_driver
    driver = get_prd_grf_driver()
    with driver.session(database='learning') as session:
        repo = ProductRepository(session)
        repo.insert_graph()

        # print("One Movie:", repo.get_movie_by_title("The Matrix"))
