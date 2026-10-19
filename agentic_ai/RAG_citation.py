"""
RAG (Retrieval Augmented Generation) with Graph Database

This module provides functionality to:
1. Split documents into chunks (paragraphs/text segments)
2. Extract entities and relationships from text
3. Store chunks, entities, and relationships in Neo4j
4. Link entities to the chunks they were extracted from
5. Query the graph to retrieve context for entities

Graph Structure:
- Chunk nodes: Store the actual text/paragraphs with metadata
- Entity nodes: Extracted entities (Person, Company, Location, etc.)
- Relationship edges: Connections between entities (FOUNDED, CEO_OF, etc.)
- EXTRACTED_FROM edges: Link entities to their source chunks

Usage:
    rag = ProcessRAG(database='learning')
    rag.process_and_store_document(text, entities, relationships, metadata)
    context = rag.get_entity_context("Entity Name")
"""

# Standard library imports
import logging
from pathlib import Path
from typing import List, Dict, Any, Sequence, Optional
from uuid import uuid4
import asyncio

from langsmith.run_helpers import is_async
# from config import NEO4J, INDEX, embeddings

# Third party imports
from sentence_transformers import SentenceTransformer
from llama_index.core import Document, SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import MetadataMode

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.indices.property_graph.transformations import SimpleLLMPathExtractor
from llama_index.core.graph_stores.types import ChunkNode, EntityNode, Relation

from llama_index.core.ingestion.pipeline import arun_transformations, IngestionPipeline
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_neo4j import Neo4jVector

# App imports
from agentic_ai.common import get_config, get_embeddings_model, get_device, get_llm_model
from agentic_ai.common import GraphRepository, get_graph_driver
from agentic_ai.common.business_objects import (
    EntityContext, Chunk
)

g_device = get_device()
g_embeddings_model = get_embeddings_model()
g_embedding_model = SentenceTransformer(model_name_or_path=g_embeddings_model, device=g_device, similarity_fn_name='cosine')
g_driver = get_graph_driver()
g_llm_model = get_llm_model()

logger = logging.getLogger(__name__)


async def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks.

    Args:
        text: The full text to split
        chunk_size: Target size for each chunk
        overlap: Number of characters to overlap between chunks
    """
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk_text.rfind('.')
            if last_period > chunk_size * 0.5:  # Only if we're past halfway
                end = start + last_period + 1
                chunk_text = text[start:end]

        chunks.append({
            "id": uuid4(),
            "text": chunk_text.strip(),
            "index": index,
            "start_char": start,
            "end_char": end
        })

        start = end - overlap
        index += 1

    return chunks


async def build_rag_graph(document_text: str,
                          entities: List[Dict[str, Any]],
                          relationships: List[Dict[str, Any]],
                          doc_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a complete graph structure for RAG.

    Args:
        document_text: The full document text
        entities: List of extracted entities with their positions
        relationships: List of relationships between entities
        doc_metadata: Metadata about the document (source, author, etc.)

    Example entities format:
    [
        {"name": "Apple Inc.", "label": "Company", "start": 0, "end": 10},
        {"name": "Steve Jobs", "label": "Person", "start": 50, "end": 61}
    ]
    """
    # Split document into chunks
    chunks = await split_into_chunks(document_text)

    # Add metadata to chunks
    for chunk in chunks:
        chunk["metadata"] = {
            "source": doc_metadata.get("source", ""),
            "doc_id": doc_metadata.get("doc_id", ""),
            "page": doc_metadata.get("page"),
            "section": doc_metadata.get("section", "")
        }

    # Build nodes from entities
    nodes = {}
    for entity in entities:
        key = entity["name"].lower().replace(" ", "_")
        nodes[key] = {
            "label": entity["label"],
            "name": entity["name"],
            "properties": {
                "source": doc_metadata.get("source", "")
            }
        }

    # Build relations
    relations = {}
    for idx, rel in enumerate(relationships):
        relations[f"rel_{idx}"] = {
            "source_id": rel["source"],
            "target_id": rel["target"],
            "label": rel["type"],
            "properties": {
                "source": doc_metadata.get("source", "")
            }
        }

    # Link entities to chunks (find which chunk contains each entity)
    entity_chunk_links = []
    for entity in entities:
        entity_start = entity.get("start", 0)
        entity_end = entity.get("end", 0)

        # Find all chunks that contain this entity
        for chunk in chunks:
            chunk_start = chunk["start_char"]
            chunk_end = chunk["end_char"]

            # Check if entity is in this chunk
            if (entity_start >= chunk_start and entity_start < chunk_end) or \
               (entity_end > chunk_start and entity_end <= chunk_end):
                entity_chunk_links.append({
                    "entity_name": entity["name"],
                    "chunk_id": chunk["id"],
                    "relationship": "EXTRACTED_FROM"
                })

    return {
        "chunks": chunks,
        "nodes": nodes,
        "relations": relations,
        "entity_chunk_links": entity_chunk_links
    }


class ProcessRAG:
    def __init__(self, database: str = 'learning', dir_path: str = None):
        self.database = database
        self._dir_path = dir_path # check for none, raise ValueError
        self._all_docs: List[Document] = []
        self._graph_store:Optional[SimplePropertyGraphStore] = None  # check if all these have to be in an object.
        self._transformed_nodes: List = []

    async def read_docs(self):
        """
        :return:
        """

        reader = SimpleDirectoryReader(input_dir=self._dir_path, recursive=True)  # is not async? # check if directory etc.

        for docs in reader.iter_data():
            self._all_docs.extend(docs)

        return

    async def extract_info(self):

        node_parser = SimpleNodeParser.from_defaults(chunk_size=500, chunk_overlap=50)
        nodes = node_parser.get_nodes_from_documents(self._all_docs)  # get nodes from graph
        # Processes a batch of documents to extract Knowledge graphs and saves to file
        self._graph_store = SimplePropertyGraphStore()
        extractor = SimpleLLMPathExtractor(llm=g_llm_model  ,
                                           max_paths_per_chunk=10)  # 10 paths is default. Can be changed as per needs

        # Create Ingestion Pipeline for transformations
        pipeline = IngestionPipeline(
            transformations=[extractor]
        )

        # Retry mechanism for llm timeouts
        max_retries = 5
        retries = 0
        transformed_nodes = None
        while retries < max_retries:
            try:
                # Use the ingestion pipeline to run transformations
                transformed_nodes = await pipeline.arun(nodes=nodes, show_progress=True)
                break  # Exit loop on success
            except Exception as e:
                retries += 1
                logger.info(
                    f"[ERROR]: llm timeout, retrying ({retries}/{max_retries})...")
                await asyncio.sleep(5)  # Wait before retrying
                if retries == max_retries:
                    logger.error(
                        f"[ERROR]: llm timeout, exiting...")
                    raise

        # Add embeddings and insert transformed nodes into graph store
        for node in transformed_nodes:
            # Get the text content from the node
            node_text = node.get_content(metadata_mode=MetadataMode.EMBED)
            # Generate embedding using SentenceTransformer
            embedding = g_embedding_model.encode(node_text, convert_to_tensor=False)
            # Store embedding in the node
            node.embedding = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

            # Store the transformed node
            self._transformed_nodes.append(node)

            # Create ChunkNode to represent the text chunk
            chunk_node = ChunkNode(
                id_=node.id_,
                text=node_text,
                label="__Chunk__",
                properties={
                    "source": node.metadata.get("file_name", ""),
                    "file_path": node.metadata.get("file_path", ""),
                    "file_type": node.metadata.get("file_type", ""),
                    "page": node.metadata.get("page_label", ""),
                    "doc_id": node.ref_doc_id or "",
                },
                embedding=node.embedding
            )
            await self._graph_store.aupsert_nodes([chunk_node])

            # Get extracted entities and relations
            kg_nodes = node.metadata.get("nodes", [])
            kg_rels = node.metadata.get("relations", [])

            # Link entities to the chunk they came from
            for entity_node in kg_nodes:
                entity_chunk_rel = Relation(
                    label="EXTRACTED_FROM",
                    source_id=entity_node.name,
                    target_id=chunk_node.id_,
                    properties={}
                )
                await self._graph_store.aupsert_relations([entity_chunk_rel])

            # Insert entities and their relationships into graph store
            await self._graph_store.aupsert_nodes(kg_nodes)
            await self._graph_store.aupsert_relations(kg_rels)

        return

    async def store_graph(self) -> Dict[str, Any]:
        """
        Extract data from SimplePropertyGraphStore and store in Neo4j.

        This method:
        1. Extracts all entities (EntityNode) from the graph store
        2. Extracts all chunks (ChunkNode) from the graph store
        3. Extracts all relations (entity-entity and entity-chunk)
        4. Transforms data to Neo4j format
        5. Calls GraphRepository.insert_graph() to persist to Neo4j

        Returns:
            Dict containing statistics about inserted data

        Raises:
            ValueError: If graph_store is not initialized (call extract_info first)
        """
        # Validation
        if self._graph_store is None:
            raise ValueError("Graph store not initialized. Call extract_info() first.")

        # Step 1: Extract all nodes from graph store
        all_nodes = self._graph_store.get()  # Returns List[LabelledNode]

        # Step 2: Separate entities and chunks
        chunks = []
        entities = {}

        for node in all_nodes:
            if isinstance(node, ChunkNode):
                # Transform ChunkNode to Neo4j chunk format
                chunks.append({
                    "id": node.id_ if node.id_ else str(uuid4()),
                    "text": node.text,
                    "metadata": {
                        "source": node.properties.get("source", ""),
                        "file_path": node.properties.get("file_path", ""),
                        "file_type": node.properties.get("file_type", ""),
                        "page": node.properties.get("page", ""),
                        "doc_id": node.properties.get("doc_id", ""),
                    }
                })
            elif isinstance(node, EntityNode):
                # Transform EntityNode to Neo4j entity format
                key = node.name.lower().replace(" ", "_")
                entities[key] = {
                    "label": node.label,
                    "name": node.name,
                    "properties": node.properties.copy() if node.properties else {}
                }

        # Step 3: Extract all relations
        all_triplets = self._graph_store.get_triplets()  # List[Tuple[LabelledNode, Relation, LabelledNode]]

        relations = {}
        entity_chunk_links = []
        rel_idx = 0

        for source_node, relation, target_node in all_triplets:
            # Check if this is an entity-chunk relationship
            if isinstance(target_node, ChunkNode) and relation.label == "EXTRACTED_FROM":
                entity_chunk_links.append({
                    "entity_name": source_node.name,
                    "chunk_id": target_node.id_,
                    "relationship": "EXTRACTED_FROM"
                })
            elif isinstance(source_node, EntityNode) and isinstance(target_node, EntityNode):
                # This is an entity-entity relationship
                relations[f"rel_{rel_idx}"] = {
                    "source_id": source_node.name,
                    "target_id": target_node.name,
                    "label": relation.label,
                    "properties": relation.properties.copy() if relation.properties else {}
                }
                rel_idx += 1

        # Step 4: Build graph data structure for Neo4j
        graph_data = {
            "chunks": chunks,
            "nodes": entities,
            "relations": relations,
            "entity_chunk_links": entity_chunk_links
        }

        # Step 5: Insert into Neo4j
        async with g_driver.session(database=self.database) as session:
            repo = GraphRepository(session)
            await repo.insert_graph(graph_data)

        # Return statistics
        stats = {
            "chunks_inserted": len(chunks),
            "entities_inserted": len(entities),
            "relations_inserted": len(relations),
            "entity_chunk_links_inserted": len(entity_chunk_links)
        }

        logger.info(f"Graph stored successfully: {stats}")
        return stats

    async def process_and_store_document(self,
                                   document_text: Sequence,
                                   entities: List[Dict[str, Any]],
                                   relationships: List[Dict[str, Any]],
                                   doc_metadata: Dict[str, Any]):
        """
        Process a document and store it in Neo4j with chunks, entities, and relationships.

        Args:
            document_text: The full document text
            entities: List of extracted entities [{"name": "...", "label": "...", "start": 0, "end": 10}, ...]
            relationships: List of relationships [{"source": "...", "target": "...", "type": "..."}, ...]
            doc_metadata: Document metadata {"source": "...", "doc_id": "...", "page": 1}
        """
        # Build the graph structure
        graph_data = await build_rag_graph(document_text, entities, relationships, doc_metadata)

        logger.info(f"Created {len(graph_data['chunks'])} chunks")
        logger.info(f"Created {len(graph_data['nodes'])} entity nodes")
        logger.info(f"Created {len(graph_data['relations'])} relationships")
        logger.info(f"Created {len(graph_data['entity_chunk_links'])} entity-chunk links")

        # Insert into Neo4j
        async with g_driver.session() as session:
            repo = GraphRepository(session)
            await repo.insert_graph(graph_data)
            logger.info("Graph inserted successfully into Neo4j")

        return graph_data

    async def get_entity_context(self, entity_name: str) -> Optional[EntityContext]:
        """Get full context for an entity including relationships and source chunks."""
        async with g_driver.session(database=self.database) as session:
            repo = GraphRepository(session)
            return await repo.get_entity_with_context(entity_name)

    async def get_chunks_for_entity(self, entity_name: str) -> List[Chunk]:
        """Get all text chunks that mention a specific entity."""
        async with g_driver.session(database=self.database) as session:
            repo = GraphRepository(session)
            return await repo.get_chunks_for_entity(entity_name)


if __name__ == '__main__':
    # Example usage
    rag = ProcessRAG(database='learning',dir_path='/Users/godzilla/Documents/Work/Learning/AgenticAI/agentic_ai/test')

    # Sample document
    document = """
    Apple Inc. was founded by Steve Jobs and Steve Wozniak in Cupertino, California in 1976.
    The company started in a garage and initially focused on personal computers.
    Steve Jobs served as CEO and led the company through incredible innovation.
    In 2007, Apple launched the iPhone, which revolutionized the smartphone industry.
    The iPhone became one of the most successful products in history.
    """

    # These would typically come from an NER (Named Entity Recognition) model
    extracted_entities = [
        {"name": "Apple Inc.", "label": "Company", "start": 5, "end": 15},
        {"name": "Steve Jobs", "label": "Person", "start": 30, "end": 41},
        {"name": "Steve Wozniak", "label": "Person", "start": 46, "end": 60},
        {"name": "Cupertino", "label": "Location", "start": 64, "end": 73},
        {"name": "California", "label": "Location", "start": 75, "end": 85},
        {"name": "iPhone", "label": "Product", "start": 250, "end": 256}
    ]

    extracted_relationships = [
        {"source": "Steve Jobs", "target": "Apple Inc.", "type": "FOUNDED"},
        {"source": "Steve Wozniak", "target": "Apple Inc.", "type": "FOUNDED"},
        {"source": "Steve Jobs", "target": "Apple Inc.", "type": "CEO_OF"},
        {"source": "Apple Inc.", "target": "iPhone", "type": "LAUNCHED"},
        {"source": "Apple Inc.", "target": "Cupertino", "type": "LOCATED_IN"}
    ]

    metadata = {
        "source": "company_history.txt",
        "doc_id": "doc_123",
        "page": 1,
        "section": "Company Overview"
    }

    async def run_example():
        # Process documents and extract knowledge graph
        await rag.read_docs()
        await rag.extract_info()

        # Store in Neo4j
        stats = await rag.store_graph()
        print(f"\n=== Graph Storage Statistics ===")
        print(f"Chunks inserted: {stats['chunks_inserted']}")
        print(f"Entities inserted: {stats['entities_inserted']}")
        print(f"Relations inserted: {stats['relations_inserted']}")
        print(f"Entity-chunk links: {stats['entity_chunk_links_inserted']}")

        # Query example: Get structured entity context
        # Replace "Apple Inc." with an actual entity from your documents
        entity_name = "Apple Inc."  # Example - adjust based on your data
        context = await rag.get_entity_context(entity_name)

        if context:
            print(f"\n=== Context for {entity_name} ===")
            print(f"Entity: {context.entity.name} ({context.entity.label})")
            print(f"Relationships: {len(context.relationships)}")
            for rel in context.relationships:
                print(f"  - {rel.type}: {rel.source} → {rel.target}")
            print(f"Mentioned in {len(context.chunks)} chunks")

            # Serialize to JSON
            print("\n=== JSON Output ===")
            import json
            print(json.dumps(context.model_dump(), indent=2))
        else:
            print(f"Entity '{entity_name}' not found")

        # Get chunks for entity
        chunks = await rag.get_chunks_for_entity(entity_name)
        print(f"\n=== Chunks mentioning {entity_name} ===")
        for chunk in chunks:
            print(f"Chunk {chunk.metadata.index}: {chunk.text[:100]}...")

        # Show JSON serialization options
        if chunks:
            print("\n=== Serialization Examples ===")
            print("As dict:", chunks[0].model_dump())
            print("As JSON string:", chunks[0].model_dump_json())

    asyncio.run(run_example())
