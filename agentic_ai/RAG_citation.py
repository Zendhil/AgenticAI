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

from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.indices.property_graph.transformations import SimpleLLMPathExtractor

from llama_index.core.ingestion.pipeline import arun_transformations
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_neo4j import Neo4jVector

# App imports
from agentic_ai.common import get_config, get_embeddings_model, get_device, get_llm_model
from agentic_ai.common import GraphRepository, get_graph_driver

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
        # Retry mechanism for llm timeouts
        max_retries = 5
        retries = 0
        transformed_nodes = None
        while retries < max_retries:
            try:
                transformed_nodes = await arun_transformations(nodes, [extractor], show_progress=True)
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

        # Insert transformed nodes into graph store
        for node in transformed_nodes:
            kg_nodes = node.metadata.get("nodes", [])
            kg_rels = node.metadata.get("relations", [])
            self._graph_store.upsert_nodes(kg_nodes)
            self._graph_store.upsert_relations(kg_rels)

        triplets = self._graph_store.get_triplets()
        for triplet in triplets:
            print("triplets: ", triplet)


        return

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

async def get_entity_context(self, entity_name: str) -> Dict[str, Any]:
    """Get full context for an entity including relationships and source chunks."""
    async with g_driver.session(database=self.database) as session:
        repo = GraphRepository(session)
        return await repo.get_entity_with_context(entity_name)

async def get_chunks_for_entity(self, entity_name: str) -> List[Dict[str, Any]]:
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
        # Process and store the document
        await rag.read_docs()
        await rag.extract_info()
        exit()
        graph_data = await rag.process_and_store_document(document, extracted_entities, extracted_relationships, metadata)

        # Query example: Get all context for "Steve Jobs"
        context = await get_entity_context("Steve Jobs")
        print("\n=== Context for Steve Jobs ===")
        print(f"Label: {context.get('label')}")
        print(f"Relationships: {context.get('relationships')}")
        print(f"Mentioned in {len(context.get('chunks', []))} chunks")

        # Get chunks mentioning Steve Jobs
        chunks = await get_chunks_for_entity("Steve Jobs")
        print("\n=== Chunks mentioning Steve Jobs ===")
        for chunk in chunks:
            print(f"Chunk {chunk['index']}: {chunk['text'][:100]}...")

    asyncio.run(run_example())
