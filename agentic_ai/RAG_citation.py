# Standard library imports
import logging
from pathlib import Path
# from config import NEO4J, INDEX, embeddings

# Third party imports
from sentence_transformers import SentenceTransformer
from llama_index.core import Document
from llama_index.core.graph_stores.simple_labelled import SimplePropertyGraphStore
from llama_index.core.indices.property_graph.transformations import SimpleLLMPathExtractor
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.ingestion.pipeline import run_transformations
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_neo4j import Neo4jVector

# App imports
from agentic_ai.common import get_config, get_embeddings_model, get_device,graph_db_connect

g_device = get_device()
g_embeddings_model = get_embeddings_model()
g_embedding_model = SentenceTransformer(model_name_or_path=g_embeddings_model,device = g_device,similarity_fn_name='cosine')



# docs = []
# for pdf in Path("docs").glob("*.pdf"):
#     for page in PyPDFLoader(str(pdf)).load():
#         page.metadata = {"source": pdf.name, "page": page.metadata["page"] + 1}
#         docs.append(page)
#
# splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
# chunks = splitter.split_documents(docs)
#
# Neo4jVector.from_documents(chunks, embeddings, index_name=INDEX, **NEO4J)
# print(f"Ingested {len(chunks)} chunks from {len(docs)} pages")

class ProcessRAG:
    def __init__(self):
        pass
    def read_data(self):

        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = node_parser.get_nodes_from_documents(documents)  # get nodes from graph

        # Processes a batch of documents to extract Knowledge graphs and saves to file
        graph_store = SimplePropertyGraphStore()
        extractor = SimpleLLMPathExtractor(llm=llm,
                                           max_paths_per_chunk=10)  # 10 paths is default. Can be changed as per needs

        # Retry mechanism for llm timeouts
        max_retries = 10
        retries = 0
        transformed_nodes = None
        while retries < max_retries:
            try:
                transformed_nodes = run_transformations(nodes, [extractor], show_progress=True)
                break  # Exit loop on success
            except Exception as e:
                retries += 1
                logger.info(
                    f"[ERROR] Link {documents[0].metadata['source']}: llm timeout, retrying ({retries}/{max_retries})...")
                time.sleep(5)  # Wait before retrying
        if retries == max_retries:
            logger.info(
                f"[ERROR] Link {documents[0].metadata['source']}: failed after {max_retries} retries. Skipping.")
            return

    def store_info(self):
        query = "What is my name"
        embeddings = g_embedding_model.encode(query)
        print(embeddings)

    def fetch_info(self):
        pass


if __name__ == '__main__':
    ProcessRAG().store_info()