# Standard library imports
import logging
from pathlib import Path
from config import NEO4J, INDEX, embeddings

# Third party imports
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jVector

# App imports
from agentic_ai.common import get_config, get_device






docs = []
for pdf in Path("docs").glob("*.pdf"):
    for page in PyPDFLoader(str(pdf)).load():
        page.metadata = {"source": pdf.name, "page": page.metadata["page"] + 1}
        docs.append(page)

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_documents(docs)

Neo4jVector.from_documents(chunks, embeddings, index_name=INDEX, **NEO4J)
print(f"Ingested {len(chunks)} chunks from {len(docs)} pages")

class ProcessRAG:
    def __init__(self):
        pass
    def read_data(self):
        pass
    def store_info(self):
        pass
    def fetch_info(self):
        pass

if __name__ == '__main__':
    pass