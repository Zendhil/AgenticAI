# Standard library imports
import yaml
import logging
from pathlib import Path
import platform
from typing import Any

# Third Party libraries
from llama_index.llms.ollama import Ollama
from langchain_ollama import OllamaLLM
# Correct
from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
from agentic_ai.common.environment import get_device

logger = logging.getLogger(__name__)
base_dir=Path(__file__).parent
MODEL_CONFIG_FILE = base_dir.joinpath("llm_models.yaml")




class Config:
    def __init__(self):
        self._config:dict=self._load_models_config()
        logging.basicConfig(
            level=self._config.get("logging_settings").get("log_level"),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        )
    def _load_models_config(self)->dict:

        try:
            with open(MODEL_CONFIG_FILE, 'r') as stream:
               self._config = yaml.safe_load(stream)
        except FileNotFoundError:
            logger.error("Config file not found")
            raise FileNotFoundError
        except Exception as e:
            logger.error(f"Exception: {e}")
            raise e
        return self._config

    def _get_config(self)->dict:
        if self._config is None:
            self._config = self._load_models_config()
        return self._config

    def get_embeddings_model(self, function='default')->Any:
        embeddings_model = self._config.get("embeddings")
        if not embeddings_model:
            raise KeyError(f"Embedding model not found")
        model = embeddings_model.get(function)
        if not model:
            raise KeyError(f"Embedding function model not found")
        return model

    def get_llm_model(self, function='default')->Any:
        llm_model = self._config.get("llm_model")
        if not llm_model:
            raise KeyError(f"LLM model not found")
        llm_model_str = llm_model.get(function)
        if not llm_model_str:
            raise KeyError(f"Embedding function model not found")

        return ChatOllama(model=llm_model_str)


config = Config()  #singleton class
get_config = config._get_config
get_embeddings_model = config.get_embeddings_model
get_llm_model = config.get_llm_model


class EmbeddingService:
    """Singleton service for text embeddings using SentenceTransformer."""

    _instance = None
    _model = None
    _dim = None

    @classmethod
    def get_instance(cls):
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the embedding model (called only once)."""
        if EmbeddingService._model is None:
            model_name = get_embeddings_model()
            device = get_device()
            logger.info(f"Loading embedding model: {model_name} on device: {device}")
            EmbeddingService._model = SentenceTransformer(model_name, device=device)
            EmbeddingService._dim = EmbeddingService._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding dimension: {EmbeddingService._dim}")

    def encode(self, text: str) -> list[float]:
        """Encode text to embedding vector."""
        if EmbeddingService._model is None:
            raise RuntimeError("Embedding model not initialized")
        embedding = EmbeddingService._model.encode(text)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if EmbeddingService._dim is None:
            raise RuntimeError("Embedding model not initialized")
        return EmbeddingService._dim

    @classmethod
    def get_embedding_model(cls)->Any:
        if cls._model is None:
            raise RuntimeError("Embedding model not initialized")
        return cls._model

es = EmbeddingService()
get_embedding_model = es.get_embedding_model

if __name__ == '__main__':
    print(get_config())
    print(get_embeddings_model())
    # print(get_llm_model())
    llm_model = get_llm_model()
    # llm_model.complete(["waht model are you using?"])