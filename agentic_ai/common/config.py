# Standard library imports
import yaml
import logging
from pathlib import Path
import platform
from typing import Any

# Third Party libraries
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)
base_dir=Path(__file__).parent
MODEL_CONFIG_FILE = base_dir.joinpath("llm_models.yaml")

class Config:
    def __init__(self):
        self._config:dict=self._load_models_config()

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

        return Ollama(model=llm_model_str, request_timeout=300)


config = Config()  #singleton class
get_config = config._get_config
get_embeddings_model = config.get_embeddings_model
get_llm_model = config.get_llm_model

if __name__ == '__main__':
    print(get_config())
    print(get_embeddings_model())
    print(get_llm_model())