# Standard library imports
import yaml
import logging
from pathlib import Path
import platform
import torch

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

class Environment:
    def __init__(self):
        print("init environment")
        self._device:str = self._detect_device()

    def _detect_device(self):
        print("detect device")
        self._device = 'cpu'

        if platform.system() == 'Darwin' and torch.backends.mps.is_available():
            self._device = 'mps'  # Use MPS on Mac
        # Check for CUDA (NVIDIA GPU)
        elif torch.cuda.is_available():
            self._device = 'cuda'  # Use CUDA on Nvidia GPU
        else:
            self._device = 'cpu'  # Fallback to CPU if no GPU is available
        #Todo: TPUs!!!
        print(f"Running on: {self._device}")  # Log the device used
        return self._device

    def get_device(self)->str:
        print("get device")
        if self._device is None:
            self._device = self._detect_device()
        return self._device

config = Config()  #singleton class
get_config = config._get_config

environment = Environment()
get_device=environment.get_device

if __name__ == '__main__':
    print(get_config())
    print(get_device())