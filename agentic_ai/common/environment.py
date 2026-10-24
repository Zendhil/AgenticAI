
# Standard library imports
import torch
import platform
from pathlib import Path
import logging
from dotenv import load_dotenv
import os

#third party libraries
from neo4j import AsyncGraphDatabase


root_dir = Path(__file__).parent.parent
print (root_dir)
load_dotenv()

logger = logging.getLogger(__name__)


class Environment:
    def __init__(self):
        print("init environment")
        self._device:str = self._detect_device()
        self._graph_db_dtl = self.get_prd_grf_db_details()

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

    def get_prd_grf_db_details(self, database='DEFAULT_DATABASE'):
        # Use get() to retrieve the DB_URL, defaulting to None if not set
        self._graph_db_dtl = {
            "uri": os.getenv("PRD_GRF_DB_URI"),
            "user": os.getenv("PRD_GRF_DB_USER"),
            "pwd": os.getenv("PRD_GRF_DB_PWD"),
            "db":os.getenv(database, "neo4j")  # default DB
        }

        for k,v in self._graph_db_dtl.items():
            if not v :
                raise ValueError(f"Graph Database details not set in the environment variables:{k}")

        return self._graph_db_dtl



    def get_device(self)->str:
        print("get device")
        if self._device is None:
            self._device = self._detect_device()
        return self._device

    def get_redis_url(self) -> str:
        """Get Redis URL from environment variables."""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        return redis_url

environment = Environment()
get_device=environment.get_device
get_prd_grf_db_details = environment.get_prd_grf_db_details
get_redis_url = environment.get_redis_url

if __name__ == '__main__':

    print(get_device())
    print(get_prd_grf_db_details())