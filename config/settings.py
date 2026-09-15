import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

base_path = Path(os.getenv("BASE_PATH", Path.cwd()))
backend_directory = base_path / os.getenv("BACKEND_PATH", "")
data_directory = base_path / "Data" / "Wikipedia"
pipeline_directory = base_path / "Data" / "Module-7"
corpus_directory = backend_directory / "corpus"
grid_output_directory = backend_directory / "grid_models"
chat_model_path = backend_directory / "chat_model"
best_model_path = chat_model_path / "model-best"
config_path = backend_directory / "config.cfg"
sentence_corpus_path = data_directory / "mineral_sentence_corpus"
frontend_url = os.getenv("FRONTEND_URL", "http://localhost:8080")
