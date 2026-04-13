import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
load_dotenv()
class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Agri-Brain"
    
    # Database
    MONGO_URL: str = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    DB_NAME: str = os.getenv("DB_NAME", "agri_brain_db")
    
    
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "jina")
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "JAYADIR/mdts-agxqa-circuit-full-bm25")
    RERANKER_BATCH_SIZE: int = int(os.getenv("RERANKER_BATCH_SIZE", "16"))
    RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "7"))
    ENABLE_RERANKER: bool = os.getenv("ENABLE_RERANKER", "1").strip().lower() in ("1", "true", "yes", "y", "on")

    print(os.getenv("EMBEDDING_MODEL"))
    class Config:
        case_sensitive = True

settings = Settings()