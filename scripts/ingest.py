from pathlib import Path
from src.ingestion.pipeline import run_ingestion_pipeline

if __name__ == "__main__":
    run_ingestion_pipeline(Path("data/raw"))