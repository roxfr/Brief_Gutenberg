import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    config = {
        "CSV_INPUT_PATH": os.path.join("../data", "gutenberg.csv"),
        "CSV_CLEANED_PATH": os.path.join("../data", "gutenberg_cleaned.csv"),
        "CHROMA_PATH": "../model/chroma",
        "EMBEDDINGS_FILE": os.path.join("../model", "embeddings.csv"),
        "DOCS_FILE": os.path.join("../model", "docs.csv"),
        "FAISS_INDEX": os.path.join("../model", "faiss_index.index"),
        "CHUNK_SIZE": int(500),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_API_BASE": os.getenv("AZURE_OPENAI_API_BASE"), 
        "AZURE_OPENAI_API_ENDPOINT": os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "AZURE_DEPLOYMENT_NAME": os.getenv("AZURE_DEPLOYMENT_NAME"),
        "AZURE_API_VERSION": os.getenv("AZURE_API_VERSION"),
        "AZURE_DEPLOYMENT": os.getenv("AZURE_DEPLOYMENT"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY")
    }
    return config
