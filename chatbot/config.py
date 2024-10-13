import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    config = {
        "CSV_INPUT_PATH": os.path.join("../data", "gutenberg.csv"),
        "CSV_CLEANED_PATH": os.path.join("../data", "gutenberg_cleaned.csv"),
        "CHROMA_PATH": os.path.join("../models", "chroma_langchain_db"),
        "EMBEDDINGS_FILE": os.path.join("../models", "embeddings.csv"),
        "DOCS_FILE": os.path.join("../models", "docs.csv"),
        "FAISS_INDEX": os.path.join("../models", "faiss_index.index"),
        "CHAT_MODEL": os.path.join("../models", "Llama-2-13B-Chat-GGUF/llama-2-13b-chat.Q8_0.gguf"),
        "CHUNK_SIZE": int(100),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_API_BASE": os.getenv("AZURE_OPENAI_API_BASE"), 
        "AZURE_OPENAI_API_ENDPOINT": os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "AZURE_DEPLOYMENT_NAME": os.getenv("AZURE_DEPLOYMENT_NAME"),
        "AZURE_API_VERSION": os.getenv("AZURE_API_VERSION"),
        "AZURE_DEPLOYMENT": os.getenv("AZURE_DEPLOYMENT"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
        "HUGGING_TOKEN": os.getenv("HUGGING_TOKEN")
    }
    return config