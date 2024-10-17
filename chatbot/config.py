import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    config = {
        "CSV_INPUT_PATH": os.path.join("../data", "gutenberg.csv"),
        "CSV_CLEANED_PATH": os.path.join("../data", "gutenberg_cleaned.csv"),
        "CHROMA_PATH": os.path.join("../data", "chroma_langchain_db"),
        "CHAT_MODEL": os.path.join("../models", "llama-3.2-8B-Instruct-agent-003-128k-code-DPO.Q8_0.gguf"),
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