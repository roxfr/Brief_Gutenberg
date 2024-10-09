from langchain_openai.embeddings import AzureOpenAIEmbeddings
from config import load_config

config = load_config()
AZURE_OPENAI_API_BASE = config["AZURE_OPENAI_API_BASE"]
AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
AZURE_DEPLOYEMENT = config["AZURE_DEPLOYEMENT"]
CHUNK_SIZE = config["CHUNK_SIZE"]

class EmbeddingModel:
    def __init__(self):
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_API_BASE,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_DEPLOYEMENT,
            chunk_size=CHUNK_SIZE
        )

    def get_embedding_model(self):
        return self.embedding_model
