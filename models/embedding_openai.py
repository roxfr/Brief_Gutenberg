from langchain_openai.embeddings import AzureOpenAIEmbeddings
from config import load_config

config = load_config()
AZURE_OPENAI_API_BASE = config["AZURE_OPENAI_API_BASE"]
AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
AZURE_DEPLOYMENT = config["AZURE_DEPLOYMENT"]
CHUNK_SIZE = config["CHUNK_SIZE"]

class EmbeddingModelSingleton:
    """Singleton pour le modèle d'embeddings Azure OpenAI"""    
    _instance = None
    def __new__(cls):
        """Crée une nouvelle instance si elle n'existe pas encore"""
        if cls._instance is None:
            cls._instance = super(EmbeddingModelSingleton, cls).__new__(cls)
            cls._instance.embedding_model = AzureOpenAIEmbeddings(
                azure_endpoint=AZURE_OPENAI_API_BASE,
                openai_api_key=AZURE_OPENAI_API_KEY,
                azure_deployment=AZURE_DEPLOYMENT,
                chunk_size=CHUNK_SIZE
            )
        return cls._instance
    def get_embedding_model(self):
        """Retourne le modèle d'embeddings Azure OpenAI"""
        return self.embedding_model