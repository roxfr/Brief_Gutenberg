from sentence_transformers import SentenceTransformer
from utils.config import load_config


config = load_config()
EMBEDDINGS_MODEL = config["EMBEDDINGS_MODEL"]

class EmbeddingWrapperSingleton:
    """Singleton pour l'embedding des documents et des requêtes"""
    _instance = None
    def __new__(cls):
        """Crée une nouvelle instance si elle n'existe pas encore"""
        if cls._instance is None:
            cls._instance = super(EmbeddingWrapperSingleton, cls).__new__(cls)
            cls._instance.embeddings_model = SentenceTransformer(EMBEDDINGS_MODEL)
        return cls._instance
    def embed_documents(self, texts: list) -> list:
        """Encode les documents et retourne les embeddings"""
        embeddings = self.embeddings_model.encode(texts, show_progress_bar=True)
        return embeddings
    def embed_query(self, query: str) -> list:
        """Encode une requête et retourne l'embedding"""
        return self.embeddings_model.encode([query])[0]