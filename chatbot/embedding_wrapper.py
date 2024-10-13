from tqdm import tqdm
from sentence_transformers import SentenceTransformer


embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')

class EmbeddingWrapper:
    def embed_documents(self, texts: list) -> list:
        """Encode les documents et retourne les embeddings."""
        embeddings = embeddings_model.encode(texts, show_progress_bar=True)
        return embeddings

    def embed_query(self, query: str) -> list:
        """Encode une requête et retourne l'embedding."""
        return embeddings_model.encode([query])[0]