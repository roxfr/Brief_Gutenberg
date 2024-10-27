import logging
from sentence_transformers import SentenceTransformer
from utils.config import load_config


logging.basicConfig(level=logging.INFO)

config = load_config()
EMBEDDINGS_MODEL = config["EMBEDDINGS_MODEL"]

class EmbeddingWrapperSingleton:
    """Singleton pour l'embedding des documents et des requêtes"""
    _instance = None

    def __new__(cls):
        """Crée une nouvelle instance si elle n'existe pas encore"""
        if cls._instance is None:
            cls._instance = super(EmbeddingWrapperSingleton, cls).__new__(cls)
            try:
                cls._instance.embeddings_model = SentenceTransformer(EMBEDDINGS_MODEL)
                logging.info(f"Modèle d'embedding chargé : {EMBEDDINGS_MODEL}")
            except Exception as e:
                logging.error(f"Erreur lors du chargement du modèle d'embedding : {e}")
                raise
        return cls._instance

    def embed_documents(self, texts: list) -> list:
        """Encode les documents et retourne les embeddings"""
        if not texts:
            logging.warning("Aucun texte fourni pour l'embedding.")
            return []
        try:
            embeddings = self.embeddings_model.encode(texts, show_progress_bar=True)
            logging.info(f"Documents encodés : {len(texts)}")
            return embeddings
        except Exception as e:
            logging.error(f"Erreur lors de l'encodage des documents : {e}")
            return []

    def embed_query(self, query: str) -> list:
        """Encode une requête et retourne l'embedding"""
        if not query:
            logging.warning("Aucune requête fournie pour l'embedding.")
            return []
        try:
            embedding = self.embeddings_model.encode([query])[0]
            logging.info("Requête encodée avec succès.")
            return embedding
        except Exception as e:
            logging.error(f"Erreur lors de l'encodage de la requête : {e}")
            return []