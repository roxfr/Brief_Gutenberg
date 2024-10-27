import os
import logging
import uuid
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from chromadb import PersistentClient, Settings
from models.embedding_wrapper import EmbeddingWrapperSingleton


DEFAULT_N_GPU_LAYERS = -1  # Nombre de couches
DEFAULT_N_BATCH = 8  # Nombre de requêtes à traiter
DEFAULT_TEMPERATURE = 0.0  # Contrôle la créativité
DEFAULT_MAX_TOKENS = 512  # Nombre maximum de tokens
DEFAULT_SEED = 42  # Semence pour la randomisation
DEFAULT_T_TOP = 0.0  # Contrôle la diversité
DEFAULT_STREAMING = False
DEFAULT_N_CTX = 512  # Taille du contexte
DEFAULT_TOP_P = 0.0  # Paramètre pour la diversité
DEFAULT_REPETITION_PENALTY = 1.2  # Pénalité
DEFAULT_N_PREDICT = 50  # Nombre de tokens

def process_batch(batch_data):
    """Traite un lot de données pour les ajouter au cache"""
    metadata = [{'title': row['Title'], 'author': row['Author'], 'ebook_no': str(row['EBook-No.'])} for _, row in batch_data.iterrows()]
    unique_ids = [str(uuid.uuid4()) for _ in range(len(batch_data))]
    return list(zip(unique_ids, batch_data['Summary'].tolist(), metadata))

def create_llama(model_path: str, n_gpu_layers: int = DEFAULT_N_GPU_LAYERS, n_batch: int = DEFAULT_N_BATCH, 
                 temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS, 
                 seed: int = DEFAULT_SEED, t_top: float = DEFAULT_T_TOP, streaming: bool = DEFAULT_STREAMING,
                 n_ctx: int = DEFAULT_N_CTX, top_p: float = DEFAULT_TOP_P, 
                 repetition_penalty: float = DEFAULT_REPETITION_PENALTY, n_predict: int = DEFAULT_N_PREDICT) -> LlamaCpp:
    """Créer et retourner un modèle"""
    model_path = os.path.abspath(model_path)
    logging.info(f"Chargement du modèle")
    try:
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            n_ctx=n_ctx,
            top_p=top_p,
            model_kwargs={
                "t_top": t_top,
                "repetition_penalty": repetition_penalty,
                "n_predict": n_predict,
            },
            streaming=streaming,
            verbose=False,
        )
        logging.info("Modèle chargé avec succès")
        return llm
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle : {e}")
        raise

def create_vector_store(data, persist_directory: str, collection_name: str = "project_gutenberg") -> Chroma:
    """Créer ou mettre à jour un magasin de vecteurs"""
    required_columns = ['Author', 'Title', 'Summary', 'Subject', 'EBook-No.']
    if data.empty or not all(col in data.columns for col in required_columns):
        logging.error("Le DataFrame est vide ou manque des colonnes requises")
        return None

    client = PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(collection_name)
        logging.info("Collection existante chargée")
        return Chroma(client=client, collection_name=collection_name, embedding_function=EmbeddingWrapperSingleton())
    except Exception:
        logging.warning("Collection inexistante")
        collection = client.create_collection(collection_name)

    logging.info("Ajout des textes dans le cache...")
    batches = [data.iloc[i:i + 32] for i in range(0, len(data), 32)]
    for batch in batches:
        unique_ids = [str(uuid.uuid4()) for _ in range(len(batch))]
        documents = batch['Summary'].tolist()
        metadata = [
            {
                'Title': row['Title'],
                'Author': row['Author'],
                'EBook-No': row['EBook-No.'],
                'Subject': row['Subject']
            } for _, row in batch.iterrows()
        ]

        if not documents or not metadata:
            logging.warning("Aucun document ou métadonnées à ajouter pour le lot.")
            continue

        embeddings = EmbeddingWrapperSingleton().embed_documents(documents)
        collection.add(ids=unique_ids, documents=documents, metadatas=metadata, embeddings=embeddings)

    return Chroma(client=client, collection_name=collection_name, embedding_function=EmbeddingWrapperSingleton())