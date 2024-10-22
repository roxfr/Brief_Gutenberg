import os
import logging
import uuid
from langchain_community.llms import LlamaCpp
from langchain_chroma import Chroma
from chromadb import PersistentClient, Settings
from models.embedding_wrapper import EmbeddingWrapperSingleton


# Constantes par défaut
DEFAULT_N_GPU_LAYERS = -1
DEFAULT_N_BATCH = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024
DEFAULT_SEED = 42
DEFAULT_T_TOP = 1.0
DEFAULT_STREAMING = False

def process_batch(batch_data):
    """Traite un lot de données pour les ajouter au cache"""
    metadata = [{'title': row['Title'], 'author': row['Author'], 'ebook_no': str(row['EBook-No.'])} for _, row in batch_data.iterrows()]
    unique_ids = [str(uuid.uuid4()) for _ in range(len(batch_data))]
    return list(zip(unique_ids, batch_data['Summary'].tolist(), metadata))

def create_llama(model_path: str, n_gpu_layers: int = DEFAULT_N_GPU_LAYERS, n_batch: int = DEFAULT_N_BATCH, 
                 temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS, 
                 seed: int = DEFAULT_SEED, t_top: float = DEFAULT_T_TOP, streaming: bool = DEFAULT_STREAMING) -> LlamaCpp:
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
            model_kwargs={"t_top": t_top},
            streaming=streaming,
            verbose=False,
        )
        logging.info("Modèle chargé avec succès")
        return llm
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle : {e}")
        raise

def create_vector_store(data, persist_directory: str, collection_name: str = "project_gutenberg") -> Chroma:
    """Créer ou mettre à jour un magasin de vecteurs."""
    if data.empty or not all(col in data.columns for col in ['Title', 'Author', 'Summary', 'EBook-No.']):
        logging.error("Le DataFrame est vide ou manque des colonnes requises")
        return None

    client = PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(collection_name)
        logging.info("Collection existante chargée")
    except Exception:
        logging.warning("Collection inexistante")
        collection = client.create_collection(collection_name)

    logging.info("Ajout des textes dans le cache...")    
    batches = [data.iloc[i:i + 32] for i in range(0, len(data), 32)]
    for batch in batches:
        unique_ids = [str(uuid.uuid4()) for _ in range(len(batch))]
        documents = batch['Summary'].tolist()
        metadata = [{'title': row['Title'], 'author': row['Author'], 'ebook_no': str(row['EBook-No.'])} for _, row in batch.iterrows()]
        
        collection.add(ids=unique_ids, documents=documents, metadatas=metadata)

    return Chroma(client=client, collection_name=collection_name, embedding_function=EmbeddingWrapperSingleton())