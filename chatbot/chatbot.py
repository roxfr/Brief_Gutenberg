# python -m venv gutenberg
# .\gutenberg\Scripts\Activate.ps1 # Sous PowerShell

import os
import logging
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import uuid
import chromadb
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.schema.runnable import RunnablePassthrough
from embedding_wrapper import EmbeddingWrapperSingleton
from tools import get_author, get_subject, get_characters, get_full_text_from_url
from langchain_chroma import Chroma
from chromadb import PersistentClient, Settings
import argparse

# Constantes
MAX_CACHE_SIZE = 1024  # Limite de la taille du cache
DEFAULT_BATCH_SIZE = 32  # Taille du lot par défaut
DEFAULT_N_GPU_LAYERS = 40  # Nombre de couches GPU pour le modèle Llama
DEFAULT_N_BATCH = 512  # Taille du lot pour le modèle Llama
DEFAULT_TEMPERATURE = 0.0  # Température du modèle
DEFAULT_MAX_TOKENS = 1024  # Nombre maximum de tokens pour le modèle
DEFAULT_SEED = 42  # Seed pour la randomisation
DEFAULT_T_TOP = 1.0  # Top-p sampling
DEFAULT_STREAMING = False  # Streaming du modèle

# Configurations
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

from config import load_config
config = load_config()
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
CHROMA_PATH = config["CHROMA_PATH"]
CHAT_MODEL = config["CHAT_MODEL"]
BATCH_SIZE = config.get("BATCH_SIZE", DEFAULT_BATCH_SIZE)

# Prompt template
template = """
        **Contexte** : Vous êtes un expert en littérature, spécialisé dans les œuvres du Projet Gutenberg.

        **Instructions** :
        - Répondez uniquement avec des informations vérifiées provenant de ces textes.
        - Utilisez le magasin de vecteurs (chromadb) pour vous aider dans vos réponses.
        - Évitez de faire des suppositions ou des interprétations personnelles.
        - Si vous n'avez pas l'information, répondez par "Je ne sais pas."

        **Format** :
        - Fournissez des réponses concises et directes (une phrase maximum).
        - Mentionnez les outils disponibles pour la réponse : {tools}.

        **Question** : '{question}'.
"""

# Synonymes
synonyms = {
    "auteur": ["auteur", "écrivain", "compositeur", "rédacteur"],
    "sujet": ["sujet", "thème", "sujet principal", "question", 
              "objet", "problématique", "contenu", "résumé"],
    "personnages": ["personnages", "protagonistes", "acteurs", 
                    "figures", "rôles", "personnifications", 
                    "personnages principaux", "caractères"],
    "texte": ["texte complet", "texte intégral", "texte entier", 
              "document complet", "version intégrale", 
              "document intégral", "texte total", "version complète"]
}

# Outils
tools = {
    "get_author": "Fonction pour trouver l'auteur d'un livre",
    "get_subject": "Fonction pour obtenir le sujet d'un livre",
    "get_characters": "Fonction pour extraire les personnages d'un livre",
    "get_full_text_from_url": "Fonction pour récupérer le texte intégral ou complet d'un livre",
}

def load_csv(file_path: str) -> pd.DataFrame:
    """Charge le fichier CSV et retourne un DataFrame"""
    if not os.path.exists(file_path):
        logging.error(f"Le fichier CSV n'existe pas : {file_path}")
        raise FileNotFoundError(f"Le fichier CSV n'existe pas : {file_path}")
    
    try:
        df = pd.read_csv(file_path, sep=';')
        logging.info(f"Fichier CSV chargé : {file_path}")
        return df
    except pd.errors.EmptyDataError:
        logging.error("Le fichier CSV est vide")
        raise
    except pd.errors.ParserError as e:
        logging.error(f"Erreur de parsing du fichier CSV : {e}")
        raise
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier CSV : {e}")
        raise

def get_existing_vector_store(persist_directory: str, collection_name: str):
    """Récupère un magasin de vecteurs existant ou en crée un nouveau"""
    client = PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(collection_name)
        logging.info("Collection existante chargée")
        return collection, True, client
    except Exception:
        logging.warning("Collection inexistante")
        return None, False, client

def persist_cache(cache, collection):
    """Persiste le cache en mémoire dans le magasin de vecteurs"""
    for unique_id, document, metadata in cache:
        collection.add(ids=[unique_id], documents=[document], metadatas=[metadata])

def create_vector_store(data: pd.DataFrame, batch_size: int = BATCH_SIZE, 
                        persist_directory: str = CHROMA_PATH, 
                        collection_name: str = "project_gutenberg") -> Chroma:
    """Crée ou met à jour un magasin de vecteurs avec un cache en mémoire"""
    if data.empty or not all(col in data.columns for col in ['Title', 'Author', 'Summary', 'EBook-No.']):
        logging.error("Le DataFrame est vide ou manque des colonnes requises")
        return None

    in_memory_cache = []
    
    collection, exists, client = get_existing_vector_store(persist_directory, collection_name)
    if not exists:
        logging.info("Création d'une nouvelle collection...")
        collection = client.create_collection(collection_name)

    logging.info("Ajout des textes dans le cache en mémoire...")
    
    with ThreadPoolExecutor() as executor:
        batches = [data.iloc[i:i + batch_size] for i in range(0, len(data), batch_size)]
        results = list(executor.map(process_batch, batches))
        for batch in results:
            in_memory_cache.extend(batch)

    logging.info(f"{len(in_memory_cache)} textes ajoutés dans le cache en mémoire")

    if in_memory_cache:
        persist_cache(in_memory_cache, collection)

    logging.info(f"{len(data)} textes ajoutés ou mis à jour dans le magasin de vecteurs")
    return Chroma(client=client, collection_name=collection_name, embedding_function=EmbeddingWrapperSingleton())

def process_batch(batch_data):
    """Traite un lot de données pour les ajouter au cache"""
    metadata = [{'title': row['Title'], 'author': row['Author'], 'ebook_no': str(row['EBook-No.'])} for _, row in batch_data.iterrows()]
    unique_ids = [str(uuid.uuid4()) for _ in range(len(batch_data))]
    return list(zip(unique_ids, batch_data['Summary'].tolist(), metadata))

def configure_llama(model_path: str = CHAT_MODEL,
                    n_gpu_layers: int = DEFAULT_N_GPU_LAYERS, n_batch: int = DEFAULT_N_BATCH,
                    temperature: float = DEFAULT_TEMPERATURE, max_tokens: int = DEFAULT_MAX_TOKENS,
                    seed: int = DEFAULT_SEED, t_top: float = DEFAULT_T_TOP, streaming: bool = DEFAULT_STREAMING) -> LlamaCpp:
    """Configure et retourne le modèle Llama"""
    model_path = os.path.abspath(model_path)
    logging.info(f"Chargement du modèle Llama depuis : {model_path}")
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
        logging.info("Modèle Llama chargé avec succès")
        return llm
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle Llama : {e}")
        raise

def setup_qa_chain(llm: LlamaCpp, vector_store: Chroma) -> RunnableSequence:
    """Mise en place de la chaîne QA."""
    prompt = PromptTemplate(template=template, input_variables=["question", "tools"])
    retriever = vector_store.as_retriever(k=5)
    qa_chain = RunnableSequence(
        {
            "context": retriever, 
            "question": RunnablePassthrough(),
            "tools": RunnablePassthrough()
        }
        | prompt
        | llm
        | RunnablePassthrough()
    )    
    return qa_chain

def process_question(question: str, vector_store: Chroma) -> str:
    question_lower = question.lower()    
    if any(syn in question_lower for syn in synonyms["auteur"]):
        return get_author(vector_store, question)
    elif any(syn in question_lower for syn in synonyms["sujet"]):
        return get_subject(vector_store, question)
    elif any(syn in question_lower for syn in synonyms["personnages"]):
        return get_characters(vector_store, question)
    elif any(syn in question_lower for syn in synonyms["texte"]):
        return get_full_text_from_url(vector_store, question)    
    return None

def main():
    parser = argparse.ArgumentParser(description='Configuration de l\'application')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Taille du lot pour le traitement')
    args = parser.parse_args()

    data = load_csv(CSV_CLEANED_PATH)
    logging.info(f"Colonnes du DataFrame : {data.columns.tolist()}")
    llm = configure_llama()
    vector_store = create_vector_store(data, batch_size=args.batch_size)
    qa_chain = setup_qa_chain(llm, vector_store)

    while True:
        question = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
        if question.lower() == 'exit':
            logging.info("Sortie du programme")
            break
        
        #response = process_question(question, vector_store)
        response = None # Sans process_question
        if response is None:
            response = qa_chain.invoke({"question": question, "tools": list(tools.keys())})
        
        logging.info(f"Question : {question}\nRéponse : {response}\n")

if __name__ == "__main__":
    main()