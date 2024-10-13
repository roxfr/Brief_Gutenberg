import os
import logging
import pandas as pd
from tqdm import tqdm
# Langchain
#from langchain.llms import LlamaCpp
from langchain_community.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
#from langchain_core.embeddings import Embeddings
from langchain.schema.runnable import RunnablePassthrough
# Embeddings
from embedding_wrapper import EmbeddingWrapper
from tools import get_author, get_subject, get_characters, get_full_text_from_url
# Chroma
import uuid
import chromadb
from langchain_chroma import Chroma
from chromadb import PersistentClient, Settings


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from config import load_config
config = load_config()
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
CHROMA_PATH = config["CHROMA_PATH"]
CHAT_MODEL = config["CHAT_MODEL"]
CHUNK_SIZE = config["CHAT_MODEL"]

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def load_csv(file_path: str) -> pd.DataFrame:
    """Charge le fichier CSV et retourne un DataFrame."""
    try:
        df = pd.read_csv(file_path, sep=';')
        logging.info(f"Fichier CSV chargé : {file_path}")
        return df
    except Exception as e:
        logging.error(f"Erreur lors du chargement du fichier CSV : {e}")
        raise

def get_existing_vector_store(persist_directory: str, collection_name: str):
    """Récupère un magasin de vecteurs existant ou en crée un nouveau."""
    client = PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(collection_name)
        logging.info("Collection existante chargée.")
        return collection, True, client
    except Exception:
        logging.warning("Collection inexistante.")
        return None, False, client

def create_vector_store(data: pd.DataFrame, batch_size: int = CHUNK_SIZE, 
                        persist_directory: str = CHROMA_PATH, 
                        collection_name: str = "project_gutenberg") -> Chroma:
    """Crée ou met à jour un magasin de vecteurs."""
    if data.empty or not all(col in data.columns for col in ['Title', 'Author', 'Summary', 'EBook-No.']):
        logging.error("Le DataFrame est vide ou manque des colonnes requises.")
        return None
    collection, exists, client = get_existing_vector_store(persist_directory, collection_name)
    if not exists:
        logging.info("Création d'une nouvelle collection...")
        collection = client.create_collection(collection_name)
        logging.info("Ajout ou mise à jour des textes dans le magasin de vecteurs...")
        for i in tqdm(range(0, len(data), batch_size), desc="Traitement des lots"):
            batch_data = data.iloc[i:i + batch_size]
            metadata = [{'author': row['Author'], 'ebook_no': str(row['EBook-No.'])} for _, row in batch_data.iterrows()]       
            unique_ids = [str(uuid.uuid4()) for _ in range(len(batch_data))]
            collection.add(ids=unique_ids, documents=batch_data['Summary'].tolist(), metadatas=metadata)
        logging.info(f"{len(data)} textes ajoutés ou mis à jour dans le magasin de vecteurs.")
    return Chroma(client=client, collection_name=collection_name, embedding_function=EmbeddingWrapper())

def configure_llama(model_path: str = CHAT_MODEL,
                    n_gpu_layers: int = 40, n_batch: int = 512,
                    temperature: float = 0.7, max_tokens: int = 150) -> LlamaCpp:
    """Configure et retourne le modèle Llama."""
    model_path = os.path.abspath(model_path)
    logging.info(f"Chargement du modèle Llama depuis : {model_path}")
    try:
        llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            #verbose=True,
            verbose=False,
        )
        logging.info("Modèle lama chargé avec succès.")
        return llm
    except Exception as e:
        logging.error(f"Erreur lors du chargement du modèle Llama : {e}")
        raise

tools = {
    "get_author": "Fonction pour trouver l'auteur d'un livre",
    "get_subject": "Fonction pour obtenir le sujet d'un livre",
    "get_characters": "Fonction pour extraire les personnages d'un livre",
    "get_full_text_from_url": "Fonction pour récupérer le texte intégral ou complet d'un livre",
}

# template = """
#     Tu es un agent expert en littérature spécialisé dans les œuvres du projet Gutenberg. 
#     Lorsque l'utilisateur pose une question : '{question}', 
#     tu dois répondre uniquement avec des informations vérifiées issues de ces livres, 
#     en utilisant le magasin de vecteurs à ta disposition {tools}.

#     - Si tu ne sais pas la réponse, ne fais pas d'hypothèses ou d'inventions. 
#     Réponds simplement par "Je ne sais pas." ou en donnant ce que tu sais avec précision.
#     - Fournis des réponses directes et factuelles. 
#     Par exemple, si la question concerne l'auteur, donne uniquement le nom de l'auteur. 
#     Si c'est sur le sujet, donne un résumé court.
    
#     Voici des exemples :
#     - Bonne réponse : "L'auteur de L'Assommoir est Émile Zola."
#     - Mauvaise réponse : "Je ne sais pas, mais cela pourrait être quelque chose comme John Doe." (Ne fais pas ça !)
# """

template = """
            Tu es un agent expert en littérature spécialisé dans les œuvres du projet Gutenberg.
            Lorsque l'utilisateur pose une question : '{question}', 
            répond uniquement avec des informations vérifiées issues de ces livres.
            Ne fais pas d'hypothèses. Si tu ne sais pas, dis "Je ne sais pas."
            Fournis des réponses directes, factuelles et en une seule phrase.
            Exemples :
            Bonne réponse : "L'auteur de L'Assommoir est Émile Zola."
            Mauvaise réponse : "Je ne sais pas, mais cela pourrait être quelque chose comme John Doe."
            Outils disponibles : {tools}.
"""

questions = [
        "Qui est l'auteur du livre L'Assommoir ?",
        "Quel sujet est traité dans House of Atreus ?",
        "Qui sont les personnages principaux dans Uninhabited House ?",
        "Quels sont les titres des livres de l'auteur Dickens Charles ?",
        "Peux-tu me récupérer le texte du livre Blue Bird sur internet ?"
]

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
    data = load_csv(CSV_CLEANED_PATH)
    logging.info(f"Colonnes du DataFrame : {data.columns.tolist()}")
    llm = configure_llama()
    vector_store = create_vector_store(data)
    qa_chain = setup_qa_chain(llm, vector_store)
    logging.info(f"Outils : {list(tools.keys())}")
    while True:
        try:
            question = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
            if question.lower() == 'exit':
                logging.info("Sortie du programme.")
                break
            #response = process_question(question, vector_store)
            response = None
            if response is None:
                response = qa_chain.invoke({"question": question, "tools": list(tools.keys())})
            logging.info(f"Question : {question}\n")
            logging.info(f"Réponse : {response}\n")
        except Exception as e:
            logging.error(f"Une erreur est survenue : {e}")

if __name__ == "__main__":
    main()