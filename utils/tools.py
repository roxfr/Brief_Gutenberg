import os
import logging
import requests
import pandas as pd
from langchain_chroma import Chroma
import spacy
from models.embedding_wrapper import EmbeddingWrapperSingleton

nlp = spacy.load("en_core_web_sm")
logging.basicConfig(level=logging.INFO)

TITLE_KEY = 'Title'
AUTHOR_KEY = 'Author'
SUBJECT_KEY = 'Subject'
SUMMARY_KEY = 'Summary'
EBOOK_NO_KEY = 'EBook-No'

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

def search_in_vector_store(vector_store: Chroma, query: str, k: int = 3, filter=None):
    """Effectue une recherche dans le magasin de vecteurs"""
    try:
        results = vector_store.similarity_search(query, k=k, filter=filter)
        logging.info(f"Résultats trouvés pour '{query}': {len(results)}")
        return results
    except Exception as e:
        logging.error(f"Erreur lors de la recherche pour '{query}': {e}")
        return []

def get_books_by_author(vector_store: Chroma, author: str) -> list:
    """Recherche des livres par auteur"""
    logging.info(f"Recherche des livres pour l'auteur : {author}")
    results = search_in_vector_store(vector_store, author, k=5)
    
    if results:
        return [result.metadata.get(TITLE_KEY, 'Titre non trouvé') for result in results]
    logging.warning(f"Aucun livre trouvé pour l'auteur : {author}")
    return []

def get_author_by_title(vector_store: Chroma, title: str) -> str:
    """Recherche l'auteur d'un livre par son titre"""
    logging.info(f"Recherche de l'auteur pour : {title}")
    results = search_in_vector_store(vector_store, title)

    if results:
        return results[0].metadata.get(AUTHOR_KEY, 'Auteur non trouvé')
    logging.warning(f"Aucun auteur trouvé pour le titre : {title}")
    return "Auteur non trouvé"

def get_subject_by_title(vector_store: Chroma, title: str) -> str:
    """Retourne le sujet d'un livre par son titre"""
    logging.info(f"Recherche du sujet pour : {title}")
    results = search_in_vector_store(vector_store, title)
    if results:
        subject = results[0].metadata.get(SUBJECT_KEY, 'Sujet non trouvé')
        if isinstance(subject, list):
            return subject[0] if subject else 'Sujet non trouvé'
        return subject
    logging.warning(f"Aucun sujet trouvé pour le titre : {title}")
    return "Sujet non trouvé"

def get_characters_by_title(vector_store: Chroma, title: str) -> list:
    """Extrait les personnages d'un livre par son titre"""
    logging.info(f"Recherche du/des personnage(s) pour : {title}")
    results = search_in_vector_store(vector_store, title)
    if results:
        summary = results[0].metadata.get(SUMMARY_KEY, "")
        if summary:
            doc = nlp(summary)
            characters = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
            return list(characters) if characters else ['Aucun personnage trouvé']
    logging.warning(f"Aucun personnage trouvé pour le titre : {title}")
    return ["Aucun personnage trouvé"]

def get_all_text_from_url(ebook_no: str) -> str:
    """Récupère le texte d'un livre à partir de son numéro d'ebook"""
    url = f"https://www.gutenberg.org/files/{ebook_no}/{ebook_no}-0.txt"
    try:
        response = requests.get(url)
        response.raise_for_status()
        text_lines = response.text.splitlines()[:50]
        return "\n".join(text_lines)
    except requests.exceptions.RequestException as e:
        logging.error(f"Erreur lors de la récupération du texte : {e}")
        return None

def get_all_text_book_by_title_from_url(vector_store: Chroma, title: str) -> str:
    """Récupère le texte complet d'un livre en ligne par son titre"""
    logging.info(f"Recherche du texte en ligne pour : {title}")
    results = search_in_vector_store(vector_store, title)
    if results:
        ebook_no = results[0].metadata.get(EBOOK_NO_KEY)
        if ebook_no:
            return get_all_text_from_url(ebook_no)
    logging.warning(f"Aucun texte trouvé pour le titre : {title}")
    return "Texte non trouvé"