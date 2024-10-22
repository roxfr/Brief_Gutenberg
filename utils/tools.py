import os
import logging
import requests
import pandas as pd
import time
from langchain_chroma import Chroma


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

synonyms = {
    "auteur": ["auteur", "écrivain", "compositeur", "rédacteur"],
    "sujet": ["sujet", "thème", "sujet principal", "question", "objet"],
    "personnages": ["personnages", "protagonistes", "acteurs"],
    "texte": ["texte complet", "texte intégral", "document complet"]
}

def process_question(question: str, vector_store) -> str:
    """Identifie la question et retourne la réponse appropriée"""
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

def find_in_dataframe(title: str, data: pd.DataFrame, column_name: str) -> str:
    """Recherche une valeur dans le DataFrame par titre"""
    row = data[data['Title'].str.contains(title, case=False)]
    return row.iloc[0][column_name] if not row.empty else None

def get_author(vector_store: Chroma, title: str) -> str:
    """Recherche l'auteur d'un livre par son titre"""
    logging.info(f"Recherche de l'auteur pour : {title}")
    start_time = time.time()
    try:
        results = vector_store.similarity_search(title, k=1)
        end_time = time.time()
        logging.info(f"Temps de recherche : {end_time - start_time:.2f} secondes")
        if results:
            author_parts = results[0].metadata.get('author', [])
            return " ".join(author_parts) if author_parts else None
    except Exception as e:
        logging.error(f"Erreur lors de la recherche dans le vector store : {e}")
    return None

def get_subject(vector_store: Chroma, title: str) -> str:
    """Retourne le sujet d'un livre par son titre"""
    logging.info(f"Recherche du sujet pour : {title}")
    start_time = time.time()
    try:
        results = vector_store.similarity_search(title, k=1)
        end_time = time.time()
        logging.info(f"Temps de recherche : {end_time - start_time:.2f} secondes")
        if results:
            subject_parts = results[0].metadata.get('subject', [])
            return ", ".join(subject_parts) if subject_parts else None
    except Exception as e:
        logging.error(f"Erreur lors de la recherche dans le vector store : {e}")
    return None

def get_characters(vector_store: Chroma, title: str) -> list:
    """Extrait les personnages d'un livre par son titre"""
    logging.info(f"Recherche du/des personnage(s) pour : {title}")
    start_time = time.time()
    try:
        results = vector_store.similarity_search(title, k=1)
        end_time = time.time()
        logging.info(f"Temps de recherche : {end_time - start_time:.2f} secondes")
        if results:
            summary = results[0].metadata.get('summary', "")
            characters = {word for word in summary.split() if word.istitle()}
            return list(characters) if characters else None
    except Exception as e:
        logging.error(f"Erreur lors de la recherche dans le vector store : {e}")
    return None

def get_text_from_url(ebook_no: str) -> str:
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

def get_full_text_from_url(vector_store: Chroma, title: str) -> str:
    """Récupère le texte complet d'un livre en ligne par son titre"""
    logging.info(f"Recherche du texte en ligne pour : {title}")
    start_time = time.time()
    try:
        results = vector_store.similarity_search(title, k=1)
        end_time = time.time()
        logging.info(f"Temps de recherche : {end_time - start_time:.2f} secondes")
        if results:
            ebook_no = results[0].metadata.get('ebook_no')
            if ebook_no:
                return get_text_from_url(ebook_no)
    except Exception as e:
        logging.error(f"Erreur lors de la recherche dans le vector store : {e}")
    return None