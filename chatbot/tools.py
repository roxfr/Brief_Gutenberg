import logging
import requests
import pandas as pd
import time
from langchain_chroma import Chroma


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

def fetch_text_from_url(ebook_no: str) -> str:
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
                return fetch_text_from_url(ebook_no)
    except Exception as e:
        logging.error(f"Erreur lors de la recherche dans le vector store : {e}")
    return None