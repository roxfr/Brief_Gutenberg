import os
import logging
import pandas as pd
from langchain_chroma import Chroma
from models_llm.model import create_vector_store
from utils.tools import (
	get_books_by_author,
	get_author_by_title,
	get_subject_by_title,
	get_characters_by_title,
    get_all_text_from_url,
	get_all_text_book_by_title_from_url
)

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

def test_vector_store(vector_store: Chroma):
    """Fonction pour tester les fonctionnalités du magasin de vecteurs"""
    try:
        # Test de la recherche par auteur
        author = "john doe"
        books_by_author = get_books_by_author(vector_store, author)
        print(f"Livres par {author}: {books_by_author}")

        # Test de la recherche de l'auteur par titre
        title = "give liberty give death"
        author_by_title = get_author_by_title(vector_store, title)
        print(f"Auteur de '{title}': {author_by_title}")

        # Test de la recherche du sujet par titre
        subject_by_title = get_subject_by_title(vector_store, title)
        print(f"Sujet de '{title}': {subject_by_title}")

        # Test de l'extraction des personnages par titre
        characters_by_title = get_characters_by_title(vector_store, title)
        print(f"Personnages de '{title}': {characters_by_title}")

        # Test de la récupération de texte depuis une URL
        ebook_no = "12345"  # Remplacez par un numéro d'ebook valide
        text = get_all_text_from_url(ebook_no)
        print(f"Texte du livre (EBook-No {ebook_no}):\n{text}")

        # Test de la récupération de texte complet par titre
        title = "moby-dick whale"
        full_text = get_all_text_book_by_title_from_url(vector_store, title)
        print(f"Texte complet de '{title}':\n{full_text}")

    except Exception as e:
        logging.error(f"Erreur lors du test des fonctions : {e}")

if __name__ == "__main__":
    csv_file_path = "data\gutenberg_cleaned.csv"
    data = load_csv(csv_file_path)
    vector_store = create_vector_store(data, "data\chroma_langchain_db")
    test_vector_store(vector_store)