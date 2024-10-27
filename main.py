import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from utils.tools import load_csv
from models.model import create_llama, create_vector_store
from utils.config import load_config
from utils.qa_chain import setup_qa_chain
from utils.tools import (
	get_books_by_author,
	get_author_by_title,
	get_subject_by_title,
	get_characters_by_title,
	get_all_text_book_by_title_from_url
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
CHROMA_PATH = config["CHROMA_PATH"]
MODEL_PATH = config["MODEL_PATH"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de durée de vie pour initialiser et nettoyer les ressources"""
    logger.info("Initialisation des ressources...")
    logger.info("Llama où t'es ? 🦙")
    app.state.llm = create_llama(MODEL_PATH)
    app.state.vector_store = create_vector_store(load_csv(CSV_CLEANED_PATH), CHROMA_PATH)
    app.state.qa_chain = setup_qa_chain(app.state.llm, app.state.vector_store)

    yield

    logger.info("Fermeture des ressources...")

app = FastAPI(lifespan=lifespan)

@app.post("/ask/")
async def ask_question(question: str):
    """Route pour poser une question au LLM"""
    try:
        question = question.lower()
        response = app.state.qa_chain.invoke({"question": question})
        if response:
            return {"response": response or "Je ne sais pas. 😳"}
        return {"response": "Je ne sais pas. 😳"}
    except Exception as e:
        logger.error(f"Erreur lors de la réponse à la question : {e}")
        return {"error": "Erreur interne du serveur 😕"}

@app.post("/author/")
async def author_by_title(title: str):
    """Route pour trouver l'auteur d'un livre par son titre"""
    try:
        response = get_author_by_title(app.state.vector_store, title.lower())
        return {"author": response}
    except Exception as e:
        logger.error(f"Erreur lors de la recherche de l'auteur : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur 😕")

@app.post("/books/")
async def books_by_author(author: str):
    """Route pour récupérer les livres écrits par un auteur spécifique"""
    try:
        response = get_books_by_author(app.state.vector_store, author.lower())
        return {"books": response}
    except Exception as e:
        logger.error(f"Erreur lors de la recherche des livres : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur 😕")

@app.post("/subject/")
async def subject_by_title(title: str):
    """Route pour retourner le sujet d'un livre par son titre"""
    try:
        response = get_subject_by_title(app.state.vector_store, title.lower())
        return {"subject": response}
    except Exception as e:
        logger.error(f"Erreur lors de la recherche du sujet : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur 😕")

@app.post("/characters/")
async def characters_by_title(title: str):
    """Route pour extraire les personnages d'un livre par son titre"""
    try:
        response = get_characters_by_title(app.state.vector_store, title.lower())
        return {"characters": response}
    except Exception as e:
        logger.error(f"Erreur lors de la recherche des personnages : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur 😕")

@app.post("/all_text/")
async def all_text_book_by_title_from_url(title: str):
    """Route pour récupérer le texte complet d'un livre en ligne par son titre"""
    try:
        response = get_all_text_book_by_title_from_url(app.state.vector_store, title.lower())
        return {"text": response}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du texte : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur 😕")

@app.get("/test")
async def test_route():
    """Route de test pour vérifier le bon fonctionnement du serveur"""
    return {"message": "Le serveur fonctionne correctement 😉"}

if __name__ == "__main__":
    logger.info("Démarrage du serveur FastAPI...")
    uvicorn.run(app, host="127.0.0.1", port=8000)