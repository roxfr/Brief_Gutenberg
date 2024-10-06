#http://localhost:8000/characters/{book_id}
#http://localhost:8000/text/{book_id}
#http://localhost:8000/question/
#http://localhost:8000/full_text_question/
#http://localhost:8000/test
#http://localhost:8000/docs

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from chatbots.chatbot_lch import (load_books_data, create_vector_store, 
                                    download_book, extract_characters, 
                                    answer_questions, initialize)

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialisation de FastAPI
app = FastAPI()

# Initialisation du chatbot
retriever, embeddings = initialize()

@app.get("/characters/{book_id}")
def get_characters(book_id: int):
    """Récupère les personnages d'un livre donné par son ID."""
    book_text = download_book(book_id)
    characters = extract_characters(book_text)
    return {"characters": characters}

@app.get("/text/{book_id}")
def get_text(book_id: int, num_lines: int = 10):
    """Récupère les premières lignes d'un livre donné par son ID."""
    book_text = download_book(book_id)
    lines = "\n".join(book_text.splitlines()[:num_lines])
    return {"book_id": book_id, "lines": lines}

@app.post("/question/")
def answer_question(question: str):
    """Répond à une question basée sur les résumés des livres."""
    response = retriever.run(question)
    return {"response": response}

@app.post("/full_text_question/")
def answer_full_text_question(book_id: int, question: str):
    """Répond à une question basée sur le texte complet d'un livre."""
    book_text = download_book(book_id)
    response = answer_questions(book_text, question)
    return {"response": response}

@app.get("/test")
def test_route():
    """Route de test pour vérifier le bon fonctionnement du serveur."""
    return {"message": "Le serveur fonctionne correctement."}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Gérer les exceptions HTTP."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"message": exc.detail}
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    """Gérer les exceptions générales."""
    logger.error(f"Une erreur s'est produite: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Une erreur interne s'est produite."}
    )

@app.get("/404")
async def not_found_route():
    raise HTTPException(status_code=404, detail="Page non trouvée.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
