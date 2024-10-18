import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from utils.tools import get_author, get_subject, get_characters, get_full_text_from_url
from chatbot import load_books_data, download_book, extract_characters, answer_questions, initialize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

data = load_books_data()
retriever, embeddings = initialize()

@app.get("/characters/{book_id}")
def get_characters_api(book_id: int):
    """Récupère les personnages d'un livre donné par son ID."""
    try:
        book_text = download_book(book_id)
        characters = extract_characters(book_text)
        return {"characters": characters}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des personnages: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des personnages.")

@app.get("/text/{book_id}")
def get_text(book_id: int, num_lines: int = 10):
    """Récupère les premières lignes d'un livre donné par son ID."""
    try:
        book_text = download_book(book_id)
        lines = "\n".join(book_text.splitlines()[:num_lines])
        return {"book_id": book_id, "lines": lines}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du texte: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération du texte.")

@app.post("/question/")
def answer_question(question: str):
    """Répond à une question basée sur les résumés des livres."""
    try:
        response = retriever.run(question)
        return {"response": response}
    except Exception as e:
        logger.error(f"Erreur lors de la réponse à la question: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la réponse à la question.")

@app.post("/full_text_question/")
def answer_full_text_question(book_id: int, question: str):
    """Répond à une question basée sur le texte complet d'un livre."""
    try:
        book_text = download_book(book_id)
        response = answer_questions(book_text, question)
        return {"response": response}
    except Exception as e:
        logger.error(f"Erreur lors de la réponse à la question sur le texte complet: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la réponse à la question.")

@app.get("/author/{title}")
def get_author_api(title: str):
    """Récupère l'auteur d'un livre par son titre."""
    try:
        author = get_author(embeddings, title)
        return {"title": title, "author": author}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de l'auteur: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération de l'auteur.")

@app.get("/subject/{title}")
def get_subject_api(title: str):
    """Récupère le sujet d'un livre par son titre."""
    try:
        subject = get_subject(embeddings, title)
        return {"title": title, "subject": subject}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du sujet: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération du sujet.")

@app.get("/characters_by_title/{title}")
def get_characters_by_title(title: str):
    """Récupère les personnages d'un livre par son titre."""
    try:
        characters = get_characters(embeddings, title)
        return {"title": title, "characters": characters}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des personnages par titre: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des personnages par titre.")

@app.get("/full_text/{title}")
def get_full_text(title: str):
    """Récupère le texte complet d'un livre par son titre."""
    try:
        full_text = get_full_text_from_url(embeddings, title)
        return {"title": title, "full_text": full_text}
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du texte complet: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération du texte complet.")

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