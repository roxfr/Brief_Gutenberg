import logging
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from utils.tools import load_csv
from models.model import create_llama, create_vector_store
from utils.config import load_config
from utils.qa_chain import setup_qa_chain


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
CHROMA_PATH = config["CHROMA_PATH"]
MODEL_PATH = config["MODEL_PATH"]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestionnaire de durée de vie pour initialiser et nettoyer les ressources."""
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
    """Route pour poser une question au LLM."""
    try:
        question = question.lower()
        response = app.state.qa_chain.invoke({"question": question})
        if response:
            return {"response": response or "Je ne sais pas."}
        return {"response": "Je ne sais pas."}
    except Exception as e:
        logger.error(f"Erreur lors de la réponse à la question : {e}")
        return {"error": "Erreur interne du serveur"}

@app.get("/test")
async def test_route():
    """Route de test pour vérifier le bon fonctionnement du serveur."""
    return {"message": "Le serveur fonctionne correctement"}

if __name__ == "__main__":
    logger.info("Démarrage du serveur FastAPI...")
    uvicorn.run(app, host="127.0.0.1", port=8000)