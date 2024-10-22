# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# http://localhost:8000/docs
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from utils.tools import load_csv
from models.model import create_llama, create_vector_store
from utils.config import load_config
from utils.qa_chain import setup_qa_chain
from utils.tools import find_in_dataframe, get_author, get_subject, get_characters, get_full_text_from_url

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
CHROMA_PATH = config["CHROMA_PATH"]
MODEL_PATH = config["MODEL_PATH"]

app = FastAPI()

# Modèle LLM et vector store
llm = create_llama(MODEL_PATH)
vector_store = create_vector_store(load_csv(CSV_CLEANED_PATH), CHROMA_PATH)
qa_chain = setup_qa_chain(llm, vector_store)

@app.post("/ask/")
def ask_question(question: str):
    """Route pour poser une question au LLM"""
    try:
        tools = {
            "find_in_dataframe": find_in_dataframe,
            "get_author": get_author,
            "get_subject": get_subject,
            "get_characters": get_characters,
            "get_full_text_from_url": get_full_text_from_url
        }
        response = qa_chain.invoke({"question": question, "tools": tools})
        if response:
            return {"response": response}
        return {"response": "Je ne sais pas."}
    except Exception as e:
        logger.error(f"Erreur lors de la réponse à la question : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

@app.get("/test")
def test_route():
    """Route de test pour vérifier le bon fonctionnement du serveur"""
    return {"message": "Le serveur fonctionne correctement"}

@app.exception_handler(Exception)
async def generic_exception_handler(exc):
    """Gérer les exceptions générales"""
    logger.error(f"Une erreur s'est produite : {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Une erreur interne s'est produite"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)