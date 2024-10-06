import logging
from uuid import uuid4
from utils import (
    load_embeddings,
    load_vector_store,
    load_docs,
    save_docs,
    load_data,
    split_documents,
    create_vector_store,
    configure_qa_chain
)
from embedding_model import EmbeddingModel
from language_model import LanguageModel
from config import load_config
config = load_config()
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
EMBEDDINGS_FILE = config["EMBEDDINGS_FILE"]
DOCS_FILE = config["DOCS_FILE"]
FAISS_INDEX = config["FAISS_INDEX"]
CHUNK_SIZE = config["CHUNK_SIZE"]

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_chat_loop(qa_chain, docs):
    while True:
        user_input = input("Votre question (ou tapez 'exit' pour quitter) : ")
        if user_input.lower() == 'exit':
            break
        if len(user_input.strip()) < 5:
            print("Veuillez poser une question plus détaillée.")
            continue
        context = "\n".join([doc['content'] for doc in docs]) if docs else ""
        response = qa_chain.invoke({"query": user_input, "context": context})
        print("Réponse :", response.get('output', "Aucune réponse générée."))

def main():
    data = load_data(CSV_CLEANED_PATH)
    all_splits = split_documents(data, CHUNK_SIZE)    
    embedding_model = EmbeddingModel().get_embedding_model()
    embeddings = load_embeddings(EMBEDDINGS_FILE)
    vectorstore = load_vector_store(embedding_model, FAISS_INDEX)
    if vectorstore is None:
        vectorstore = create_vector_store(all_splits, embedding_model, EMBEDDINGS_FILE, FAISS_INDEX, DOCS_FILE)
    else:
        logger.info("Magasin de vecteurs chargé avec succès.")
    docs = load_docs(DOCS_FILE)
    if docs is None or len(docs) == 0:
        logger.warning("Aucun document trouvé. Régénération des documents...")
        docs = [{'id': str(uuid4()), 'content': split.page_content} for split in all_splits]
        save_docs(docs, DOCS_FILE)
        logger.info("Documents régénérés avec succès.")
    llm_model = LanguageModel().get_language_model()
    qa_chain = configure_qa_chain(llm_model, vectorstore)
    run_chat_loop(qa_chain, docs)

if __name__ == "__main__":
    main()
