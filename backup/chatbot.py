import logging
from models_llm.model import create_llama, create_vector_store
from utils.qa_chain import setup_qa_chain
from utils.tools import load_csv
from utils.config import load_config


config = load_config()
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
CHROMA_PATH = config["CHROMA_PATH"]
MODEL_PATH = config["MODEL_PATH"]

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    data = load_csv(CSV_CLEANED_PATH)
    llm = create_llama(MODEL_PATH)
    vector_store = create_vector_store(data, CHROMA_PATH)
    qa_chain = setup_qa_chain(llm, vector_store)

    while True:
        question = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
        question = question.lower()
        if question == 'exit':
            break
        response = qa_chain.invoke({"question": question})
        print(f"Réponse : {response}")

if __name__ == "__main__":
    main()