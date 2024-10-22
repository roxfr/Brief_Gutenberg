import logging
from models.model import create_llama, create_vector_store
from utils.qa_chain import setup_qa_chain
from utils.tools import load_csv
from utils.config import load_config
from utils.tools import find_in_dataframe, get_author, get_subject, get_characters, get_full_text_from_url

config = load_config()
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
CHROMA_PATH = config["CHROMA_PATH"]
MODEL_PATH = config["MODEL_PATH"]

tools = {
            "find_in_dataframe": find_in_dataframe,
            "get_author": get_author,
            "get_subject": get_subject,
            "get_characters": get_characters,
            "get_full_text_from_url": get_full_text_from_url,
        }

def main():
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    data = load_csv(CSV_CLEANED_PATH)
    llm = create_llama(MODEL_PATH)
    vector_store = create_vector_store(data, CHROMA_PATH)
    qa_chain = setup_qa_chain(llm, vector_store)

    while True:
        question = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
        if question.lower() == 'exit':
            break
        
        response = qa_chain.invoke({"question": question, "tools": tools})
        print(f"Réponse : {response}")

if __name__ == "__main__":
    main()