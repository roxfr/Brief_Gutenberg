import pandas as pd
import requests
import logging
import re
#from langchain_community.document_loaders import CSVLoader
from embedding_model import EmbeddingModel
from language_model import LanguageModel
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from config import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = load_config()
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]

try:
    df = pd.read_csv(CSV_CLEANED_PATH, delimiter=';', encoding='utf-8-sig')
except Exception as e:
    logging.error(f"Erreur lors de la lecture du fichier CSV : {e}")
    raise
print(df.head())
documents = [
    Document(
        page_content=row['Summary'],
        metadata={
            'Author': row['Author'],
            'Title': row['Title'],
            'Subject': row['Subject'],
            'EBook-No.': row['EBook-No.'],
            'Release Date': row['Release Date']
        }
    )
    for _, row in df.iterrows()
]
print("Documents chargés avec succès.")

vectorstore = FAISS.from_documents(documents, EmbeddingModel().get_embedding_model())
print("Vectorstore initialisé.")

chain = ConversationalRetrievalChain.from_llm(
    llm=LanguageModel().get_language_model(),
    retriever=vectorstore.as_retriever()
)
print("Chaîne de conversation créée.")

def find_author(title):
    logging.info(f"Recherche de l'auteur pour : {title}")
    response = chain.invoke({"question": f"Who is the author of '{title}'?"})
    return response['answer'] or "Auteur non trouvé."

def get_subject(title):
    logging.info(f"Recherche du sujet pour : {title}")
    response = chain.invoke({"question": f"What is the subject of '{title}'?"})
    return response['answer'] or "Sujet non trouvé."

def extract_characters(title):
    logging.info(f"Recherche du/des personnage(s) pour : {title}")
    response = chain.invoke({"question": f"Who are the characters in '{title}'?"})
    return response['answer'] or ["Aucun personnage cité."]

def fetch_full_text(title):
    logging.info(f"Récupération du texte en ligne pour : {title}")
    response = chain.invoke({"question": f"Get the full text of '{title}'"})
    
    if response['answer']:
        return response['answer']
    
    ebook_no_row = [doc for doc in documents if title.lower() in doc.get('page_content', '').lower()]
    if ebook_no_row:
        ebook_no = ebook_no_row[0]['EBook-No.']
        url = f"https://www.gutenberg.org/files/{ebook_no}/{ebook_no}-0.txt"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Erreur lors de la récupération du texte : {e}")
            return f"Erreur lors de la récupération du texte : {e}"
    return "Livre non trouvé."

prompt = ChatPromptTemplate.from_messages([
    ("human", "Vous êtes un assistant intelligent et bien informé sur les ouvrages du projet Gutenberg. "
               "L'utilisateur a posé la question suivante : '{user_input}'. "
               "Veuillez déterminer l'action appropriée à prendre et extraire le titre du livre mentionné dans la question si possible.")
])

def handle_user_query(user_input):
    action_chain = prompt | EmbeddingModel().get_embedding_model() | StrOutputParser()
    action_output = action_chain.invoke({"user_input": user_input})

    action_match = re.search(r'Action : (\w+)', action_output)
    title_match = re.search(r'Titre : (.+)', action_output)

    action = action_match.group(1) if action_match else None
    cleaned_title = title_match.group(1).strip() if title_match else None

    if action:
        logging.info(f"Action : {action}")
    else:
        logging.warning("Action non trouvée dans le prompt.")
    
    if cleaned_title:
        logging.info(f"Titre du livre : '{cleaned_title}'")
    else:
        logging.warning("Titre non trouvé dans la sortie.")
    
    return action, cleaned_title

while True:
    user_input = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
    if user_input.lower() == 'exit':
        break

    action, cleaned_title = handle_user_query(user_input)

    if action and cleaned_title:
        if action == "find_author":
            result = find_author(cleaned_title)
        elif action == "get_subject":
            result = get_subject(cleaned_title)
        elif action == "extract_characters":
            result = extract_characters(cleaned_title)
        elif action == "fetch_full_text":
            result = fetch_full_text(cleaned_title)
        else:
            logging.warning(f"Action non reconnue : '{action}'")
            result = "Action non reconnue."
    else:
        result = "Impossible de déterminer l'action ou le titre du livre."

    print("Réponse : " + result)
