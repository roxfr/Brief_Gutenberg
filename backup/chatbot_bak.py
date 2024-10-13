import requests
import re
import pandas as pd
import logging
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from embedding_model import EmbeddingModel
from language_model import LanguageModel
from config import load_config

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Chargement de la configuration
config = load_config()
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
CHROMA_PATH = config["CHROMA_PATH"]

# Chargement des documents
loader = CSVLoader(CSV_CLEANED_PATH)
docs = loader.load()

# Préparation des données pour le DataFrame
data = []
for doc in docs:
    content = doc.page_content.split("\n")
    metadata = {line.split(": ")[0].strip(): line.split(": ")[1].strip() for line in content if ": " in line}
    data.append(metadata)

df = pd.DataFrame(data)
logging.info("Documents chargés et DataFrame créé.")

# Division des documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_documents(docs)

# Initialisation du modèle d'embedding
embedding_model_instance = EmbeddingModel()
embedding_model = embedding_model_instance.get_embedding_model()

persist_directory = CHROMA_PATH
vectordb = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=persist_directory)
logging.info("Modèle d'embedding initialisé et Chroma créé.")

def find_author(title):
    """Recherche l'auteur d'un livre par son titre."""
    logging.info(f"Recherche de l'auteur pour : {title}")
    row = df[df['Title'].str.contains(title, case=False)]
    if not row.empty:
        return row.iloc[0]['Author']
    return "Auteur non trouvé."

def get_subject(title):
    """Retourne le sujet d'un livre par son titre."""
    logging.info(f"Recherche du sujet pour : {title}")
    row = df[df['Title'].str.contains(title, case=False)]
    if not row.empty:
        return row.iloc[0]['Subject']
    return "Sujet non trouvé."

def extract_characters(title):
    """Extrait les personnages d'un livre par son titre à partir du résumé."""
    logging.info(f"Recherche du/des personnage(s) pour : {title}")
    row = df[df['Title'].str.contains(title, case=False)]
    if not row.empty:
        summary = row.iloc[0]['Summary']
        characters = set(word for word in summary.split() if word.istitle())
        return list(characters) if characters else ["Aucun personnage cité."]
    return ["Aucun personnage cité."]

def fetch_full_text(title):
    """Récupère le texte complet d'un livre en ligne par son titre."""
    logging.info(f"Récupération du texte en ligne pour : {title}")
    ebook_no_row = df[df['Title'].str.contains(title, case=False)]
    if not ebook_no_row.empty:
        ebook_no = ebook_no_row.iloc[0]['EBook-No.']
        url = f"https://www.gutenberg.org/files/{ebook_no}/{ebook_no}-0.txt"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logging.error(f"Erreur lors de la récupération du texte : {e}")
            return f"Erreur lors de la récupération du texte : {e}"
    return "Livre non trouvé."

# Définition des outils
tools = {
    "find_author": "Trouvez l'auteur du livre.",
    "get_subject": "Donnez le sujet du livre.",
    "extract_characters": "Extraire les personnages du résumé.",
    "fetch_full_text": "Récupérer le texte complet d'un livre."
}

# Création du prompt pour le modèle
prompt = ChatPromptTemplate.from_messages([
    ("human", "Vous êtes un assistant intelligent et bien informé sur les ouvrages du projet Gutenberg. "
               "L'utilisateur a posé la question suivante : '{user_input}'. "
               "Vous avez accès aux outils suivants : {tools}. "
               "Veuillez déterminer l'action appropriée à prendre (Action :) et \
                extraire le titre du livre mentionné dans la question si possible (Titre :).")
])

llm = LanguageModel().get_language_model()

def handle_user_query(user_input):
    """Gère les requêtes de l'utilisateur et extrait l'action et le titre."""
    action_chain = prompt | llm | StrOutputParser()
    action_output = action_chain.invoke({"user_input": user_input, "tools": list(tools.keys())})

    action_match = re.search(r'Action : (\w+)', action_output)
    title_match = re.search(r'Titre : (.+)', action_output)

    if action_match:
        action = action_match.group(1)
        logging.info(f"Action : {action}")
    else:
        logging.warning("Action non trouvée dans le prompt.")
        return None, None

    if title_match:
        cleaned_title = title_match.group(1).strip()
    else:
        logging.warning("Titre non trouvé dans la sortie.")
        return action, None

    logging.info(f"Titre du livre : '{cleaned_title}'")
    return action, cleaned_title

# Boucle principale pour l'interaction utilisateur
while True:
    user_input = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
    if user_input.lower() == 'exit':
        break

    if len(user_input.split()) <= 7:
        print("Veuillez poser une question de plus de 7 mots pour que je puisse vous aider.")
        continue

    action, cleaned_title = handle_user_query(user_input)

    if action in tools:
        result = globals()[action](cleaned_title)
        print(result)
    else:
        logging.warning(f"Action non reconnue : '{action}'")