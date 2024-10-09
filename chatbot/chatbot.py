import requests
import re
import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from embedding_model import EmbeddingModel
from language_model import LanguageModel

loader = CSVLoader("../data/gutenberg2.csv")
docs = loader.load()

data = []
for doc in docs:
    content = doc.page_content.split("\n")
    metadata = {line.split(": ")[0].strip(): line.split(": ")[1].strip() for line in content if ": " in line}
    data.append(metadata)

df = pd.DataFrame(data)
print(df.head())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_documents(docs)

embedding_model_instance = EmbeddingModel()
embedding_model = embedding_model_instance.get_embedding_model()

persist_directory = '../model/chroma/'
vectordb = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=persist_directory)

def find_author(title):
    row = df[df['Title'].str.contains(title, case=False)]
    if not row.empty:
        return row.iloc[0]['Author']
    return "Auteur non trouvé."

def get_subject(title):
    row = df[df['Title'].str.contains(title, case=False)]
    if not row.empty:
        return row.iloc[0]['Subject']
    return "Sujet non trouvé."

def extract_characters(title):
    row = df[df['Title'].str.contains(title, case=False)]
    if not row.empty:
        summary = row.iloc[0]['Summary']
        characters = set(word for word in summary.split() if word.istitle())
        return list(characters) if characters else ["Aucun personnage cité."]
    return ["Aucun personnage cité."]

def fetch_full_text(title):
    ebook_no_row = df[df['Title'].str.contains(title, case=False)]
    if not ebook_no_row.empty:
        ebook_no = ebook_no_row.iloc[0]['EBook-No.']
        url = f"https://www.gutenberg.org/files/{ebook_no}/{ebook_no}-0.txt"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            return f"Erreur lors de la récupération du texte : {e}"
    return "Livre non trouvé."

tools = {
    "find_author": "Trouvez l'auteur du livre.",
    "get_subject": "Donnez le sujet du livre.",
    "extract_characters": "Extraire les personnages du résumé.",
    "fetch_full_text": "Récupérer le texte complet d'un livre."
}

prompt = ChatPromptTemplate.from_messages([
    ("human", "Vous êtes un assistant intelligent et bien informé sur les ouvrages du projet Gutenberg. "
               "L'utilisateur a posé la question suivante : '{user_input}'. "
               "Vous avez accès aux outils suivants : {tools}. "
               "Veuillez déterminer l'action appropriée à prendre et extraire le titre du livre mentionné dans la question si possible.")
])

llm = LanguageModel().get_language_model()

def handle_user_query(user_input):
    action_chain = prompt | llm | StrOutputParser()
    action = action_chain.invoke({"user_input": user_input, "tools": list(tools.keys())})

    title_extraction_chain = prompt | llm | StrOutputParser()
    extracted_title = title_extraction_chain.invoke({"user_input": user_input})

    match = re.search(r'`(\w+)`', action)
    return match.group(1), extracted_title if match else (None, None)

last_title = None

while True:
    user_input = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
    if user_input.lower() == 'exit':
        break

    action, extracted_title = handle_user_query(user_input)

    if action in tools:
        result = globals()[action](extracted_title)
        print(result)
    else:
        print(f"Action non reconnue : '{action}'")