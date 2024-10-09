# __init__.py


# chatbot.py
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


# clean_data.py
import pandas as pd


from config import load_config
config = load_config()
CSV_INPUT_PATH = config["CSV_INPUT_PATH"]
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]

df = pd.read_csv(CSV_INPUT_PATH, encoding="utf-8", header=0)
df.fillna({
    'Author': 'Auteur non défini',
    'Title': 'Titre non défini',
    'Credits': 'Crédits non définis',
    'Summary': 'Résumé non défini',
    'Subject': 'Sujet non défini',
    'EBook-No.': 'Numéro d\'ebook non défini',
    'Release Date': 'Date de publication non définie'
}, inplace=True)
text_columns = ['Author', 'Title', 'Summary', 'Subject', 'EBook-No.', 'Release Date']
for col in text_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r'[^\x00-\x7F]+', '', regex=True)
        .str.replace(r'[\n\r]', ' ', regex=True)
        .str.strip()
    )
df.drop(columns=['Ebook ID', 'Credits', 'Language', 'LoC Class', 
                 'Subject_2', 'Subject_3', 'Subject_4', 'Category', 
                 'Most Recently Updated', 'Copyright Status', 'Downloads'], 
        inplace=True, errors='ignore')
df.dropna(how='all', inplace=True)
df = df.head(5000)
df.to_csv(CSV_CLEANED_PATH, index=False, header=True, encoding='utf-8')

# config.py
import os
from dotenv import load_dotenv


def load_config():
    load_dotenv()
    config = {
        "CSV_INPUT_PATH": os.path.join("../datas", "gutenberg.csv"),
        "CSV_CLEANED_PATH": os.path.join("../datas", "gutenberg2.csv"),
        "EMBEDDINGS_FILE": os.path.join("../models", "embeddings.csv"),
        "DOCS_FILE": os.path.join("../models", "docs.csv"),
        "FAISS_INDEX": os.path.join("../models", "faiss_index.index"),
        "CHUNK_SIZE": int(500),
        "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY"),
        "AZURE_OPENAI_API_BASE": os.getenv("AZURE_OPENAI_API_BASE"), 
        "AZURE_OPENAI_API_ENDPOINT": os.getenv("AZURE_OPENAI_API_ENDPOINT"),
        "AZURE_DEPLOYMENT_NAME": os.getenv("AZURE_DEPLOYMENT_NAME"),
        "AZURE_API_VERSION": os.getenv("AZURE_API_VERSION"),
        "AZURE_DEPLOYEMENT": os.getenv("AZURE_DEPLOYEMENT"),
        "LANGCHAIN_ENDPOINT": os.getenv("LANGCHAIN_ENDPOINT"),
        "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY")
    }
    return config

# embedding_model.py
from langchain_openai.embeddings import AzureOpenAIEmbeddings


from config import load_config
config = load_config()
AZURE_OPENAI_API_BASE = config["AZURE_OPENAI_API_BASE"]
AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
AZURE_DEPLOYEMENT = config["AZURE_DEPLOYEMENT"]
CHUNK_SIZE = config["CHUNK_SIZE"]

class EmbeddingModel:
    def __init__(self):
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_OPENAI_API_BASE,
            openai_api_key=AZURE_OPENAI_API_KEY,
            azure_deployment=AZURE_DEPLOYEMENT,
            chunk_size=CHUNK_SIZE
        )
    def get_embedding_model(self):
        return self.embedding_model

# language_model.py
from langchain_openai import AzureChatOpenAI

from config import load_config
config = load_config()
AZURE_OPENAI_API_ENDPOINT = config["AZURE_OPENAI_API_ENDPOINT"]
AZURE_DEPLOYMENT_NAME = config["AZURE_DEPLOYMENT_NAME"]
AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
AZURE_API_VERSION = config["AZURE_API_VERSION"]

class LanguageModel:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_API_ENDPOINT,
            azure_deployment=AZURE_DEPLOYMENT_NAME,
            openai_api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_API_VERSION,
            temperature=0.0
        )
    def get_language_model(self):
        return self.llm

# utils.py
import os
import pandas as pd
import faiss
import logging
from uuid import uuid4
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_embeddings(embeddings, filepath):
    """Sauvegarde les embeddings dans un fichier CSV."""
    df = pd.DataFrame(embeddings)
    df.to_csv(filepath, index=False, header=False)
    logger.info(f"Embeddings sauvegardés dans {filepath}.")

def load_embeddings(filepath):
    """Charge les embeddings depuis un fichier CSV."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath, header=None)
        logger.info(f"Embeddings chargés depuis {filepath}.")
        return df.values.tolist()
    else:
        logger.warning("Fichier d'embeddings non trouvé.")
        return []

def save_docs(docs, filepath):
    """Sauvegarde les documents dans un fichier CSV."""
    df = pd.DataFrame(docs)
    df.to_csv(filepath, index=False, header=['id', 'content'])
    logger.info(f"Documents sauvegardés dans {filepath}.")

def load_docs(filepath):
    """Charge les documents depuis un fichier CSV."""
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        logger.info(f"Documents chargés depuis {filepath}.")
        return df.to_dict(orient='records')
    else:
        logger.warning("Fichier de documents non trouvé.")
        return None

def save_vector_store(vector_store, filepath):
    """Sauvegarde le magasin de vecteurs dans un fichier."""
    faiss.write_index(vector_store.index, filepath)
    logger.info(f"Magasin de vecteurs sauvegardé dans {filepath}.")

def load_vector_store(embedding_model, filepath):
    """Charge le magasin de vecteurs depuis un fichier."""
    if os.path.exists(filepath):
        index = faiss.read_index(filepath)
        docstore = InMemoryDocstore()
        index_to_docstore_id = {}
        vector_store = FAISS(
            embedding_function=embedding_model.embed_query,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )        
        logger.info(f"Magasin de vecteurs chargé depuis {filepath}.")
        return vector_store
    else:
        logger.warning("Répertoire de sauvegarde non trouvé.")
        return None

def load_data(csv_path):
    loader = CSVLoader(file_path=csv_path, encoding='utf-8')
    return loader.load()

def split_documents(data, chunk_size):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    return text_splitter.split_documents(data)

def create_vector_store(all_splits, embedding_model, embeddings_file, faiss_index, docs_file):
    """Crée un magasin de vecteurs à partir des documents divisés."""
    index = faiss.IndexFlatL2(len(embedding_model.embed_query("test")))
    docstore = InMemoryDocstore()
    index_to_docstore_id = {}
    vectorstore = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )
    logger.info("Création du magasin de vecteurs...")
    embeddings = []    
    for i, split in enumerate(all_splits):
        if hasattr(split, 'page_content'):
            embedding = embedding_model.embed_query(split.page_content)
            embeddings.append(embedding)
            doc_id = str(uuid4())
            vectorstore.add_documents([split], ids=[doc_id])
            index_to_docstore_id[doc_id] = i
            logger.info(f"Document ajouté : ID = {doc_id}, Contenu = {split.page_content[:50]}...")
        else:
            logger.warning(f"Le split à l'index {i} ne possède pas l'attribut 'page_content'.")
    if embeddings:
        flat_embeddings = [e for embedding in embeddings for e in embedding]
        save_embeddings(flat_embeddings, embeddings_file)
    save_vector_store(vectorstore, faiss_index)
    save_docs(all_splits, docs_file)
    logger.info("Magasin de vecteurs créé avec succès.")
    return vectorstore

def configure_qa_chain(llm_model, vectorstore):
    return RetrievalQA.from_chain_type(llm=llm_model, retriever=vectorstore.as_retriever())

