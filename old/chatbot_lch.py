import logging
import numpy as np
import os
import pandas as pd
import pickle
import spacy
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DocArrayInMemorySearch

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Chargement des variables d'environnement
load_dotenv()

# Définition des constantes
MODEL_DIR = '../models'
EMBEDDINGS_PATH = os.path.join(MODEL_DIR, 'embeddings.pkl')
VECTOR_STORE_PATH = os.path.join(MODEL_DIR, 'vector_store.pkl')
MAX_LINES = 5000
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

# Chargement du modèle de langue spaCy
nlp = spacy.load("fr_core_news_sm")

# Initialisation du modèle d'embeddings
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment='text-embedding-ada-002',
    chunk_size=1
)

# Initialisation du modèle LLM
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_DEPLOYMENT_NAME"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    temperature=0.0
)

def load_books_data():
    """Charge les données des livres depuis un fichier CSV et remplace les valeurs manquantes."""
    try:
        df = pd.read_csv('../datas/gutenberg.csv', encoding="utf-8")
    except FileNotFoundError:
        logger.error("Le fichier CSV n'a pas été trouvé.")
        return None
    
    df.fillna({
        'Ebook ID': 'ID non défini',
        'Author': 'Auteur non défini',
        'Title': 'Titre non défini',
        'Credits': 'Crédits non définis',
        'Summary': 'Résumé non défini',
        'Language': 'Langue non définie',
        'LoC Class': 'Classe LoC non définie',
        'Subject': 'Sujet non défini',
        'Subject_2': 'Deuxième sujet non défini',
        'Subject_3': 'Troisième sujet non défini',
        'Subject_4': 'Quatrième sujet non défini',
        'Category': 'Catégorie non définie',
        'EBook-No.': 'Numéro d\'ebook non défini',
        'Release Date': 'Date de publication non définie',
        'Most Recently Updated': 'Dernière mise à jour non définie',
        'Copyright Status': 'Statut de copyright non défini',
        'Downloads': 'Téléchargements non définis'
    }, inplace=True)

    text_columns = ['Author', 'Title', 'Summary', 'Language', 'LoC Class', 'Subject', 
                    'Subject_2', 'Subject_3', 'Subject_4', 'Category', 'Release Date', 
                    'Most Recently Updated', 'Copyright Status']
    
    for col in text_columns:
        df[col] = df[col].astype(str).str.replace('[^\x00-\x7F]+', '', regex=True).str.strip()

    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    logger.info(f"Taille du DataFrame : {df.shape}")
    return df

def load_embeddings():
    """Charge les embeddings à partir d'un fichier si disponible."""
    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            embeddings, docs = pickle.load(f)
        logger.info("Embeddings chargés avec succès.")
        return embeddings, docs
    return None, None

def generate_embeddings(docs):
    """Génère des embeddings pour les documents fournis."""
    all_embeddings = []
    total_docs = min(len(docs), MAX_LINES)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, 
                                                   chunk_overlap=DEFAULT_CHUNK_OVERLAP)

    for i, doc in enumerate(docs[:MAX_LINES]):
        logger.info(f"Génération de l'embedding pour le document : {doc.metadata['title']} ({i + 1}/{total_docs})")
        
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            embedding = embedding_model.embed_documents([chunk])[0]
            
            if not isinstance(embedding, list) or len(embedding) == 0 or not all(isinstance(x, float) for x in embedding):
                logger.error(f"L'embedding pour le document {doc.metadata['title']} est au mauvais format.")
                raise ValueError("L'embedding doit être une liste de flottants non vide.")
           
            all_embeddings.append(embedding)

    embeddings_array = np.array(all_embeddings).astype('float32')
    if embeddings_array.ndim != 2 or embeddings_array.shape[0] != len(all_embeddings):
        logger.error(f"Les embeddings ont le format {embeddings_array.shape}, attendu : (nombre de documents, dimension des embeddings).")
        raise ValueError("Les embeddings doivent être un tableau 2D.")

    logger.info(f"Total d'embeddings générés : {len(all_embeddings)}")
    return all_embeddings

def save_vector_store(vector_store, docs, embeddings):
    """Sauvegarde le magasin de vecteurs et les embeddings dans le dossier 'models'."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(VECTOR_STORE_PATH, 'wb') as f:
        pickle.dump((vector_store, docs), f)

    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(embeddings, f)

    logger.info("Vector store et embeddings sauvegardés avec succès.")

def create_vector_store(df):
    """Crée un magasin de vecteurs à partir des données des livres."""
    embeddings, docs = load_embeddings()
    if embeddings is not None and docs is not None:
        logger.info("Chargement du VectorStore à partir des embeddings existants.")
        return DocArrayInMemorySearch(embeddings, docs)
    
    limited_df = df.head(MAX_LINES)

    docs = []
    for _, row in limited_df.iterrows():
        if isinstance(row['Summary'], str) and row['Summary']:
            limited_summary = "\n".join(row['Summary'].splitlines()[:MAX_LINES])
            docs.append(Document(page_content=limited_summary, 
                                 metadata={"title": row['Title'], "ebook_no": row['EBook-No.']}))


    if not docs:
        logger.error("Aucun document à ajouter au VectorStore.")
        raise ValueError("Aucun document à ajouter au VectorStore.")

    embeddings = generate_embeddings(docs)

    vector_store = DocArrayInMemorySearch(embeddings, docs)
    save_vector_store(vector_store, docs, embeddings)

    logger.info("VectorStore créé avec succès.")
    return vector_store

def create_qa_chain(vector_store):
    """Crée une chaîne QA à partir du LLM et du magasin de vecteurs."""
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa_chain

def main():
    """Point d'entrée principal du programme."""
    logger.info("Chargement des données des livres...")
    df = load_books_data()
    
    if df is None:
        logger.error("Impossible de charger les données des livres, sortie du programme.")
        return

    vector_store = create_vector_store(df)
    
    qa_chain = create_qa_chain(vector_store)

    while True:
        user_input = input("Votre question (ou tapez 'exit' pour quitter) : ")
        if user_input.lower() == 'exit':
            break

        try:
            response = qa_chain.invoke({"query": user_input})
            print("Réponse :", response['output'] if 'output' in response else "Aucune réponse générée.")

        except Exception as e:
            logger.error(f"Erreur lors du traitement de la question : {e}")
            print("Désolé, une erreur est survenue lors du traitement de votre question.")

if __name__ == "__main__":
    main()