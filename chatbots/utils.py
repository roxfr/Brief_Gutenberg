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