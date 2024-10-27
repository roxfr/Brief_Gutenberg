import os
import logging
from langchain_chroma import Chroma
from chromadb import PersistentClient, Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PERSIST_DIRECTORY = os.path.join("data", "chroma_langchain_db")
COLLECTION_NAME = "project_gutenberg"

def display_vector_store(persist_directory: str, collection_name: str):
    """Affiche le contenu du vector store"""
    try:
        client = PersistentClient(path=persist_directory, settings=Settings(anonymized_telemetry=False))
        collection = client.get_collection(collection_name)        
        total_documents = collection.count()

        if total_documents > 0:
            results = collection.query(query_texts=[""], n_results=total_documents)
            if results and 'documents' in results:
                for doc, metadata in zip(results['documents'], results['metadatas']):
                    print(f"Document : {doc}")
                    print(f"Métadonnées : {metadata}")
                    print("-" * 40)
                print(f"Total documents : {total_documents}")
        else:
            print("Aucun document trouvé dans le vector store.")
    except Exception as e:
        logging.error(f"Erreur lors de l'affichage du vector store : {e}")
if __name__ == "__main__":
    display_vector_store(PERSIST_DIRECTORY, COLLECTION_NAME)