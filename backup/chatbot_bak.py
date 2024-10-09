from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding_model import EmbeddingModel
from language_model import LanguageModel
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

loader = CSVLoader("../data/gutenberg2.csv")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_documents(docs)
embedding = EmbeddingModel().get_embedding_model()

persist_directory = '../model/chroma/'
vectordb = Chroma.from_documents(documents=splits, embedding=embedding, persist_directory=persist_directory)

llm = LanguageModel().get_language_model()
qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectordb.as_retriever(), return_source_documents=True)

question = "Quel est le nom de l'auteur du livre L'Assommoir?"
result = qa_chain.invoke({"query": question})
print(result["result"])