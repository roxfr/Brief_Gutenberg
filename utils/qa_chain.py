import logging
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
# from utils.tools import (
#     get_author_by_title,
#     get_books_by_author,
#     get_subject_by_title,
#     get_characters_by_title,
#     get_all_text_book_by_title_from_url
# )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# tools = {
#     "get_author_by_title": {"description": "Finds the author of a book by its title.", "function": get_author_by_title},
#     "get_books_by_author": {"description": "Retrieves books written by a specific author.", "function": get_books_by_author},
#     "get_subject_by_title": {"description": "Returns the subject of a book by its title.", "function": get_subject_by_title},
#     "get_characters_by_title": {"description": "Extracts the characters of a book by its title.", "function": get_characters_by_title},
#     "get_all_text_book_by_title_from_url": {"description": "Retrieves the complete text of a book online by its title.", "function": get_all_text_book_by_title_from_url},
# }

def setup_qa_chain(llm, vector_store):
    template = """
    You are an intelligent assistant specialized in literature, particularly books from Project Gutenberg.

    You have the following tools available to answer questions:    
    - **get_books_by_author**: Retrieves books written by a specific author.
    - **get_author_by_title**: Finds the author of a book by its title.
    - **get_subject_by_title**: Returns the subject of a book by its title.
    - **get_characters_by_title**: Extracts the characters of a book by its title.
    - **get_all_text_book_by_title_from_url**: Retrieves the complete text of a book online by its title.

    Please respond to the following question using the appropriate tool.

    **Context**: {context}
    **Question**: {question}
    **Answer**: Only provide the final answer, no additional steps or code.
    """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    retriever = vector_store.as_retriever(k=1)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    context_runnable = retriever | RunnableLambda(format_docs)
    question_runnable = RunnablePassthrough()

    rag_chain = (
        {"context": context_runnable, "question": question_runnable}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
