from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.schema.runnable import RunnablePassthrough


# def setup_qa_chain(llm, vector_store) -> RunnableSequence:
#     """Mise en place de la chaîne QA."""
#     template = """
#         **Contexte** : Vous êtes un expert en littérature, spécialisé dans les œuvres du Projet Gutenberg.
#         **Instructions** :
#         - Répondez uniquement avec des informations vérifiées provenant de ces textes.
#         - Si vous ne trouvez pas l'information, répondez uniquement par "Je ne sais pas."
#         - Utilisez le magasin de vecteurs (chromadb) pour vous aider dans vos réponses.
#         - Évitez de faire des suppositions ou des interprétations personnelles.
#         **Format** : Fournissez des réponses concises et directes.
#         **Outils disponibles** : {tools}
#         **Question** : '{question}'.
#     """

def setup_qa_chain(llm, vector_store) -> RunnableSequence:
    """Set up the QA chain."""
    template = """
        **Context**: You are a literature expert, specializing in the works of the Gutenberg Project.
        **Instructions**:
        - Respond only with verified information from these texts.
        - If you cannot find the information, respond only with "I don't know."
        - Use the vector store (chromadb) to assist in your answers.
        - Avoid making assumptions or personal interpretations.
        **Format**: Provide concise and direct answers.
        **Available Tools**: {tools}
        **Question**: '{question}'.
    """

    prompt = PromptTemplate(template=template, input_variables=["question", "tools"])
    retriever = vector_store.as_retriever(k=5)
    qa_chain = RunnableSequence(
        {
            "context": retriever, 
            "question": RunnablePassthrough(),
            "tools": RunnablePassthrough()
        }
        | prompt
        | llm
        | RunnablePassthrough()
    )
    return qa_chain