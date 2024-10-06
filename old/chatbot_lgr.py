import os
import pandas as pd
import requests
from dotenv import load_dotenv, find_dotenv
import spacy
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from data_loader import load_books_data

def load_environment():
    """Charge les variables d'environnement pour la connexion à Azure OpenAI."""
    load_dotenv(find_dotenv())
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_API_ENDPOINT")
    azure_deployment = os.getenv("AZURE_DEPLOYMENT_NAME")

    if None in {azure_openai_api_key, azure_endpoint, azure_deployment}:
        raise ValueError("Informations Azure OpenAI non trouvées.")
    return azure_openai_api_key, azure_endpoint, azure_deployment

@tool
def get_book_info(subject: str, df: pd.DataFrame, info_type: str):
    """Récupère des informations sur un livre en fonction du type fourni."""
    if df is None or subject is None:
        return "Données manquantes."
    
    book_info = df[df['Subject'].str.contains(subject, case=False, na=False)]
    
    if not book_info.empty:
        return book_info.iloc[0].get(info_type.capitalize(), "Inconnu")
    
    return f"Désolé, je n'ai trouvé aucune {info_type} pour le sujet '{subject}'."

@tool
def extract_characters_from_summary(summary):
    """Extrait une liste de personnages mentionnés dans le résumé d'un livre."""
    if pd.isna(summary):
        return []
    nlp = spacy.load("en_core_web_sm")
    characters = {ent.text for ent in nlp(summary).ents if ent.label_ == "PERSON"}
    return list(characters)

@tool
def fetch_book_text(book_id):
    """Récupère le texte complet d'un livre depuis Gutenberg."""
    txt_url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    try:
        response = requests.get(txt_url)
        response.raise_for_status()
        return response.text
    except requests.HTTPError as e:
        return f"Erreur {e.response.status_code} : Problème avec l'ID du livre {book_id}."
    except requests.RequestException as e:
        return f"Erreur de requête pour l'ID du livre {book_id} : {e}"

@tool
def fetch_lines(book_id, num_lines):
    """Récupère les premières lignes d'un livre spécifié."""
    book_text = fetch_book_text(book_id)
    if book_text:
        return f"Premières lignes du livre :\n" + "\n".join(book_text.splitlines()[:num_lines])
    return "Une erreur s'est produite lors de la récupération du livre."

def initialize_tools():
    """Initialise les outils disponibles."""
    return [
        lambda subject, df: get_book_info(subject, df, 'title'),
        lambda subject, df: get_book_info(subject, df, 'author'),
        lambda title, df: get_book_info(title, df, 'subject'),
        extract_characters_from_summary,
        fetch_book_text,
        fetch_lines
    ]

def create_agent(azure_openai_api_key, azure_endpoint, azure_deployment, tools):
    """Crée un agent de conversation utilisant Azure OpenAI et les outils définis."""
    model = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        openai_api_key=azure_openai_api_key,
        api_version="2023-05-15"
    )

    template = '''
    Vous êtes un agent de conversation dédié à fournir des informations sur Project Gutenberg et ses œuvres littéraires. Vous avez accès aux outils suivants : {tools}

    Veuillez répondre à la question suivante en suivant ces étapes :

    1. **Lisez la question** : {input}
    2. **Analysez** : Réfléchissez à quel outil utiliser pour fournir la meilleure réponse.
    3. **Formulez votre action** :
    - **Action** : <nom de l'outil à utiliser>
    - **Action Input** : {{ "subject": "<subject>" }} (si applicable, sinon laissez ce champ vide)

    ---
    L'exécution commence ici :
    - Question : {input}
    - Pensée : {agent_scratchpad}
    '''

    prompt = PromptTemplate.from_template(template)

    try:
        agent = create_react_agent(model, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=10,
            max_execution_time=60.0,
            return_intermediate_steps=True,
            verbose=True,
            handle_parsing_errors=True
        )
    except Exception as e:
        print(f"Erreur lors de la création de l'agent : {e}")
        return None

def main():
    azure_openai_api_key, azure_endpoint, azure_deployment = load_environment()
    df = load_books_data()
    tools = initialize_tools()
    agent_executor = create_agent(azure_openai_api_key, azure_endpoint, azure_deployment, tools)

    if agent_executor is None:
        print("L'agent n'a pas pu être créé.")
        return

    while True:
        user_input = input("Posez votre question (ou tapez 'exit' pour quitter) : ")
        if user_input.lower() == 'exit':
            break

        try:
            response = agent_executor.invoke({"input": user_input, "df": df})
            
            if isinstance(response, dict) and "output" in response:
                output = response["output"]
                print("Réponse :", output)
            else:
                print("La réponse de l'agent n'a pas le format attendu.")
                
        except Exception as e:
            print(f"Une erreur s'est produite : {e}")

if __name__ == "__main__":
    main()
