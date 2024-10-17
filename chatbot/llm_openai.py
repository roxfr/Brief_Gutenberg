from langchain_openai import AzureChatOpenAI
from config import load_config

config = load_config()
AZURE_OPENAI_API_ENDPOINT = config["AZURE_OPENAI_API_ENDPOINT"]
AZURE_DEPLOYMENT_NAME = config["AZURE_DEPLOYMENT_NAME"]
AZURE_OPENAI_API_KEY = config["AZURE_OPENAI_API_KEY"]
AZURE_API_VERSION = config["AZURE_API_VERSION"]

class LanguageModel:
    """Classe singleton représentant un modèle de langage basé sur Azure OpenAI"""    
    _instance = None
    def __new__(cls):
        """Crée une nouvelle instance si elle n'existe pas encore"""
        if cls._instance is None:
            cls._instance = super(LanguageModel, cls).__new__(cls)
            cls._instance.llm = AzureChatOpenAI(
                azure_endpoint=AZURE_OPENAI_API_ENDPOINT,
                azure_deployment=AZURE_DEPLOYMENT_NAME,
                openai_api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_API_VERSION,
                temperature=0.0
            )
        return cls._instance
    def get_language_model(self):
        """Retourne le modèle de langage Azure OpenAI"""
        return self.llm