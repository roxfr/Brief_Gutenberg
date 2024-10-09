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
