import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import spacy
from utils.config import load_config


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

nlp = spacy.load("en_core_web_sm")

config = load_config()
CSV_INPUT_PATH = config["CSV_INPUT_PATH"]
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
N_ROWS = 5000
MAX_LENGHT = 20

text_columns = ['Author', 'Title', 'Summary', 'Subject']

df = pd.read_csv(CSV_INPUT_PATH, encoding="utf-8-sig", header=0, nrows=N_ROWS, usecols=text_columns + ['EBook-No.', 'Release Date'])

df.fillna({
    'Author': 'undefined author',
    'Title': 'undefined title',
    'Summary': 'undefined summary',
    'Subject': 'undefined subject',
    'EBook-No.': 'undefined ebook number',
    'Release Date': 'undefined release date'
}, inplace=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Nettoie le texte en supprimant les caractères indésirables et en le mettant en minuscules"""
    text = re.sub(r'[`“”()“”:;,\n\r]+', ' ', text)
    text = re.sub(r'\'\'|``+', ' ', text)
    text = ' '.join(text.split()).strip()
    return text.lower()

def remove_dates(author_name):
    """Supprime les dates et les caractères indésirables du nom de l'auteur"""
    author_name = author_name.replace('?', '').replace('-', ' ').replace('–', ' ')
    cleaned_name = ''.join(char for char in author_name if not char.isdigit()).strip()
    return ' '.join(cleaned_name.split())

def summarize_text(text, max_length=MAX_LENGHT):
    """Résumer le texte en extrayant les mots-clés"""
    doc = nlp(text)
    keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    summary = " ".join(keywords[:max_length])
    return summary

def lemmatize_text(text):
    """Lemmatiser le texte en enlevant les stop words"""
    words = text.split()
    lemmatized = [
        lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words
    ]
    return ' '.join(lemmatized)

cleaned_dfs = {}

df['Author'] = df['Author'].astype(str).apply(remove_dates)
df['Summary'] = df['Summary'].astype(str).apply(summarize_text)

for col in text_columns:
    cleaned_column = df[col].astype(str).apply(clean_text)
    lemmatized_column = cleaned_column.apply(lemmatize_text)
    cleaned_dfs[col] = pd.DataFrame({col: lemmatized_column})

cleaned_dfs['EBook-No.'] = df[['EBook-No.']]
cleaned_dfs['Release Date'] = df[['Release Date']]

final_df = pd.concat(cleaned_dfs.values(), axis=1)

final_df.to_csv(CSV_CLEANED_PATH, index=False, header=True, encoding='utf-8-sig', sep=';')