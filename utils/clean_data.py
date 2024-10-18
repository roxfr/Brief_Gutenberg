import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

from utils.config import load_config
config = load_config()
CSV_INPUT_PATH = config["CSV_INPUT_PATH"]
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]
N_ROWS = 5000

text_columns = ['Author', 'Title', 'Summary', 'Subject']

df = pd.read_csv(CSV_INPUT_PATH, encoding="utf-8-sig", header=0, nrows=N_ROWS, usecols=text_columns + ['EBook-No.', 'Release Date'])

df.fillna({
    'Author': 'Undefined Author',
    'Title': 'Undefined Title',
    'Summary': 'Undefined Summary',
    'Subject': 'Undefined Subject',
    'EBook-No.': 'Undefined eBook Number',
    'Release Date': 'Undefined Release Date'
}, inplace=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# def clean_text(text):
#     text = re.sub(r'[`“”\'‘’"()“”:;,.\n\r-]+', ' ', text)
#     text = re.sub(r'\'\'|``+', ' ', text) 
#     return ' '.join(text.split()).strip()

def clean_text(text):
    text = re.sub(r'[`“”()“”:;,\n\r]+', ' ', text)
    text = re.sub(r'\'\'|``+', ' ', text)
    return ' '.join(text.split()).strip()

def lemmatize_text(text):
    words = text.split()
    lemmatized = [
        lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words
    ]
    return ' '.join(lemmatized)

cleaned_dfs = {}

for col in text_columns:
    cleaned_column = df[col].astype(str).apply(clean_text)
    lemmatized_column = cleaned_column.apply(lemmatize_text)
    cleaned_dfs[col] = pd.DataFrame({col: lemmatized_column})

cleaned_dfs['EBook-No.'] = df[['EBook-No.']]
cleaned_dfs['Release Date'] = df[['Release Date']]

final_df = pd.concat(cleaned_dfs.values(), axis=1)

final_df.to_csv(CSV_CLEANED_PATH, index=False, header=True, encoding='utf-8-sig', sep=';')