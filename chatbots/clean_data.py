import pandas as pd


from config import load_config
config = load_config()
CSV_INPUT_PATH = config["CSV_INPUT_PATH"]
CSV_CLEANED_PATH = config["CSV_CLEANED_PATH"]

df = pd.read_csv(CSV_INPUT_PATH, encoding="utf-8", header=0)
df.fillna({
    'Author': 'Auteur non défini',
    'Title': 'Titre non défini',
    'Credits': 'Crédits non définis',
    'Summary': 'Résumé non défini',
    'Subject': 'Sujet non défini',
    'EBook-No.': 'Numéro d\'ebook non défini',
    'Release Date': 'Date de publication non définie'
}, inplace=True)
text_columns = ['Author', 'Title', 'Summary', 'Subject', 'EBook-No.', 'Release Date']
for col in text_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.replace(r'[^\x00-\x7F]+', '', regex=True)
        .str.replace(r'[\n\r]', ' ', regex=True)
        .str.strip()
    )
df.drop(columns=['Ebook ID', 'Credits', 'Language', 'LoC Class', 
                 'Subject_2', 'Subject_3', 'Subject_4', 'Category', 
                 'Most Recently Updated', 'Copyright Status', 'Downloads'], 
        inplace=True, errors='ignore')
df.dropna(how='all', inplace=True)
df = df.head(5000)
df.to_csv(CSV_CLEANED_PATH, index=False, header=True, encoding='utf-8')