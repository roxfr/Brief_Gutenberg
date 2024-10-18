import os
import requests
from bs4 import BeautifulSoup
import csv

# Chemin du fichier
from ..utils.config import load_config
config = load_config()
CSV_INPUT_PATH = config["CSV_INPUT_PATH"]

# Vérifier si le fichier existe
if os.path.isfile(CSV_INPUT_PATH):
    print("Le fichier 'gutenberg.csv' existe.")
else:
    print("Le fichier 'gutenberg.csv' n'existe pas.")

# Nombre total de pages à scraper
total = 5000

# Définir les en-têtes souhaitées
headers = [
    "Author",
    "Title",
    "Note",
    "Summary",
    "Language",
    "LoC Class",
    "LoC Class",
    "Subject",
    "Category",
    "EBook-No.",
    "Release Date",
    "Most Recently Updated",
    "Copyright Status",
    "Downloads"
]

# Liste pour stocker les données
data_list = []

for ebook_id in range(total):
    url = f'https://www.gutenberg.org/ebooks/{ebook_id}'
    response = requests.get(url)

    # Vérifier si la page existe (code 200)
    if response.status_code != 200:
        print(f"URL non trouvée pour l'ID: {ebook_id}. Code de statut: {response.status_code}")
        continue

    soup = BeautifulSoup(response.content, 'html.parser')

    # Trouver le tableau "about"
    about_table = soup.find('table', class_='bibrec')

    # Initialiser un dictionnaire avec des valeurs vides
    data = {header: "" for header in headers}

    if about_table is None:
        print(f"Le tableau 'about' n'a pas été trouvé pour l'URL: {url}.")
    else:
        for row in about_table.find_all('tr'):
            th = row.find('th')
            td = row.find('td')
            if th and td:
                field = th.text.strip()
                value = td.text.strip()
                # Mettre à jour le dictionnaire si le champ est dans les en-têtes
                if field in data:
                    data[field] = value
        
        # Ajouter les données avec l'ebook_id à la liste  
        data_list.append((ebook_id, [data[header] for header in headers]))
        print(f"Données extraites pour l'ID {ebook_id}.")

# Trier les données par ebook_id
data_list.sort(key=lambda x: x[0])

# Ouvrir un fichier TSV pour écrire (tab-separated values)
with open(CSV_INPUT_PATH, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile, delimiter=',')
    
    # Écrire les en-têtes  
    csv_writer.writerow(headers)

    # Écrire les valeurs triées pour chaque page
    for ebook_id, values in data_list:
        csv_writer.writerow(values)

print(f"Extraction terminée. {len(data_list)} pages ont été traitées.")