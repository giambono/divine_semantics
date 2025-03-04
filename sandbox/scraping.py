import os
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd

entities = pd.read_csv('entities.csv')

# Cartella di output
output_dir = "enciclopedia_dantesca"
os.makedirs(output_dir, exist_ok=True)

base_url = "https://www.treccani.it/enciclopedia/elenco-opere/Enciclopedia_Dantesca/{}/"

# Range di pagine
for page_num in range(1, 198):  # da 1 a 197
    page_url = base_url.format(page_num)
    response = requests.get(page_url)

    if response.status_code != 200:
        print(f"Errore nella pagina {page_num}: {response.status_code}")
        continue

    soup = BeautifulSoup(response.content, 'lxml')

    # Trova tutti i link alle singole voci
    links = []
    for link in soup.select('a[href*="_(Enciclopedia-Dantesca)"]'):
        href = link['href']
        full_url = requests.compat.urljoin(page_url, href)
        links.append(full_url)

    print(f"Pagina {page_num}: trovati {len(links)} link.")

    # Scarica e salva ciascuna voce
    for voce_url in links:
        voce_response = requests.get(voce_url)

        if voce_response.status_code != 200:
            print(f"Errore su {voce_url}: {voce_response.status_code}")
            continue

        voce_soup = BeautifulSoup(voce_response.content, 'lxml')

        # Estrai il titolo della voce
        title = voce_soup.find('h1').text.strip().replace("/", "_").replace(" ", "_")

        found = any(title.lower() in item.lower() for item in entities)
        if found:
            # Salva la voce (sia come HTML sia come testo se vuoi)
            with open(os.path.join(output_dir, f"{title}.html"), "w", encoding="utf-8") as f:
                f.write(voce_response.text)

            # Estrazione e salvataggio testo pulito
            content_div = voce_soup.find('div', class_="text")
            if content_div:
                clean_text = content_div.get_text("\n", strip=True)
                with open(os.path.join(output_dir, f"{title}.txt"), "w", encoding="utf-8") as f:
                    f.write(clean_text)

            time.sleep(1)  # Per non sovraccaricare il server
