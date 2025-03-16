"""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#%%

# Set your OpenAI API key
openai.api_key = OPENAI_API_KEY

"""
import openai
import json

from src.fake import simulated_openai_create


def extract_graph_from_tercet(sprompt, simulate=False, model="gpt-3.5-turbo"):
    prompt = f"""
Sei un esperto di analisi narrativa e di parsing semantico. Il tuo compito è convertire le terzine tratte dalla Divina Commedia di Dante in un grafo strutturato. Per ciascuna terzina, estrai le seguenti informazioni:

1. **Nodi:** Qualsiasi Persona, Luogo, Animale, Peccato o Verso esplicitamente o implicitamente menzionato.
2. **Relazioni:** Interazioni significative tra questi nodi, utilizzando i tipi di relazione ammessi.

## Tipi di nodo ammessi (e le loro proprietà):
- Verse (Verso): cantica, canto, numero_verso, testo, autore, tema
- Person (Persona): id, ruolo, identità_storica, versi, citazione, tema
- Location (Luogo): id, descrizione, versi, tema
- Animal (Animale): id, simbolismo, versi, descrizione, tema
- Sin (Peccato): id, cerchio, descrizione, punizione, versi, tema

## Tipi di relazione ammessi (e le loro proprietà):
- MENTIONS: source (Verso), target (qualsiasi nodo), numero_verso, canto
- APPEARS_IN: source (qualsiasi nodo), target (Verso), numero_verso, canto
- GUIDES: source (Persona), target (Persona), versi
- ENCOUNTERS: source (Persona), target (qualsiasi nodo), versi
- BLOCKS_PATH: source (Animale), target (Persona), versi
- DESCRIBES: source (Verso), target (qualsiasi nodo), versi
- SUFFERS: source (Persona), target (Peccato), versi
- PUNISHES: source (Peccato), target (Persona), versi
- LOCATED_IN: source (Luogo), target (Luogo)
- REPRESENTS: source (qualsiasi nodo), target (concetto astratto), descrizione
- SEEK: source (Persona), target (Persona), versi
- JUDGES: source (Persona), target (Persona), versi
- NEXT_VERSE: source (Verso), target (Verso)
- ASSOCIATED_WITH: source (Peccato), target (Luogo)
- CONTAINS: source (Canto), target (Verso)

---

## Dati in ingresso

Una stringa di dati organizzati in blocchi, dove ciascun blocco descrive una terzina. Ciascun blocco è separato da una newline (a capo).
Ogni blocco contiene:

    - cantica: Il nome della cantica ("Inferno").
    - canto: Il numero del canto ("1").
    - start_verse, end_verse: I numeri di inizio e fine verso che identificano la posizione della terzina all’interno del canto.
    - tercet: Il testo vero e proprio della terzina.
    - comment_text (opzionale): Commento critico
    - start_verse_comment, end_verse_comment (opzionali): I numeri di inizio e fine verso che identificano la posizione dei versi a cui comment_text si riferisce.

---

## Formato di output
{{
    "tercet": "{{<tercet>}}",
    "nodes": [{{<ogni nodo estratto>}}],
    "relationships": [{{<ogni relazione estratta>}}]
}}

Esempio di nodo:
{{
    "id": "Virgilio",
    "type": "Person",
    "properties": {{
        "role": "Guida",
        "historical_identity": "Poeta romano",
        "verses": ["Inferno 1:61-90"],
        "theme": "Guida spirituale"
    }}
}}

Esempio di relazione:
{{
    "source": "Virgilio",
    "target": "Narratore",
    "type": "GUIDES",
    "properties": {{
        "verses": ["Inferno 1:61-90"]
    }}
}}

---

Ora elabora le seguenti terzine:
{sprompt}

Restituisci esclusivamente l'oggetto JSON, senza spiegazioni o commenti.
"""

    request_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You must return output strictly as valid JSON. Do not include any text before or after the JSON object."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    if simulate:
        response = simulated_openai_create(**request_payload)
    else:
        response = openai.chat.completions.create(**request_payload)

    extracted_graph = json.loads(response.choices[0].message.content)
    return extracted_graph


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    from src.db_helper import retrieve_text

    load_dotenv()

    simulate = False

    if not simulate:
        openai.api_key = os.getenv("OPENAI_API_KEY")


    params = {'cantica': 1, 'canto': 1, 'start_verse': 1, 'end_verse': 200}
    _, d_list = retrieve_text(**{**params, **{"author_names": ["dante"], "type_name": "TEXT"}})

    sprompt = "\n".join([f"cantica: {d['cantica_name']}, canto: {d['canto']}, start_verse: {d['d_start_verse']}, end_verse: {d['d_end_verse']}, tercet: {d['text']}" for d in d_list])
    graph = extract_graph_from_tercet(sprompt, simulate=simulate)
    print(graph)
    # # for d in d_list:
    # for d in d_list[:3]:
    #     _params = {"cantica": d["cantica_name"], "canto": d["canto"], "start_verse": d["d_start_verse"], "end_verse": d["d_end_verse"], "tercet": d["text"]}
    #
    #     graph = extract_graph_from_tercet(**_params, simulate=simulate)
    #     print(graph)
    #
