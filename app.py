import urllib

from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from qdrant_client import QdrantClient

from src.utils import load_model


app = Flask(__name__)

# Load CSV and filter rows where author is "dante"
commedia = pd.read_csv(
    r"/home/rfflpllcn/IdeaProjects/divine_semantics_db/divine_semantics_db/data/commedia.csv",
    sep=";",
    quotechar='"'
)
commedia = commedia[commedia["author"] == "dante"]

# Initialize Qdrant client using environment variables
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")
qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

collection_name = "dante_parafrasi_multilingual_e5"

# Load your model (adjust this if you use a different loader function)
model = load_model("multilingual_e5")


# Home route serving the HTML form
@app.route('/')
def index():
    return render_template("index.html")


@app.route("/query", methods=["GET"])
def query_verse():
    # Retrieve the query text from the request parameters; default if not provided
    query_text = request.args.get("text", "che lasciai la strada")

    # Encode the query and perform the Qdrant search
    query_embedding = model.encode(query_text)
    hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=1
    )

    # Check if any hits were returned
    if not hits.points:
        return jsonify({"error": "No matching verse found"}), 404

    cum_verse_number = hits.points[0].payload['cum_verse_number']

    # Filter the CSV data for the matching verse
    filtered = commedia[commedia["cum_verse_number"] == cum_verse_number].to_dict(orient="records")
    if not filtered:
        return jsonify({"error": "No matching record in CSV found"}), 404

    result = filtered[0]

    # Build the link dynamically using cantica and canto from the result
    params = {
        "reader[cantica]": result['cantica'],
        "reader[canto]": result['canto']
    }
    base_url = "http://dantelab.dartmouth.edu/reader"
    reader_link = base_url + "?" + urllib.parse.urlencode(params)

    # Prepare and return the JSON response
    out = {
        "cantica": result['cantica'],
        "canto": result['canto'],
        "verse_number": result['verse_number'],
        "verse": result['text'],
        "user_query": query_text,
        "reader_link": reader_link
    }
    return render_template("result.html", result=out)

if __name__ == "__main__":
    app.run(debug=True)
