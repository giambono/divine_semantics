from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import os
import pandas as pd
import urllib.parse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.utils import load_model

app = Flask(__name__)
app.secret_key = "521978"  # Change this to a secure, random key

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
model = load_model("multilingual_e5")

@app.route("/")
def index():
    return render_template("index.html")

# Settings route to update the limit value
@app.route("/settings", methods=["GET", "POST"])
def settings():
    if request.method == "POST":
        new_limit = request.form.get("limit", "1")
        try:
            limit_value = int(new_limit)
        except ValueError:
            limit_value = 1
        session["limit"] = limit_value
        return redirect(url_for("settings"))
    # On GET, show the current setting (default to 1)
    current_limit = session.get("limit", 1)
    return render_template("settings.html", limit=current_limit)

@app.route("/query", methods=["GET"])
def query_verse():
    # Retrieve the query text; default if not provided
    query_text = request.args.get("text", "che lasciai la strada")

    # Use the saved limit from settings (session) or default to 1
    limit_value = session.get("limit", 1)

    # Encode the query and perform the Qdrant search using the provided limit
    query_embedding = model.encode(query_text)
    hits = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=limit_value
    )

    # Check if any hits were returned
    if not hits.points:
        return jsonify({"error": "No matching verse found"}), 404

    out_collect = []
    for point in hits.points:
        print("score", point.score)
        cum_verse_number = point.payload['cum_verse_number']

        # Filter the CSV data for the matching verse
        filtered = commedia[commedia["cum_verse_number"] == cum_verse_number].to_dict(orient="records")
        if not filtered:
            return jsonify({"error": "No matching record in CSV found"}), 404

        result = filtered[0]
        print("result", result)
        # Build the link dynamically using cantica and canto from the result
        params = {
            "reader[cantica]": result['cantica'],
            "reader[canto]": result['canto']
        }
        base_url = "http://dantelab.dartmouth.edu/reader"
        reader_link = base_url + "?" + urllib.parse.urlencode(params)

        # Prepare and return the result rendered via the template
        out = {
            "score": point.score,
            "cantica": result['cantica'],
            "canto": result['canto'],
            "verse_number": result['verse_number'],
            "verse": result['text'],
            "user_query": query_text,
            "reader_link": reader_link
        }
        out_collect.append(out)
    return render_template("result.html", result=out_collect)

if __name__ == "__main__":
    app.run(debug=True)
