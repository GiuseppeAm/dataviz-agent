import os
import uuid

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from flask import Flask, jsonify, render_template, request

from agent_core import auto_analyze, build_agent, run_query

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

# In-memory session store: session_id -> {"agent": ...}
_sessions: dict = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported"}), 400

    try:
        df = pd.read_csv(f)
    except Exception as exc:
        return jsonify({"error": f"Could not parse CSV: {exc}"}), 400

    if df.empty:
        return jsonify({"error": "The CSV file is empty"}), 400

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"agent": build_agent(df)}

    analysis = auto_analyze(df)
    return jsonify({"session_id": session_id, "filename": f.filename, **analysis})


@app.route("/query", methods=["POST"])
def query():
    data = request.get_json(force=True)
    session_id = data.get("session_id", "").strip()
    question = data.get("question", "").strip()

    if not session_id or session_id not in _sessions:
        return jsonify({"error": "Session not found. Please re-upload your file."}), 404
    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400

    agent = _sessions[session_id]["agent"]
    result = run_query(agent, question)
    return jsonify(result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, port=port)
