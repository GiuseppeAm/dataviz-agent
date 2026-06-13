import io
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


def _read_csv(file_storage) -> pd.DataFrame:
    """Parse an uploaded CSV robustly: handles BOM, latin-1 encoding, and
    semicolon/tab delimiters common in non-US exports. Raises ValueError with
    a user-facing message on failure."""
    raw = file_storage.read()
    if not raw:
        raise ValueError("The file is empty.")

    text = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError("Could not read the file's text. Please save it as UTF-8 CSV.")

    sample = text[:8192]
    delimiter = ","
    if sample.count(";") > sample.count(","):
        delimiter = ";"
    elif sample.count("\t") > sample.count(","):
        delimiter = "\t"

    try:
        df = pd.read_csv(io.StringIO(text), sep=delimiter)
    except pd.errors.EmptyDataError:
        raise ValueError("The file has no columns to parse.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"The CSV is malformed and could not be parsed: {exc}")

    if df.shape[1] == 0:
        raise ValueError("No columns were detected. Check the file's delimiter.")
    if df.empty:
        raise ValueError("The file has headers but no data rows.")
    return df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file was attached to the request."}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "No file was selected."}), 400
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only .csv files are supported."}), 400

    try:
        df = _read_csv(f)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Could not read the CSV: {exc}"}), 400

    try:
        agent = build_agent(df)
    except Exception as exc:
        return jsonify({"error": f"Could not initialize the analysis agent: {exc}"}), 500

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"agent": agent}

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
