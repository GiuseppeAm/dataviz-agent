# Architecture — DataViz Agent

## Overview

A web application that lets you upload any CSV file and interrogate it in natural language. The user types a question in a chat interface; a Claude-powered agent executes Python code against the dataset and returns a text answer plus an interactive chart.

```
Browser ──── HTTP ────► Flask (app.py)
                              │
                    ┌─────────┴──────────┐
                    │                    │
               agent_core.py      templates/index.html
               (LangChain +        (served once on GET /)
                Claude API)
```

---

## Files

### `app.py`
Flask web server. Exposes three routes:

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Serves the frontend HTML |
| `/upload` | POST | Accepts a CSV file, builds the agent session, returns dataset metadata |
| `/query` | POST | Accepts `{session_id, question}`, runs the agent, returns answer + chart JSON |

Sessions are stored in a module-level dict (`_sessions`) keyed by UUID. Each session holds one agent instance bound to its DataFrame. Sessions are in-memory only — they are lost on server restart.

### `agent_core.py`
All AI and data logic. Three responsibilities:

**1. Chart capture**
Monkey-patches `plotly.io.show` and `BaseFigure.show` at import time so any `fig.show()` call inside the agent's code execution is intercepted and the figure stored in `_captured_plotly`. This means the agent just calls `fig.show()` normally and the system captures it automatically.

**2. `build_agent(df)`**
Creates a `create_pandas_dataframe_agent` (LangChain) backed by `claude-sonnet-4-6`. The agent uses a ReAct loop: it writes Python code, executes it against the DataFrame via a sandboxed REPL tool, reads the result, and iterates until it can give a final answer.

The `ANALYST_PREFIX` injects an expert persona and enforces rules: lead with insight, cite numbers, use Plotly (not matplotlib), always call `fig.show()`, label every axis, end with `Final Answer:`.

A custom `handle_parsing_errors` callable is set so that when Claude produces output that doesn't match the ReAct format, the agent receives a correction message and retries.

**3. `run_query(agent, question)`**
Invokes the agent, then collects any captured Plotly figure (serialised to JSON via `fig.to_json()`) or matplotlib fallback (serialised to base64 PNG). If a parsing exception is thrown mid-execution, `_collect_charts()` still recovers whatever was generated before the error and `_extract_answer()` pulls a readable message from the raw LangChain exception.

**4. `auto_analyze(df)`**
Fast structural analysis run at upload time (no LLM call): shape, numeric/categorical split, null counts, strongest pairwise correlation, first numeric column stats. Used to populate the sidebar and generate suggested questions.

### `templates/index.html`
Single-file frontend — HTML, CSS, and vanilla JS with no build step.

**Layout:** 3-column CSS Grid
- **Sidebar (256px):** Upload zone → transforms to dataset metadata panel after upload (filename, rows/cols, quick insights, column list with type badges)
- **Chat column (1fr):** Message thread (user right, agent left), suggested question chips, auto-resizing textarea input
- **Artifact panel (2fr):** Dedicated area for the interactive Plotly chart. Shows a shimmer skeleton while the agent is thinking. Has Download PNG and Fullscreen buttons. Collapsible via toggle button in the chat header.

**Chart rendering:** The backend sends `chart_json` (Plotly figure serialised as JSON). The frontend calls `Plotly.newPlot()` with `responsive: true` — zoom, pan, hover tooltips and legend toggle all work natively. Plotly.js is loaded from CDN.

**Key JS flows:**
- `handleFile()` → POST `/upload` → `onDatasetLoaded()` populates sidebar + suggestions
- `sendMessage()` → POST `/query` → `appendAssistantMsg()` + `renderChart()`
- Toggle button: adds/removes `.chart-hidden` class on `.app`, which switches `grid-template-columns`

### `requirements.txt`
Python dependencies: `anthropic`, `langchain`, `langchain-anthropic`, `langchain-experimental`, `flask`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`, `python-dotenv`, `gunicorn`.

### `Procfile`
`web: gunicorn app:app` — used by Railway and Render for production deployment.

### `start.bat`
Windows launcher: starts Flask in a new console window, waits 2 seconds, opens `http://localhost:5000` in the browser.

### `.env`
Contains `ANTHROPIC_API_KEY`. Loaded by `python-dotenv` at startup. Never committed (in `.gitignore`).

### `old/`
Original course files (IBM Watsonx notebook + standalone script). Ignored by git.

---

## External Services

| Service | Role |
|---------|------|
| **Anthropic API** | Powers the agent via `claude-sonnet-4-6`. Called on every `/query` request. Requires `ANTHROPIC_API_KEY` in `.env` |
| **Plotly CDN** (`cdn.plot.ly`) | Loads `plotly-2.35.2.min.js` in the browser for interactive chart rendering |

---

## Data flow — one query

```
User types question
       │
       ▼
POST /query  {session_id, question}
       │
       ▼
agent_core.run_query()
  ├── clears _captured_plotly
  ├── agent.invoke(question)
  │     ├── Claude generates Python code   ◄── Anthropic API
  │     ├── LangChain executes code in REPL
  │     │     └── fig.show() → captured by monkey-patch
  │     └── Claude writes Final Answer
  ├── _collect_charts() → fig.to_json()
  └── returns {answer, chart_json, code}
       │
       ▼
Flask returns JSON
       │
       ▼
Browser: appendAssistantMsg() + Plotly.newPlot()
```
