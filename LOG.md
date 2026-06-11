# Session Log

## Session 1 — 2026-06-11

### Starting point
Course notebook `chat-with-your-dataframe-using-langchain-v1.ipynb` (IBM Watsonx + LangChain) and a standalone Python script `data-visualization-agent.py` adapted to use Claude instead of Watsonx. Both worked as REPL-based scripts with `plt.show()` at the end of a loop.

### What we did

**1. Explored and tested the original script**
- Reviewed the notebook exercises (scatter plots, bar charts, box plots, scatter of absences vs G3)
- Confirmed the standalone script was functionally equivalent to the notebook, minus the exercises
- Listed suggested test questions grouped by type (simple analysis, histograms, aggregations, relational, advanced)

**2. Moved originals to `old/` and updated `.gitignore`**
- `data-visualization-agent.py` → `old/`
- `chat-with-your-dataframe-using-langchain-v1.ipynb` → `old/`
- Added `old/` to `.gitignore`

**3. Built the full web application**

Created from scratch:

- `agent_core.py` — LangChain agent with expert analyst persona (`ANALYST_PREFIX`), dark Plotly theme, chart capture via monkey-patching, `auto_analyze()` for fast structural summary
- `app.py` — Flask server with `/upload` and `/query` endpoints, UUID-based in-memory sessions
- `templates/index.html` — full frontend: 3-column layout (sidebar / chat / artifact panel), drag-and-drop CSV upload, suggested questions, inline Plotly charts
- `requirements.txt` — added `flask`, `numpy`, `plotly`, `gunicorn`
- `Procfile` — `web: gunicorn app:app` for cloud deployment
- `start.bat` — Windows double-click launcher
- `ARCHITECTURE.md` — this architecture document

**4. Upgraded to interactive Plotly charts**
- Switched from matplotlib static images to Plotly interactive charts
- Monkey-patched `plotly.io.show` and `BaseFigure.show` to capture figures without file I/O
- Set a custom dark Plotly template (`dataviz`) matching the UI palette
- Chart sent as JSON to the frontend and rendered via `Plotly.newPlot()`
- Artifact panel: Download PNG button (`Plotly.downloadImage`), Fullscreen toggle (Esc to close)

**5. Fixed parsing errors and chart recovery**
- Root cause: Claude's rich markdown responses (tables, headers) broke LangChain's ReAct output parser
- Fix 1: added `Final Answer:` format instruction to `ANALYST_PREFIX`
- Fix 2: changed `handle_parsing_errors` from `True` to a callable that sends Claude a correction message
- Fix 3: `run_query` now calls `_collect_charts()` in the `except` block — chart is recovered even when a parsing exception is thrown; `_extract_answer()` extracts readable text from the raw error

**6. Frontend improvements**
- Increased text contrast: `--t2` 66% → 78%, `--t3` 44% → 58%, borders more visible
- Layout columns changed from `256px 380px 1fr` to `256px 1fr 2fr` (better proportions)
- Added chart panel toggle button (top-right of chat header) with `.chart-hidden` CSS class toggle

**7. Added `load_dotenv()` to `app.py`**
- Original script loaded `.env` but `app.py` didn't — API key was missing on server start

---

### Current state
The app runs locally. Start with `start.bat` or `python app.py`, open `http://localhost:5000`.

---

## Next session — TODO

### Hosting on Railway
1. Create a GitHub repository and push the project
2. Connect repo to Railway (railway.app → New Project → Deploy from GitHub)
3. Set environment variable: `ANTHROPIC_API_KEY = sk-ant-...`
4. Railway auto-detects `Procfile` and runs `gunicorn app:app`
5. Verify the public URL works end-to-end

### Known limitations to address (optional)
- **Sessions are in-memory**: if the server restarts or scales to multiple workers, sessions are lost and the user must re-upload the CSV. For production, sessions should be persisted (Redis, or re-upload on reconnect)
- **No conversation memory**: each query is independent; the agent does not remember previous questions in the same session. Could add `ConversationBufferMemory` to the LangChain agent
- **Single worker**: `gunicorn app:app` with default settings runs one worker — fine for demo, not for concurrent users
- **Large files**: 50 MB limit is set but no streaming; very wide DataFrames may hit LangChain context limits
