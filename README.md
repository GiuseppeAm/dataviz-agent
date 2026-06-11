# DataViz Agent

Upload any CSV file and interrogate it in natural language. A Claude-powered agent writes and executes Python code against your dataset, returning a text analysis and an interactive Plotly chart.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey)
![Claude](https://img.shields.io/badge/Claude-Sonnet_4.6-orange)

---

## Features

- Drag-and-drop CSV upload
- Natural language queries in a chat interface
- Interactive charts (zoom, pan, hover) via Plotly — rendered as artifacts in a dedicated panel
- Expert analyst persona: leads with insights, cites exact numbers, flags anomalies
- Auto-analysis on upload: shape, null counts, top correlation, suggested questions
- Collapsible chart panel, fullscreen mode, PNG download
- Generated Python code visible in collapsible blocks

## Stack

| Layer | Technology |
|-------|-----------|
| LLM | Claude Sonnet 4.6 via Anthropic API |
| Agent | LangChain `create_pandas_dataframe_agent` |
| Charts | Plotly (interactive) with matplotlib fallback |
| Backend | Flask |
| Frontend | Vanilla HTML/CSS/JS — no build step |

---

## Setup

**1. Clone and install dependencies**

```bash
git clone https://github.com/<your-username>/dataviz-agent.git
cd dataviz-agent
pip install -r requirements.txt
```

**2. Set your API key**

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

**3. Run**

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000).

On Windows you can also double-click `start.bat`.

---

## Project structure

```
├── app.py               # Flask server — /upload and /query endpoints
├── agent_core.py        # LangChain agent, Plotly capture, auto-analysis
├── templates/
│   └── index.html       # Full frontend (HTML + CSS + JS)
├── requirements.txt
├── Procfile             # For Railway / Render deployment
├── start.bat            # Windows launcher
├── student-mat.csv      # Sample dataset (UCI Student Alcohol Consumption)
├── ARCHITECTURE.md      # Detailed technical documentation
└── LOG.md               # Session log and roadmap
```

---

## Deploy to Railway

1. Push to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
3. Add environment variable: `ANTHROPIC_API_KEY`
4. Railway detects `Procfile` automatically — deploy is live in ~2 minutes

---

## Sample dataset

`student-mat.csv` is the [UCI Student Alcohol Consumption](https://www.kaggle.com/datasets/uciml/student-alcohol-consumption) dataset (395 students, 33 columns). Try questions like:

- *"Plot the distribution of final grades (G3)"*
- *"Show average G3 grouped by mother's education level"*
- *"What is the correlation between study time and final grade?"*
- *"Generate scatter plots of daily and weekend alcohol consumption vs G3"*
