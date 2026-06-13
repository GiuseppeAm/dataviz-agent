import io
import math
import base64
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Plotly: patch show() before anything else imports it ──────────────────────
import plotly.graph_objects as go
import plotly.io as pio
import plotly.basedatatypes as _plotly_base

from langchain_anthropic import ChatAnthropic
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

warnings.filterwarnings("ignore")

# ── Capture Plotly figures instead of displaying them ─────────────────────────
_captured_plotly: list = []

def _pio_show_patch(fig, *args, **kwargs):
    _captured_plotly.append(fig)

def _fig_show_patch(self, *args, **kwargs):
    _captured_plotly.append(self)

pio.show = _pio_show_patch
_plotly_base.BaseFigure.show = _fig_show_patch

# ── Plotly dark template matching the UI ──────────────────────────────────────
pio.templates["dataviz"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#1b1714",
        plot_bgcolor="#222018",
        font=dict(
            color="#ece7df",
            size=12,
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif",
        ),
        colorway=["#cf9240", "#6b9ec4", "#8fbd72", "#c47a6b", "#a87bc4", "#c4b76b"],
        xaxis=dict(
            gridcolor="#2e2820",
            linecolor="#3d352a",
            zerolinecolor="#3d352a",
            tickfont=dict(color="#9c9087", size=11),
            title_font=dict(color="#ece7df", size=12),
        ),
        yaxis=dict(
            gridcolor="#2e2820",
            linecolor="#3d352a",
            zerolinecolor="#3d352a",
            tickfont=dict(color="#9c9087", size=11),
            title_font=dict(color="#ece7df", size=12),
        ),
        title=dict(font=dict(size=14, color="#ece7df")),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="#3d352a",
            borderwidth=1,
            font=dict(color="#c8c2ba", size=11),
        ),
        margin=dict(t=56, r=24, b=48, l=64),
        hoverlabel=dict(
            bgcolor="#282219",
            bordercolor="#3d352a",
            font=dict(color="#ece7df", size=12),
        ),
    )
)
pio.templates.default = "dataviz"

# ── Matplotlib fallback theme ─────────────────────────────────────────────────
_BG, _SURFACE, _BORDER = "#1b1714", "#282219", "#3d352a"
_TEXT1, _TEXT2         = "#ece7df", "#9c9087"
_ACCENT, _GRID         = "#cf9240", "#2a2420"

plt.rcParams.update({
    "figure.facecolor":   _BG,     "figure.figsize":     (9, 5),
    "axes.facecolor":     _SURFACE,"axes.edgecolor":     _BORDER,
    "axes.labelcolor":    _TEXT1,  "axes.titlecolor":    _TEXT1,
    "axes.titlesize":     13,      "axes.labelsize":     11,
    "axes.titlepad":      14,      "axes.spines.top":    False,
    "axes.spines.right":  False,   "text.color":         _TEXT1,
    "xtick.color":        _TEXT2,  "ytick.color":        _TEXT2,
    "xtick.labelsize":    9,       "ytick.labelsize":    9,
    "grid.color":         _GRID,   "grid.linewidth":     0.6,
    "lines.color":        _ACCENT, "patch.edgecolor":    _BORDER,
    "savefig.facecolor":  _BG,     "savefig.bbox":       "tight",
    "savefig.dpi":        150,     "font.family":        "sans-serif",
    "font.size":          10,
})
sns.set_palette([_ACCENT, "#6b9ec4", "#8fbd72", "#c47a6b", "#a87bc4", "#c4b76b"])


# ── Expert analyst persona ────────────────────────────────────────────────────
ANALYST_PREFIX = """\
You are a senior data scientist and expert data analyst with deep expertise in \
exploratory data analysis, statistics, and data storytelling.

Behavioral rules — follow these without exception:

1. Lead with the key insight, not the method. State what the data shows \
   before how you found it.
2. Cite exact numbers. Every qualitative claim must be backed by a specific \
   value from the data.
3. Surface anomalies proactively. Mention outliers, skew, or unexpected patterns \
   even if the user didn't ask.
4. Interpret ambiguous questions in the most analytically useful way.

When creating visualizations:
5. ALWAYS use plotly.express (import as px) or plotly.graph_objects (import as go). \
   Do NOT use matplotlib or seaborn for charts.
6. ALWAYS call fig.show() at the end — the system captures the figure automatically.
7. Do NOT call fig.write_html(), fig.write_image(), or save any files.
8. The dark theme is applied automatically — do NOT set template in your code.
9. Every chart MUST have a title: fig.update_layout(title="...")
10. Always label axes: fig.update_layout(xaxis_title="...", yaxis_title="...")
11. After generating a chart, write one sentence describing the most important \
    visual pattern.

RESPONSE FORMAT: End every response with exactly:
Final Answer: <your analysis in plain prose — no markdown tables, no # headers>
"""


def build_agent(df: pd.DataFrame):
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    return create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        return_intermediate_steps=True,
        handle_parsing_errors=lambda e: (
            "Your response could not be parsed. Reply using exactly this format: "
            "'Final Answer: <your analysis in plain prose, no markdown tables>'"
        ),
        allow_dangerous_code=True,
        prefix=ANALYST_PREFIX,
    )


def _collect_charts() -> tuple:
    """Return (chart_json, chart_b64) from whatever was captured."""
    if _captured_plotly:
        try:
            return _captured_plotly[-1].to_json(), None
        except Exception:
            pass
    if plt.get_fignums():
        try:
            buf = io.BytesIO()
            plt.savefig(buf, format="png")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode()
            plt.close("all")
            return None, b64
        except Exception:
            pass
    return None, None


def _extract_answer(exc: Exception) -> str:
    """Try to pull a readable message from a LangChain parse error."""
    msg = str(exc)
    if "Could not parse LLM output" in msg:
        # The raw LLM output is wrapped in backticks inside the error message
        parts = msg.split("`")
        for candidate in parts[1::2]:          # every other segment (inside backticks)
            candidate = candidate.strip()
            if len(candidate) > 40:
                return candidate[:800]
        return "Chart generated. (Response could not be fully parsed.)"
    return f"Analysis error: {msg[:300]}"


def run_query(agent, question: str) -> dict:
    _captured_plotly.clear()
    plt.close("all")

    try:
        result = agent.invoke({"input": question})
    except Exception as exc:
        # Chart may have been generated before the parsing error — recover it.
        chart_json, chart_b64 = _collect_charts()
        answer = _extract_answer(exc)
        return {"answer": answer, "chart_json": chart_json, "chart_b64": chart_b64, "code": []}

    chart_json, chart_b64 = _collect_charts()

    code_steps = [
        step[0].tool_input
        for step in result.get("intermediate_steps", [])
        if hasattr(step[0], "tool_input")
    ]

    return {
        "answer":     result["output"],
        "chart_json": chart_json,
        "chart_b64":  chart_b64,
        "code":       code_steps,
    }


def _json_safe(obj):
    """Recursively replace NaN with None so the result serialises to valid JSON."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def auto_analyze(df: pd.DataFrame) -> dict:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    nulls = {
        col: int(n)
        for col, n in df.isnull().sum().items()
        if n > 0
    }

    top_corr = None
    if len(num_cols) >= 2:
        corr  = df[num_cols].corr().abs()
        mask  = np.triu(np.ones(corr.shape, dtype=bool), k=1)
        upper = corr.where(mask).stack().sort_values(ascending=False)
        if not upper.empty:
            pair     = upper.index[0]
            top_corr = {"cols": list(pair), "r": round(float(upper.iloc[0]), 2)}

    num_stats = {}
    if num_cols:
        desc = df[num_cols].describe().round(2)
        for col in num_cols:
            num_stats[col] = {
                "mean": float(desc.loc["mean", col]),
                "std":  float(desc.loc["std",  col]),
                "min":  float(desc.loc["min",  col]),
                "max":  float(desc.loc["max",  col]),
            }

    return _json_safe({
        "rows":        len(df),
        "cols":        len(df.columns),
        "numeric":     num_cols,
        "categorical": cat_cols,
        "nulls":       nulls,
        "top_corr":    top_corr,
        "num_stats":   num_stats,
        "preview":     df.head(5).to_dict(orient="records"),
    })
