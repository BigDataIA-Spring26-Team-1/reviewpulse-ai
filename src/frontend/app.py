"""
ReviewPulse AI Streamlit dashboard.

Run:
    poetry run streamlit run src/frontend/app.py
"""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

CONFIGURED_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_BASE_CANDIDATES = (
    CONFIGURED_API_BASE,
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
)
API_PROBE_TIMEOUT_SECONDS = 3
STATS_TIMEOUT_SECONDS = 180
DATA_TIMEOUT_SECONDS = 180
RAG_TIMEOUT_SECONDS = 120
DEFAULT_QUERY = "What do reviews say about customer service?"
DEFAULT_COMPARE_PRODUCTS = "Starbucks, McDonald's"
try:
    UI_SCAN_ROWS = max(1000, int(os.getenv("REVIEWPULSE_UI_SCAN_ROWS", "50000")))
except ValueError:
    UI_SCAN_ROWS = 50000
RAG_TIMEOUT_HELP = (
    "The review index is very large, so broad searches can take a while. "
    "Try fewer evidence items, add a source filter, or ask for a more specific aspect."
)

SOURCE_OPTIONS = ["all", "amazon", "yelp", "ebay", "ifixit", "youtube", "reddit"]
SOURCE_COLORS = {
    "amazon": "#d97706",
    "yelp": "#dc2626",
    "ebay": "#2563eb",
    "ifixit": "#059669",
    "youtube": "#b91c1c",
    "reddit": "#ea580c",
    "unknown": "#64748b",
}
SENTIMENT_COLORS = {
    "positive": "#059669",
    "negative": "#dc2626",
    "neutral": "#64748b",
    "mixed": "#d97706",
    "unknown": "#64748b",
}


st.set_page_config(
    page_title="ReviewPulse AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --rp-bg: #f4f7f6;
            --rp-panel: #ffffff;
            --rp-panel-strong: #eef8f5;
            --rp-ink: #132026;
            --rp-muted: #5f6f78;
            --rp-line: #dbe5e4;
            --rp-teal: #0f766e;
            --rp-blue: #2563eb;
            --rp-coral: #ef6f6c;
            --rp-amber: #d97706;
            --rp-red: #dc2626;
            --rp-green: #059669;
            --rp-shadow: 0 18px 45px rgba(19, 32, 38, 0.08);
        }

        .stApp {
            background:
                linear-gradient(135deg, rgba(244, 247, 246, 0.96), rgba(238, 248, 245, 0.92) 48%, rgba(255, 247, 237, 0.92));
            color: var(--rp-ink);
            font-size: 16px;
            line-height: 1.55;
        }

        [data-testid="stSidebar"] {
            background: #132026;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        [data-testid="stSidebar"] .stRadio > label,
        [data-testid="stSidebar"] .stSelectbox > label,
        [data-testid="stSidebar"] .stSlider > label {
            color: #cbd5e1;
        }

        [data-testid="stSidebar"] hr {
            border-color: rgba(255, 255, 255, 0.14);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1500px;
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        [data-testid="stMain"] {
            color: var(--rp-ink);
        }

        [data-testid="stMain"] p,
        [data-testid="stMain"] li,
        [data-testid="stMain"] label,
        [data-testid="stMain"] [data-testid="stMarkdownContainer"],
        [data-testid="stMain"] [data-testid="stWidgetLabel"],
        [data-testid="stMain"] [data-testid="stWidgetLabel"] p {
            color: var(--rp-ink) !important;
        }

        [data-testid="stMain"] [data-testid="stWidgetLabel"] p {
            font-size: 0.92rem;
            font-weight: 750;
        }

        [data-testid="stMain"] [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--rp-line);
            border-radius: 8px;
            color: var(--rp-ink);
        }

        [data-testid="stMain"] [data-testid="stExpander"] details > summary {
            background: #ffffff;
            border-radius: 8px;
            min-height: 3rem;
        }

        [data-testid="stMain"] [data-testid="stExpander"] details > summary p {
            color: var(--rp-ink) !important;
            font-weight: 800;
        }

        [data-testid="stMain"] .stTextInput input,
        [data-testid="stMain"] .stTextArea textarea,
        [data-testid="stMain"] .stNumberInput input,
        [data-testid="stMain"] div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1px solid #94a3b8 !important;
            color: var(--rp-ink) !important;
            box-shadow: none !important;
        }

        [data-testid="stMain"] .stTextInput input,
        [data-testid="stMain"] .stTextArea textarea,
        [data-testid="stMain"] .stNumberInput input {
            font-size: 1rem;
            min-height: 2.9rem;
        }

        [data-testid="stMain"] .stTextInput input::placeholder,
        [data-testid="stMain"] .stTextArea textarea::placeholder {
            color: #64748b !important;
            opacity: 1 !important;
        }

        [data-testid="stMain"] div[data-baseweb="select"] span,
        [data-testid="stMain"] div[data-baseweb="select"] svg {
            color: var(--rp-ink) !important;
            fill: #475569 !important;
        }

        div[data-baseweb="popover"] div[role="listbox"],
        div[data-baseweb="popover"] div[role="option"] {
            background: #ffffff !important;
            color: var(--rp-ink) !important;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stLinkButton"] > a {
            border-radius: 8px;
            border: 1px solid rgba(15, 118, 110, 0.18);
            font-weight: 700;
        }

        [data-testid="stMain"] div[data-testid="stButton"] > button,
        [data-testid="stMain"] div[data-testid="stLinkButton"] > a {
            background: #ffffff;
            border-color: #94a3b8;
            color: var(--rp-ink);
            min-height: 2.75rem;
        }

        [data-testid="stMain"] div[data-testid="stButton"] > button:hover,
        [data-testid="stMain"] div[data-testid="stLinkButton"] > a:hover {
            background: #eef8f5;
            border-color: var(--rp-teal);
            color: var(--rp-ink);
        }

        [data-testid="stMain"] button[data-testid="stBaseButton-primary"],
        [data-testid="stMain"] div[data-testid="stFormSubmitButton"] > button {
            background: var(--rp-teal) !important;
            border-color: var(--rp-teal) !important;
            color: #ffffff !important;
        }

        [data-testid="stMain"] button[data-testid="stBaseButton-primary"]:hover,
        [data-testid="stMain"] div[data-testid="stFormSubmitButton"] > button:hover {
            background: #115e59 !important;
            border-color: #115e59 !important;
            color: #ffffff !important;
        }

        [data-testid="stMain"] div[data-testid="stButton"] > button:disabled,
        [data-testid="stMain"] div[data-testid="stFormSubmitButton"] > button:disabled {
            background: #dce6e4 !important;
            border-color: #c5d4d1 !important;
            color: #40515a !important;
            opacity: 1 !important;
        }

        div[data-testid="stFormSubmitButton"] > button {
            background: #0f766e;
            color: #ffffff;
        }

        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #115e59;
            color: #ffffff;
            border-color: #115e59;
        }

        .rp-topbar {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1rem;
            align-items: end;
            padding: 1.35rem 1.45rem;
            background: rgba(255, 255, 255, 0.82);
            border: 1px solid var(--rp-line);
            border-radius: 8px;
            box-shadow: var(--rp-shadow);
            margin-bottom: 1rem;
        }

        .rp-title {
            font-size: 2rem;
            line-height: 1.05;
            margin: 0;
            font-weight: 850;
            color: var(--rp-ink);
        }

        .rp-subtitle {
            margin-top: 0.45rem;
            color: var(--rp-muted);
            font-size: 1rem;
        }

        .rp-api-pill {
            min-width: 210px;
            padding: 0.8rem 0.9rem;
            border: 1px solid var(--rp-line);
            border-radius: 8px;
            background: #f8fafc;
            text-align: right;
            color: var(--rp-muted);
            font-size: 0.84rem;
        }

        .rp-section-title {
            margin: 1.25rem 0 0.75rem;
            font-size: 1.25rem;
            font-weight: 800;
            color: var(--rp-ink);
        }

        .rp-panel {
            border: 1px solid var(--rp-line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            box-shadow: var(--rp-shadow);
            padding: 1.1rem;
            margin-bottom: 1rem;
        }

        .rp-answer {
            border-left: 5px solid var(--rp-teal);
            background: #ffffff;
            border-radius: 8px;
            padding: 1.1rem 1.2rem;
            box-shadow: var(--rp-shadow);
            margin: 1rem 0;
            font-size: 1.03rem;
            line-height: 1.62;
        }

        .rp-answer.blocked {
            border-left-color: var(--rp-red);
            background: #fff7f7;
        }

        .rp-answer.review {
            border-left-color: var(--rp-amber);
            background: #fffaf0;
        }

        .rp-metric-card {
            border: 1px solid var(--rp-line);
            border-radius: 8px;
            background: #ffffff;
            padding: 1rem;
            box-shadow: 0 10px 30px rgba(19, 32, 38, 0.06);
            min-height: 112px;
        }

        .rp-metric-label {
            color: var(--rp-muted);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .rp-metric-value {
            margin-top: 0.4rem;
            color: var(--rp-ink);
            font-size: 1.75rem;
            font-weight: 850;
            line-height: 1.1;
        }

        .rp-metric-note {
            margin-top: 0.45rem;
            color: var(--rp-muted);
            font-size: 0.82rem;
        }

        .rp-review-card {
            border: 1px solid var(--rp-line);
            border-radius: 8px;
            background: #ffffff;
            box-shadow: 0 14px 36px rgba(19, 32, 38, 0.07);
            padding: 1rem 1.05rem;
            margin: 0.75rem 0;
        }

        .rp-review-head {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: flex-start;
            margin-bottom: 0.65rem;
        }

        .rp-review-title {
            color: var(--rp-ink);
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .rp-review-meta {
            color: var(--rp-muted);
            font-size: 0.84rem;
        }

        .rp-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            justify-content: flex-start;
            margin: 0.55rem 0 0.75rem;
        }

        .rp-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 800;
            line-height: 1.1;
            background: #eef2f7;
            color: #334155;
            border: 1px solid rgba(51, 65, 85, 0.12);
        }

        .rp-review-text {
            color: #26343b;
            background: #f8fafc;
            border: 1px solid #e5edf0;
            border-radius: 8px;
            padding: 0.85rem;
            line-height: 1.55;
            margin-top: 0.75rem;
        }

        .rp-card-link {
            color: var(--rp-teal);
            font-weight: 800;
            text-decoration: none;
            white-space: nowrap;
        }

        .rp-card-link:hover {
            text-decoration: underline;
        }

        .rp-empty {
            border: 1px dashed #b7c6c7;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.7);
            padding: 1.1rem;
            color: var(--rp-muted);
        }

        .rp-brand {
            padding: 0.6rem 0 0.2rem;
        }

        .rp-brand-title {
            color: #ffffff;
            font-size: 1.32rem;
            font-weight: 850;
        }

        .rp-brand-subtitle {
            color: #a7f3d0;
            font-size: 0.84rem;
            margin-top: 0.2rem;
        }

        @media (max-width: 900px) {
            .rp-topbar {
                grid-template-columns: 1fr;
            }

            .rp-api-pill {
                text-align: left;
                min-width: auto;
            }

            .rp-title {
                font-size: 1.6rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def esc(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def format_count(value: Any) -> str:
    return f"{safe_int(value):,}"


def format_percent(value: Any) -> str:
    return f"{safe_float(value):.2%}"


def pretty_label(label: Any) -> str:
    text = str(label or "unknown").strip()
    return text.replace("_", " ").title()


def source_filter_value(selection: str) -> str:
    return "" if selection == "all" else selection


def api_error(payload: Any) -> str | None:
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    return None


@st.cache_data(ttl=15, show_spinner=False)
def resolve_api_base() -> str:
    candidates = list(dict.fromkeys(base for base in API_BASE_CANDIDATES if base))
    healthy_candidates: list[tuple[int, str]] = []
    for index, base_url in enumerate(candidates):
        try:
            response = requests.get(f"{base_url}/health", timeout=API_PROBE_TIMEOUT_SECONDS)
            if 200 <= response.status_code < 300:
                healthy_candidates.append((index, base_url))
        except requests.exceptions.RequestException:
            continue

    for index, base_url in healthy_candidates:
        try:
            response = requests.get(f"{base_url}/dashboard/summary", timeout=API_PROBE_TIMEOUT_SECONDS)
            if not 200 <= response.status_code < 300:
                continue
            payload = response.json()
            total_reviews = safe_int(payload.get("total_reviews")) if isinstance(payload, dict) else 0
            indexed_documents = safe_int(payload.get("indexed_documents")) if isinstance(payload, dict) else 0
            if total_reviews > 0 or indexed_documents > 0:
                return base_url
        except (ValueError, requests.exceptions.RequestException):
            continue

    if healthy_candidates:
        return min(healthy_candidates)[1]
    return CONFIGURED_API_BASE


def current_api_base() -> str:
    return resolve_api_base()


def api_get(path: str, *, params: dict[str, Any] | None = None, timeout: int = 30) -> Any:
    response = requests.get(f"{current_api_base()}{path}", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=15, show_spinner=False)
def load_health() -> dict[str, Any]:
    payload = api_get("/health", timeout=10)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=60, show_spinner=False)
def load_detailed_health() -> dict[str, Any]:
    payload = api_get("/health/detailed", timeout=25)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=300, show_spinner=False)
def load_stats() -> dict[str, Any]:
    payload = api_get("/stats/sources", timeout=STATS_TIMEOUT_SECONDS)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=600, show_spinner=False)
def load_data_profile() -> dict[str, Any]:
    payload = api_get("/data/profile", timeout=DATA_TIMEOUT_SECONDS)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=600, show_spinner=False)
def load_data_compare() -> dict[str, Any]:
    payload = api_get("/data/compare", timeout=DATA_TIMEOUT_SECONDS)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=600, show_spinner=False)
def load_data_quality() -> dict[str, Any]:
    payload = api_get("/data/quality", timeout=DATA_TIMEOUT_SECONDS)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=600, show_spinner=False)
def load_normalization_explanations(source: str | None = None) -> dict[str, Any]:
    params = {"source": source} if source else None
    payload = api_get("/data/normalize/explain", params=params, timeout=45)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=60, show_spinner=False)
def load_dashboard_summary() -> dict[str, Any]:
    payload = api_get("/dashboard/summary", timeout=30)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=300, show_spinner=False)
def load_review_options() -> dict[str, Any]:
    payload = api_get(
        "/reviews/options",
        params={"limit": 250, "max_rows": UI_SCAN_ROWS},
        timeout=DATA_TIMEOUT_SECONDS,
    )
    return payload if isinstance(payload, dict) else {}


def load_review_explorer(params: dict[str, Any]) -> dict[str, Any]:
    payload = api_get("/reviews/explore", params=params, timeout=DATA_TIMEOUT_SECONDS)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=300, show_spinner=False)
def load_sentiment_analytics(params: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    payload = api_get("/analytics/sentiment", params=dict(params), timeout=DATA_TIMEOUT_SECONDS)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=300, show_spinner=False)
def load_product_compare(params: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    payload = api_get("/products/compare", params=dict(params), timeout=DATA_TIMEOUT_SECONDS)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=300, show_spinner=False)
def load_aspect_summary(params: tuple[tuple[str, Any], ...]) -> dict[str, Any]:
    payload = api_get("/aspects/summary", params=dict(params), timeout=DATA_TIMEOUT_SECONDS)
    return payload if isinstance(payload, dict) else {}


@st.cache_data(ttl=60, show_spinner=False)
def load_pipeline_status() -> dict[str, Any]:
    payload = api_get("/pipeline/status", timeout=30)
    return payload if isinstance(payload, dict) else {}


def load_hitl_queue(status: str | None = "pending") -> list[dict[str, Any]]:
    params = {"limit": 250}
    if status:
        params["status"] = status
    payload = api_get("/hitl/queue", params=params, timeout=30)
    return payload if isinstance(payload, list) else []


def run_search(query: str, source_filter: str, n_results: int) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"query": query, "n_results": n_results}
    if source_filter:
        params["source_filter"] = source_filter
    payload = api_get("/search/semantic", params=params, timeout=RAG_TIMEOUT_SECONDS)
    return payload if isinstance(payload, list) else []


def run_chat(query: str, source_filter: str, n_results: int) -> dict[str, Any]:
    params: dict[str, Any] = {"query": query, "n_results": n_results}
    if source_filter:
        params["source_filter"] = source_filter
    payload = api_get("/chat", params=params, timeout=RAG_TIMEOUT_SECONDS)
    return payload if isinstance(payload, dict) else {}


def render_request_error(error: Exception, *, label: str) -> None:
    if isinstance(error, requests.exceptions.ReadTimeout):
        st.error(f"{label} timed out after {RAG_TIMEOUT_SECONDS} seconds.")
        st.info(RAG_TIMEOUT_HELP)
        return
    st.error(f"{label} failed: {error}")


def render_topbar(active_view: str) -> None:
    api_base = current_api_base()
    st.markdown(
        f"""
        <div class="rp-topbar">
            <div>
                <div class="rp-title">ReviewPulse AI</div>
                <div class="rp-subtitle">{esc(active_view)} across search, quality, and human review workflows.</div>
            </div>
            <div class="rp-api-pill">
                <strong>API</strong><br>{esc(api_base)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: Any, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="rp-metric-card">
            <div class="rp-metric-label">{esc(label)}</div>
            <div class="rp-metric-value">{esc(value)}</div>
            <div class="rp-metric-note">{esc(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def badge_html(label: str, color: str | None = None) -> str:
    if color:
        style = (
            f"background:{html.escape(color)}18;"
            f"color:{html.escape(color)};"
            f"border-color:{html.escape(color)}33;"
        )
    else:
        style = ""
    return f'<span class="rp-badge" style="{style}">{esc(label)}</span>'


def render_empty(message: str) -> None:
    st.markdown(f'<div class="rp-empty">{esc(message)}</div>', unsafe_allow_html=True)


def compact_params(params: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in params.items():
        if value is None or value == "" or value == "all" or value == []:
            continue
        cleaned[key] = value
    return cleaned


def params_key(params: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(sorted(compact_params(params).items()))


def product_rows_from_options(options: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in options.get("products", [])
        if row.get("product_name")
    ]


def product_option_label(option: dict[str, Any]) -> str:
    return str(option.get("product_label") or option.get("product_name") or "Unknown Product")


def render_answer(chat_response: dict[str, Any]) -> None:
    answer = str(chat_response.get("answer") or "")
    guardrail = chat_response.get("guardrail") or {}
    action = guardrail.get("action")
    card_class = "blocked" if action == "block" else "review" if action == "needs_human_review" else ""
    badges = []

    if guardrail:
        badges.append(
            badge_html(
                f"Guardrail: {pretty_label(guardrail.get('action'))}",
                "#d97706" if action == "needs_human_review" else "#0f766e",
            )
        )
        badges.append(badge_html(f"Confidence {safe_float(guardrail.get('confidence')):.2f}"))
    if chat_response.get("cache_hit"):
        badges.append(badge_html("Cache hit", "#2563eb"))
    if safe_int(chat_response.get("filtered_review_count")):
        badges.append(
            badge_html(
                f"Filtered {safe_int(chat_response.get('filtered_review_count'))} likely fake",
                "#dc2626",
            )
        )
    if chat_response.get("hitl_request_id"):
        badges.append(badge_html(f"HITL {chat_response.get('hitl_request_id')}", "#d97706"))

    badge_row = f'<div class="rp-badge-row">{"".join(badges)}</div>' if badges else ""
    st.markdown(
        f"""
        <div class="rp-answer {card_class}">
            {badge_row}
            <div>{esc(answer)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_review_card(item: dict[str, Any], *, index: int, label: str, include_distance: bool) -> None:
    source = str(item.get("source") or "unknown").lower()
    sentiment = str(item.get("sentiment_label") or "unknown").lower()
    title = item.get("display_name") or item.get("product_label") or item.get("product_name") or "Review"
    product_label = str(item.get("product_label") or item.get("product_name") or "").strip()
    category = item.get("display_category") or item.get("product_category") or "Uncategorized"
    entity_type = pretty_label(item.get("entity_type") or "review")
    aspect_labels = str(item.get("aspect_labels") or "").strip()
    aspect_count = safe_int(item.get("aspect_count"))
    url = str(item.get("source_url") or "").strip()

    badges = [
        badge_html(source.upper(), SOURCE_COLORS.get(source, SOURCE_COLORS["unknown"])),
        badge_html(pretty_label(sentiment), SENTIMENT_COLORS.get(sentiment, SENTIMENT_COLORS["unknown"])),
        badge_html(entity_type),
    ]
    if aspect_labels:
        aspect_note = f"{aspect_count} aspects" if aspect_count else "aspects"
        badges.append(badge_html(aspect_note, "#2563eb"))
    if include_distance and "distance" in item:
        badges.append(badge_html(f"Distance {safe_float(item.get('distance')):.4f}"))
    if item.get("fake_review_label"):
        risk_color = "#dc2626" if item.get("fake_review_label") == "likely_fake" else "#d97706"
        badges.append(
            badge_html(
                f"Risk {pretty_label(item.get('fake_review_label'))} {safe_float(item.get('fake_review_score')):.2f}",
                risk_color,
            )
        )

    link_html = ""
    if url:
        link_html = (
            f'<a class="rp-card-link" href="{esc(url)}" target="_blank" rel="noreferrer">'
            "Open source"
            "</a>"
        )

    aspect_html = ""
    if aspect_labels:
        aspect_html = f'<div class="rp-review-meta">Aspects: {esc(aspect_labels)}</div>'

    product_html = ""
    if product_label and product_label != str(title).strip():
        product_html = f'<div class="rp-review-meta">Product: {esc(product_label)}</div>'

    flags = item.get("fake_review_flags") or []
    flags_html = ""
    if flags:
        flags_html = f'<div class="rp-review-meta">Flags: {esc(", ".join(str(flag) for flag in flags))}</div>'

    review_text = str(item.get("review_text") or "").strip()
    review_text_html = ""
    if review_text:
        review_text_html = f'<div class="rp-review-text">{esc(review_text)}</div>'

    st.markdown(
        f"""
        <div class="rp-review-card">
            <div class="rp-review-head">
                <div>
                    <div class="rp-review-title">{esc(label)} {index}: {esc(title)}</div>
                    <div class="rp-review-meta">{esc(category)}</div>
                    {product_html}
                </div>
                {link_html}
            </div>
            <div class="rp-badge-row">{"".join(badges)}</div>
            {aspect_html}
            {flags_html}
            {review_text_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    api_base = current_api_base()
    st.sidebar.markdown(
        """
        <div class="rp-brand">
            <div class="rp-brand-title">ReviewPulse AI</div>
            <div class="rp-brand-subtitle">Review intelligence workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.divider()

    view = st.sidebar.radio(
        "Workspace",
        [
            "Dashboard Home",
            "Review Explorer",
            "Sentiment Analytics",
            "Product Comparison",
            "RAG Chatbot",
            "Aspect Intelligence",
            "HITL Queue",
            "Pipeline & Data Ops",
            "Settings / About",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.divider()
    try:
        health = load_health()
        status = str(health.get("status") or "unknown")
        status_color = "#059669" if status == "ok" else "#dc2626"
        st.sidebar.markdown(badge_html(f"API {status.upper()}", status_color), unsafe_allow_html=True)
        cache_state = "on" if health.get("cache_enabled") else "off"
        st.sidebar.caption(f"Cache {cache_state}")
        st.sidebar.caption(f"Guardrails {'on' if health.get('guardrails_enabled') else 'off'}")
    except Exception as error:
        st.sidebar.error(f"API offline: {error}")

    if st.sidebar.button("Refresh API status", width="stretch"):
        resolve_api_base.clear()
        load_health.clear()
        load_detailed_health.clear()
        load_dashboard_summary.clear()
        load_review_options.clear()
        load_sentiment_analytics.clear()
        load_product_compare.clear()
        load_aspect_summary.clear()
        load_pipeline_status.clear()
        st.rerun()

    st.sidebar.link_button("Swagger docs", f"{api_base}/docs", width="stretch")
    st.sidebar.link_button("OpenAPI schema", f"{api_base}/openapi.json", width="stretch")

    with st.sidebar.expander("Dependency health"):
        if st.button("Check dependencies", width="stretch"):
            load_detailed_health.clear()
            with st.spinner("Checking dependencies..."):
                try:
                    detailed = load_detailed_health()
                    for name, dependency in (detailed.get("dependencies") or {}).items():
                        dep_status = str(dependency.get("status") or "unknown")
                        color = {
                            "healthy": "#059669",
                            "degraded": "#d97706",
                            "skipped": "#64748b",
                            "unhealthy": "#dc2626",
                        }.get(dep_status, "#64748b")
                        st.markdown(
                            badge_html(f"{name}: {dep_status}", color),
                            unsafe_allow_html=True,
                        )
                        st.caption(str(dependency.get("message") or ""))
                except Exception as error:
                    st.error(f"Dependency check failed: {error}")

    return view


def render_example_prompts() -> None:
    examples = [
        "What do reviews say about customer service?",
        "Which sources mention battery life problems?",
        "Are repair experiences positive or negative?",
        "What are common complaints about shipping?",
    ]
    columns = st.columns(len(examples))
    for column, prompt in zip(columns, examples):
        with column:
            if st.button(prompt, width="stretch"):
                st.session_state["ask_query"] = prompt
                st.rerun()


def render_dashboard_home() -> None:
    try:
        summary = load_dashboard_summary()
        health = load_health()
    except Exception as error:
        st.error(f"Dashboard request failed: {error}")
        return

    if error := api_error(summary):
        st.info(error)
        return

    metric_cols = st.columns(6)
    with metric_cols[0]:
        render_metric_card("Reviews", format_count(summary.get("total_reviews")), "processed parquet rows")
    with metric_cols[1]:
        render_metric_card("Sources", format_count(summary.get("sources_integrated")), "local artifacts")
    with metric_cols[2]:
        products_value = summary.get("products_tracked")
        render_metric_card("Products", format_count(products_value) if products_value is not None else "Load filters", "available in explorer")
    with metric_cols[3]:
        indexed = summary.get("indexed_documents")
        render_metric_card("Indexed Docs", format_count(indexed) if indexed is not None else "Not built", "vector store")
    with metric_cols[4]:
        render_metric_card("HITL Items", format_count(summary.get("hitl_queue_items")), "human review queue")
    with metric_cols[5]:
        render_metric_card("API", str(health.get("status", "unknown")).upper(), current_api_base())

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="rp-section-title">System Status</div>', unsafe_allow_html=True)
        cache_label = "Enabled" if health.get("cache_enabled") else "Disabled"
        status_rows = [
            {"component": "FastAPI", "status": health.get("status", "unknown"), "details": current_api_base()},
            {"component": "Vector DB", "status": "available" if summary.get("indexed_documents") is not None else "missing", "details": summary.get("indexed_documents")},
            {"component": "Cache", "status": cache_label, "details": health.get("cache_disabled_reason") or ""},
            {"component": "Data freshness", "status": "available" if summary.get("latest_dataset_update") else "unknown", "details": summary.get("latest_dataset_update")},
        ]
        st.dataframe(pd.DataFrame(status_rows), width="stretch", hide_index=True)

    with right:
        st.markdown('<div class="rp-section-title">Data Artifacts</div>', unsafe_allow_html=True)
        artifacts = pd.DataFrame(summary.get("artifact_dirs") or [])
        if not artifacts.empty:
            st.dataframe(artifacts, width="stretch", hide_index=True)
        else:
            render_empty("No local data artifacts were found.")

    st.markdown('<div class="rp-section-title">Demo Flow</div>', unsafe_allow_html=True)
    st.markdown(
        """
        1. Start with **Review Explorer** to show real normalized reviews.
        2. Use **RAG Chatbot** for grounded answers with citations.
        3. Open **Sentiment Analytics** and **Aspect Intelligence** when you want aggregate charts.
        4. Use **Pipeline & Data Ops** to show ingestion, parquet, Airflow, and vector-store readiness.
        """
    )


def render_review_explorer() -> None:
    with st.expander("Filters", expanded=True):
        filter_cols = st.columns([1.2, 1, 1, 1])
        with filter_cols[0]:
            query = st.text_input("Search text", placeholder="battery, customer service, repair...")
        with filter_cols[1]:
            source = st.selectbox("Source", SOURCE_OPTIONS, key="explorer_source")
        with filter_cols[2]:
            sentiment = st.selectbox("Sentiment", ["all", "positive", "negative", "neutral"], key="explorer_sentiment")
        with filter_cols[3]:
            limit = st.slider("Rows", min_value=5, max_value=100, value=25, step=5)

        detail_cols = st.columns([1, 1, 1])
        with detail_cols[0]:
            product = st.text_input("Product exact match", placeholder="Optional exact product key")
        with detail_cols[1]:
            category = st.text_input("Category contains", placeholder="Optional category")
        with detail_cols[2]:
            min_rating, max_rating = st.slider("Normalized rating", 0.0, 1.0, (0.0, 1.0), 0.05)

        offset = st.number_input("Offset", min_value=0, max_value=1000000, value=0, step=limit)
        run_clicked = st.button("Search reviews", type="primary", width="stretch")

    if run_clicked:
        params = compact_params(
            {
                "query": query,
                "source": source,
                "sentiment": sentiment,
                "product": product,
                "category": category,
                "min_rating": min_rating if min_rating > 0 else None,
                "max_rating": max_rating if max_rating < 1 else None,
                "limit": limit,
                "offset": offset,
            }
        )
        with st.spinner("Searching normalized reviews..."):
            try:
                st.session_state["explorer_payload"] = load_review_explorer(params)
            except Exception as error:
                render_request_error(error, label="Review explorer request")
                return

    payload = st.session_state.get("explorer_payload")
    if not payload:
        render_empty("Choose filters and search to inspect real normalized review rows.")
        return
    if error := api_error(payload):
        st.info(error)
        return

    rows = payload.get("rows") or []
    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("Returned", format_count(len(rows)), "visible rows")
    with metric_cols[1]:
        render_metric_card("Matched", format_count(payload.get("matched_count")), "estimated during scan")
    with metric_cols[2]:
        render_metric_card("Has More", "Yes" if payload.get("has_more") else "No", "increase offset")

    if not rows:
        render_empty("No reviews found for the selected filters.")
        return

    table_cols = [
        "review_id",
        "source",
        "product_label",
        "rating_normalized",
        "sentiment_label",
        "aspect_labels",
        "review_date",
        "helpful_votes",
        "source_url",
    ]
    table_df = pd.DataFrame(rows)
    if "product_label" not in table_df.columns:
        table_df["product_label"] = ""
    for fallback_column in ("product_title", "product_name", "display_name"):
        if fallback_column in table_df.columns:
            fallback_values = table_df[fallback_column].fillna("").astype(str)
            table_df["product_label"] = table_df["product_label"].fillna("").astype(str)
            table_df["product_label"] = table_df["product_label"].where(
                table_df["product_label"].str.strip().ne(""),
                fallback_values,
            )
    for table_column in table_cols:
        if table_column not in table_df.columns:
            table_df[table_column] = ""
    st.dataframe(table_df[table_cols], width="stretch", hide_index=True)
    for index, row in enumerate(rows[:10], start=1):
        render_review_card(row, index=index, label="Review", include_distance=False)


def render_sentiment_analytics_view() -> None:
    with st.expander("Analytics filters", expanded=True):
        cols = st.columns(4)
        with cols[0]:
            source = st.selectbox("Source", SOURCE_OPTIONS, key="sent_source")
        with cols[1]:
            sentiment = st.selectbox("Sentiment", ["all", "positive", "negative", "neutral"], key="sent_sentiment")
        with cols[2]:
            product = st.text_input("Product exact match", key="sent_product")
        with cols[3]:
            category = st.text_input("Category contains", key="sent_category")
        query = st.text_input("Text/aspect filter", key="sent_query")
        load_clicked = st.button("Load sentiment analytics", type="primary", width="stretch")

    if load_clicked:
        params = params_key(
            {
                "source": source,
                "sentiment": sentiment,
                "product": product,
                "category": category,
                "query": query,
                "max_rows": UI_SCAN_ROWS,
            }
        )
        with st.spinner("Computing sentiment analytics..."):
            try:
                st.session_state["sentiment_payload"] = load_sentiment_analytics(params)
            except Exception as error:
                st.error(f"Sentiment analytics request failed: {error}")
                return

    payload = st.session_state.get("sentiment_payload")
    if not payload:
        render_empty("Load sentiment analytics to see distribution, rating buckets, source mix, and trends.")
        return
    if error := api_error(payload):
        st.info(error)
        return

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("Filtered Reviews", format_count(payload.get("record_count")), "matched rows")
    with metric_cols[1]:
        render_metric_card("Rows Scanned", format_count(payload.get("rows_scanned")), "real parquet rows")
    with metric_cols[2]:
        scan_mode = "Capped" if payload.get("scan_limited") else "Full"
        render_metric_card("Scan Mode", scan_mode, f"limit {format_count(payload.get('max_rows'))}")
    top_left, top_right = st.columns(2)
    with top_left:
        st.markdown('<div class="rp-section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
        df = pd.DataFrame(payload.get("sentiment_distribution") or [])
        if not df.empty:
            st.bar_chart(df.set_index("sentiment_label")[["count"]])
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            render_empty("No sentiment labels found for this filter.")
    with top_right:
        st.markdown('<div class="rp-section-title">Rating Distribution</div>', unsafe_allow_html=True)
        df = pd.DataFrame(payload.get("rating_distribution") or [])
        if not df.empty:
            st.bar_chart(df.set_index("rating_bucket")[["count"]])
        else:
            render_empty("No rating data found for this filter.")

    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.markdown('<div class="rp-section-title">Source-wise Sentiment</div>', unsafe_allow_html=True)
        df = pd.DataFrame(payload.get("source_sentiment") or [])
        if not df.empty:
            pivot = df.pivot(index="source", columns="sentiment_label", values="count").fillna(0)
            st.bar_chart(pivot)
        else:
            render_empty("No source sentiment rows found.")
    with bottom_right:
        st.markdown('<div class="rp-section-title">Monthly Trend</div>', unsafe_allow_html=True)
        df = pd.DataFrame(payload.get("monthly_trend") or [])
        if not df.empty:
            pivot = df.pivot(index="month", columns="sentiment_label", values="count").fillna(0)
            st.line_chart(pivot)
        else:
            render_empty("No date trend data found.")

    st.markdown('<div class="rp-section-title">Top Aspects</div>', unsafe_allow_html=True)
    df = pd.DataFrame(payload.get("top_aspects") or [])
    if not df.empty:
        st.bar_chart(df.set_index("aspect")[["count"]])
        st.dataframe(df, width="stretch", hide_index=True)
    else:
        render_empty("No extracted aspects found for this filter.")


def render_product_comparison_view() -> None:
    st.session_state.setdefault("compare_manual_products", DEFAULT_COMPARE_PRODUCTS)
    st.session_state.setdefault("compare_product_choices", [])

    if st.button("Load product options", width="stretch"):
        with st.spinner("Loading product names..."):
            try:
                options_payload = load_review_options()
                st.session_state["review_options"] = options_payload
                option_rows = product_rows_from_options(options_payload)
                option_keys = [str(row.get("product_name") or "") for row in option_rows if row.get("product_name")]
                if len(option_keys) >= 2:
                    st.session_state["compare_product_choices"] = option_keys[:2]
                    st.session_state["compare_manual_products"] = ""
            except Exception as error:
                st.error(f"Product options request failed: {error}")
                return

    options = st.session_state.get("review_options") or {}
    product_rows = product_rows_from_options(options)
    product_lookup = {
        str(row.get("product_name") or ""): row
        for row in product_rows
        if row.get("product_name")
    }
    product_options = list(product_lookup)
    if product_options:
        if options.get("scan_limited"):
            st.caption(f"Product options come from the first {format_count(options.get('rows_scanned'))} real parquet rows.")

        existing_choices = [
            choice for choice in st.session_state.get("compare_product_choices", []) if choice in product_options
        ]
        if not existing_choices:
            st.session_state["compare_product_choices"] = product_options[:2]

        selected_options = st.multiselect(
            "Suggested products",
            product_options,
            key="compare_product_choices",
            format_func=lambda key: product_option_label(product_lookup.get(str(key), {})),
            help="Pick products from the loaded real dataset with clean labels.",
        )
    else:
        selected_options = []
        st.caption("Use the examples below, or load product options to choose from the dataset.")

    with st.expander("Advanced exact product keys"):
        product_text = st.text_input(
            "Products to compare",
            key="compare_manual_products",
            placeholder="Starbucks, McDonald's",
            help="Use this only when you want to enter exact internal product keys manually.",
        )
    selected = selected_options or [part.strip() for part in product_text.split(",") if part.strip()]

    cols = st.columns(2)
    with cols[0]:
        source = st.selectbox("Source", SOURCE_OPTIONS, key="compare_source")
    with cols[1]:
        category = st.text_input("Category contains", key="compare_category")

    if st.button("Compare products", type="primary", width="stretch"):
        if not selected:
            st.warning("Choose or enter at least one product.")
            return
        params = params_key(
            {
                "products": ",".join(selected),
                "source": source,
                "category": category,
                "max_rows": UI_SCAN_ROWS,
            }
        )
        with st.spinner("Comparing products..."):
            try:
                st.session_state["compare_payload"] = load_product_compare(params)
            except Exception as error:
                st.error(f"Product comparison request failed: {error}")
                return

    payload = st.session_state.get("compare_payload")
    if not payload:
        render_empty("Enter product names or load suggested options, then compare review-level aggregates.")
        return
    if error := api_error(payload):
        st.info(error)
        return

    rows = payload.get("products") or []
    if not rows:
        render_empty("No product comparison data found.")
        return
    missing_rows = [row for row in rows if safe_int(row.get("review_count")) == 0]
    rows = [row for row in rows if safe_int(row.get("review_count")) > 0]
    if missing_rows:
        missing_names = ", ".join(str(row.get("product_label") or row.get("product_name")) for row in missing_rows)
        st.warning(f"No matching reviews found for: {missing_names}. Try removing source/category filters or choose loaded suggestions.")
    if not rows:
        render_empty("No matching review rows found for this comparison. Try Starbucks, McDonald's, or choose suggested products.")
        return

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("Products", format_count(len(rows)), "requested")
    with metric_cols[1]:
        render_metric_card("Rows Scanned", format_count(payload.get("rows_scanned")), "real parquet rows")
    with metric_cols[2]:
        scan_mode = "Capped" if payload.get("scan_limited") else "Full"
        render_metric_card("Scan Mode", scan_mode, f"limit {format_count(payload.get('max_rows'))}")

    summary_rows = [
        {
            "product_label": row.get("product_label") or row.get("product_name"),
            "review_count": row.get("review_count"),
            "average_rating_normalized": row.get("average_rating_normalized"),
            "average_sentiment_score": row.get("average_sentiment_score"),
            "latest_review_date": row.get("latest_review_date"),
        }
        for row in rows
    ]
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)
    chart_df = pd.DataFrame(summary_rows).set_index("product_label")
    st.bar_chart(chart_df[["review_count"]])

    card_cols = st.columns(min(3, max(1, len(rows))))
    for index, row in enumerate(rows):
        with card_cols[index % len(card_cols)]:
            render_metric_card(row.get("product_label") or row.get("product_name"), format_count(row.get("review_count")), "reviews")
            st.caption(f"Avg rating: {row.get('average_rating_normalized')}")
            st.caption(f"Avg sentiment: {row.get('average_sentiment_score')}")
            positive = ", ".join(item["aspect"] for item in row.get("top_positive_aspects", [])[:3]) or "None"
            negative = ", ".join(item["aspect"] for item in row.get("top_negative_aspects", [])[:3]) or "None"
            st.write(f"Positive aspects: {positive}")
            st.write(f"Negative aspects: {negative}")


def render_aspect_intelligence_view() -> None:
    cols = st.columns(4)
    with cols[0]:
        source = st.selectbox("Source", SOURCE_OPTIONS, key="aspect_source")
    with cols[1]:
        sentiment = st.selectbox("Sentiment", ["all", "positive", "negative", "neutral"], key="aspect_sentiment")
    with cols[2]:
        product = st.text_input("Product exact match", key="aspect_product")
    with cols[3]:
        category = st.text_input("Category contains", key="aspect_category")
    query = st.text_input("Text/aspect filter", key="aspect_query")

    if st.button("Load aspect intelligence", type="primary", width="stretch"):
        params = params_key(
            {
                "source": source,
                "sentiment": sentiment,
                "product": product,
                "category": category,
                "query": query,
                "max_rows": UI_SCAN_ROWS,
            }
        )
        with st.spinner("Summarizing aspects..."):
            try:
                st.session_state["aspect_payload"] = load_aspect_summary(params)
            except Exception as error:
                st.error(f"Aspect summary request failed: {error}")
                return

    payload = st.session_state.get("aspect_payload")
    if not payload:
        render_empty("Load aspect intelligence to see aspect-level sentiment breakdowns.")
        return
    if error := api_error(payload):
        st.info(error)
        return

    metric_cols = st.columns(3)
    with metric_cols[0]:
        render_metric_card("Rows With Aspects", format_count(payload.get("rows_with_aspects")), "filtered rows")
    with metric_cols[1]:
        render_metric_card("Rows Scanned", format_count(payload.get("rows_scanned")), "real parquet rows")
    with metric_cols[2]:
        scan_mode = "Capped" if payload.get("scan_limited") else "Full"
        render_metric_card("Scan Mode", scan_mode, f"limit {format_count(payload.get('max_rows'))}")
    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown('<div class="rp-section-title">Top Positive Aspects</div>', unsafe_allow_html=True)
        df = pd.DataFrame(payload.get("top_positive_aspects") or [])
        if not df.empty:
            st.bar_chart(df.set_index("aspect")[["positive_count"]])
            st.dataframe(df[["aspect", "count", "positive_count", "average_sentiment_score"]], width="stretch", hide_index=True)
        else:
            render_empty("No positive aspect data found.")
    with neg_col:
        st.markdown('<div class="rp-section-title">Top Negative Aspects</div>', unsafe_allow_html=True)
        df = pd.DataFrame(payload.get("top_negative_aspects") or [])
        if not df.empty:
            st.bar_chart(df.set_index("aspect")[["negative_count"]])
            st.dataframe(df[["aspect", "count", "negative_count", "average_sentiment_score"]], width="stretch", hide_index=True)
        else:
            render_empty("No negative aspect data found.")

    st.markdown('<div class="rp-section-title">Aspect Breakdown</div>', unsafe_allow_html=True)
    df = pd.DataFrame(payload.get("aspects") or [])
    if not df.empty:
        st.dataframe(df.drop(columns=["example"], errors="ignore"), width="stretch", hide_index=True)
    else:
        render_empty("No aspect rows found.")


def render_ask_view() -> None:
    st.session_state.setdefault("ask_query", DEFAULT_QUERY)
    render_example_prompts()

    with st.form("ask_form", clear_on_submit=False):
        query = st.text_area(
            "Question",
            key="ask_query",
            height=110,
            placeholder="Ask about products, sources, sentiment, aspects, or review evidence.",
        )

        control_left, control_middle, control_right = st.columns([1.15, 1, 0.9])
        with control_left:
            source_choice = st.selectbox("Source", SOURCE_OPTIONS, index=0)
        with control_middle:
            n_results = st.slider("Evidence", min_value=1, max_value=10, value=3)
        with control_right:
            submit = st.form_submit_button("Ask ReviewPulse", width="stretch")

    if submit:
        if not query.strip():
            st.warning("Enter a question first.")
            return

        with st.spinner("Retrieving grounded evidence..."):
            try:
                source_filter = source_filter_value(source_choice)
                chat_response = run_chat(query.strip(), source_filter, n_results)
                st.session_state["last_ask_result"] = {
                    "query": query.strip(),
                    "source": source_choice,
                    "source_filter": source_filter,
                    "n_results": n_results,
                    "chat": chat_response,
                    "search": [],
                    "search_loaded": False,
                }
            except Exception as error:
                render_request_error(error, label="Answer request")
                return

    result = st.session_state.get("last_ask_result")
    if not result:
        render_empty("Ask a question to see grounded answers, citations, and retrieved review evidence.")
        return

    chat_response = result.get("chat") or {}
    search_results = result.get("search") or []
    citations = chat_response.get("citations") or []

    render_answer(chat_response)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Citations", format_count(len(citations)), "grounded references")
    with metric_cols[1]:
        search_state = "loaded" if result.get("search_loaded") else "not loaded"
        render_metric_card("Retrieved", format_count(len(search_results)), search_state)
    with metric_cols[2]:
        render_metric_card("Source", pretty_label(result.get("source")), "active filter")
    with metric_cols[3]:
        render_metric_card("Filtered", format_count(chat_response.get("filtered_review_count")), "fake-review risk")

    left, right = st.columns([0.95, 1.05])
    with left:
        st.markdown('<div class="rp-section-title">Citations</div>', unsafe_allow_html=True)
        if citations:
            for index, item in enumerate(citations, start=1):
                render_review_card(item, index=index, label="Citation", include_distance=False)
        else:
            render_empty("No citations returned for this answer.")

    with right:
        st.markdown('<div class="rp-section-title">Retrieved Reviews</div>', unsafe_allow_html=True)
        guardrail = chat_response.get("guardrail") or {}
        can_load_search = guardrail.get("action") in {None, "allow"}
        if can_load_search and not result.get("search_loaded"):
            if st.button("Load supporting reviews", width="stretch"):
                with st.spinner("Loading supporting reviews..."):
                    try:
                        search_results = run_search(
                            str(result.get("query") or ""),
                            str(result.get("source_filter") or ""),
                            safe_int(result.get("n_results"), 3),
                        )
                        result["search"] = search_results
                        result["search_loaded"] = True
                        st.session_state["last_ask_result"] = result
                        st.rerun()
                    except Exception as error:
                        render_request_error(error, label="Supporting review request")
                        return
        if search_results:
            for index, item in enumerate(search_results, start=1):
                render_review_card(item, index=index, label="Result", include_distance=True)
        elif can_load_search and not result.get("search_loaded"):
            render_empty("Supporting review cards are optional. Load them when you want the extra retrieval view.")
        else:
            render_empty("No supporting search results returned.")


def render_source_overview() -> None:
    action_left, action_right = st.columns([1, 3])
    with action_left:
        load_clicked = st.button("Load source stats", type="primary", width="stretch")
    with action_right:
        st.caption("First load scans the sentiment parquet dataset; later reruns use the app cache.")

    if load_clicked:
        load_stats.clear()
        with st.spinner("Building source overview..."):
            try:
                st.session_state["source_stats"] = load_stats()
            except Exception as error:
                st.error(f"Source stats request failed: {error}")
                return

    stats = st.session_state.get("source_stats")
    if not stats:
        render_empty("Load source stats when you want the full source and sentiment overview.")
        return

    stats_error = api_error(stats)
    if stats_error:
        st.info(stats_error)
        return

    source_counts = stats.get("source_counts", [])
    sentiment_breakdown = stats.get("sentiment_breakdown", [])
    top_aspects = stats.get("top_aspects", [])

    df_counts = pd.DataFrame(source_counts)
    df_sentiment = pd.DataFrame(sentiment_breakdown)
    df_top_aspects = pd.DataFrame(top_aspects)

    total_reviews = safe_int(df_counts["count"].sum()) if not df_counts.empty else 0
    total_sources = safe_int(df_counts["source"].nunique()) if not df_counts.empty else 0
    positive_reviews = 0
    negative_reviews = 0
    if not df_sentiment.empty:
        positive_reviews = safe_int(
            df_sentiment.loc[df_sentiment["sentiment_label"] == "positive", "count"].sum()
        )
        negative_reviews = safe_int(
            df_sentiment.loc[df_sentiment["sentiment_label"] == "negative", "count"].sum()
        )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    with metric_1:
        render_metric_card("Total Reviews", format_count(total_reviews), "sentiment dataset")
    with metric_2:
        render_metric_card("Sources", format_count(total_sources), "active channels")
    with metric_3:
        render_metric_card("Positive", format_count(positive_reviews), "review sentiment")
    with metric_4:
        render_metric_card("Negative", format_count(negative_reviews), "review sentiment")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="rp-section-title">Source Volume</div>', unsafe_allow_html=True)
        if not df_counts.empty:
            df_counts = df_counts.sort_values("count", ascending=False)
            st.bar_chart(df_counts.set_index("source")[["count"]])
            st.dataframe(df_counts, width="stretch", hide_index=True)
        else:
            render_empty("No source count data available.")

    with right:
        st.markdown('<div class="rp-section-title">Sentiment Mix</div>', unsafe_allow_html=True)
        if not df_sentiment.empty:
            pivot = df_sentiment.pivot(
                index="source",
                columns="sentiment_label",
                values="count",
            ).fillna(0)
            st.bar_chart(pivot)
            df_sentiment_display = df_sentiment.copy()
            df_sentiment_display["sentiment_label"] = df_sentiment_display[
                "sentiment_label"
            ].map(pretty_label)
            st.dataframe(df_sentiment_display, width="stretch", hide_index=True)
        else:
            render_empty("No sentiment data available.")

    st.markdown('<div class="rp-section-title">Top Aspects</div>', unsafe_allow_html=True)
    if not df_top_aspects.empty:
        df_top_aspects = df_top_aspects.sort_values("count", ascending=False).head(25)
        st.bar_chart(df_top_aspects.set_index("aspect")[["count"]])
        st.dataframe(df_top_aspects, width="stretch", hide_index=True)
    else:
        render_empty("No aspect data available.")


def render_data_insights() -> None:
    action_left, action_right = st.columns([1, 3])
    with action_left:
        load_clicked = st.button("Load data profile", type="primary", width="stretch")
    with action_right:
        st.caption("This reads the normalized parquet dataset and can take a while on the first run.")

    if load_clicked:
        load_data_profile.clear()
        load_data_compare.clear()
        load_data_quality.clear()
        with st.spinner("Profiling dataset..."):
            try:
                st.session_state["data_profile"] = load_data_profile()
                st.session_state["data_compare"] = load_data_compare()
                st.session_state["data_quality"] = load_data_quality()
            except Exception as error:
                st.error(f"Data insights request failed: {error}")
                return

    profile = st.session_state.get("data_profile")
    comparison = st.session_state.get("data_compare")
    quality = st.session_state.get("data_quality")
    if not (profile and comparison and quality):
        render_empty("Load the data profile when you need dataset quality, source comparison, and normalization details.")
        return

    for payload in (profile, comparison, quality):
        payload_error = api_error(payload)
        if payload_error:
            st.info(payload_error)
            return

    dataset_path = profile.get("dataset_path")
    if dataset_path:
        st.caption(f"Dataset: {dataset_path}")

    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card("Records", format_count(profile.get("record_count")), "normalized rows")
    with metric_cols[1]:
        render_metric_card("Sources", format_count(profile.get("source_count")), "available sources")
    with metric_cols[2]:
        render_metric_card("Products", format_count(profile.get("distinct_products")), "distinct names")
    with metric_cols[3]:
        render_metric_card("Reviewers", format_count(profile.get("distinct_reviewers")), "distinct ids")
    with metric_cols[4]:
        render_metric_card(
            "Avg Length",
            f"{safe_float(profile.get('average_review_length_words')):.1f}",
            "words per review",
        )

    rating_summary = profile.get("rating_normalized") or {}
    date_range = profile.get("date_range") or {}
    profile_left, profile_right = st.columns([0.9, 1.1])
    with profile_left:
        st.markdown('<div class="rp-section-title">Dataset Profile</div>', unsafe_allow_html=True)
        rating_cols = st.columns(3)
        with rating_cols[0]:
            render_metric_card("Min Rating", rating_summary.get("min"), "normalized")
        with rating_cols[1]:
            render_metric_card("Mean Rating", rating_summary.get("mean"), "normalized")
        with rating_cols[2]:
            render_metric_card("Max Rating", rating_summary.get("max"), "normalized")
        st.caption(f"Date range: {date_range.get('min')} to {date_range.get('max')}")

    with profile_right:
        st.markdown('<div class="rp-section-title">Missing Fields</div>', unsafe_allow_html=True)
        null_counts = pd.DataFrame(
            [
                {"field": field_name, "null_count": null_count}
                for field_name, null_count in (profile.get("null_counts") or {}).items()
            ]
        )
        if not null_counts.empty:
            null_counts = null_counts.sort_values("null_count", ascending=False)
            st.bar_chart(null_counts.set_index("field")[["null_count"]])
            st.dataframe(null_counts, width="stretch", hide_index=True)
        else:
            render_empty("No null count data available.")

    st.markdown('<div class="rp-section-title">Source Comparison</div>', unsafe_allow_html=True)
    df_compare = pd.DataFrame(comparison.get("sources") or [])
    if not df_compare.empty:
        st.dataframe(df_compare, width="stretch", hide_index=True)
        comparison_left, comparison_middle, comparison_right = st.columns(3)
        with comparison_left:
            st.caption("Average review length")
            st.bar_chart(df_compare.set_index("source")[["average_review_length_words"]])
        with comparison_middle:
            rating_compare = df_compare[["source", "average_rating_normalized"]].fillna(0.0)
            st.caption("Average normalized rating")
            st.bar_chart(rating_compare.set_index("source"))
        with comparison_right:
            missing_compare = df_compare[["source", "missing_value_ratio"]].fillna(0.0)
            st.caption("Missing value ratio")
            st.bar_chart(missing_compare.set_index("source"))
    else:
        render_empty("No source comparison data available.")

    st.markdown('<div class="rp-section-title">Data Quality</div>', unsafe_allow_html=True)
    quality_cols = st.columns(4)
    with quality_cols[0]:
        render_metric_card("Duplicates", format_percent(quality.get("duplicate_ratio")), "exact text match")
    with quality_cols[1]:
        render_metric_card("Nulls", format_percent(quality.get("null_ratio")), "schema cells")
    with quality_cols[2]:
        render_metric_card("Empty Text", format_percent(quality.get("empty_text_ratio")), "review body")
    with quality_cols[3]:
        render_metric_card("Missing Rating", format_percent(quality.get("missing_rating_ratio")), "rating field")
    st.caption(str(quality.get("duplicate_method") or ""))

    df_quality = pd.DataFrame(quality.get("per_source") or [])
    if not df_quality.empty:
        quality_chart = df_quality[
            ["source", "duplicate_ratio", "empty_text_ratio", "missing_rating_ratio"]
        ]
        st.bar_chart(quality_chart.set_index("source"))
        st.dataframe(df_quality, width="stretch", hide_index=True)
    else:
        render_empty("No per-source quality summary available.")

    st.markdown('<div class="rp-section-title">Normalization</div>', unsafe_allow_html=True)
    norm_left, norm_right = st.columns([1, 3])
    with norm_left:
        normalization_source = st.selectbox(
            "Source",
            ["all", "amazon", "yelp", "ebay", "ifixit", "youtube"],
            key="normalization_source",
        )
        normalize_clicked = st.button("Load normalization", width="stretch")
    with norm_right:
        if normalize_clicked:
            selected_source = None if normalization_source == "all" else normalization_source
            load_normalization_explanations.clear()
            with st.spinner("Loading normalization details..."):
                try:
                    st.session_state["normalization_payload"] = load_normalization_explanations(
                        selected_source
                    )
                except Exception as error:
                    st.error(f"Normalization request failed: {error}")
                    return

        payload = st.session_state.get("normalization_payload")
        if payload:
            explanation_error = api_error(payload)
            if explanation_error:
                st.info(explanation_error)
            else:
                explanations = payload.get("explanations") or []
                if not explanations:
                    render_empty("No normalization explanations available.")
                for explanation in explanations:
                    render_normalization_card(explanation)
        else:
            render_empty("Select a source and load normalization details.")


def render_normalization_card(explanation: dict[str, Any]) -> None:
    source = pretty_label(explanation.get("source"))
    identifiers = explanation.get("sample_identifiers") or {}
    sample_found = bool(explanation.get("sample_found"))
    identifier_text = ", ".join(f"{key}: {value}" for key, value in identifiers.items()) or "No sample identifiers"
    sample_status = "Sample found" if sample_found else "Formula only"
    color = "#059669" if sample_found else "#64748b"

    st.markdown(
        f"""
        <div class="rp-review-card">
            <div class="rp-review-head">
                <div>
                    <div class="rp-review-title">{esc(source)}</div>
                    <div class="rp-review-meta">{esc(identifier_text)}</div>
                </div>
                {badge_html(sample_status, color)}
            </div>
            <div class="rp-badge-row">
                {badge_html("Rating: " + str(explanation.get("raw_rating_field") or "none"))}
                {badge_html("Date: " + str(explanation.get("raw_date_field") or "none"))}
                {badge_html(str(explanation.get("raw_scale") or "raw scale unknown"), "#2563eb")}
            </div>
            <div class="rp-review-text">
                <strong>Formula:</strong> {esc(explanation.get("formula"))}<br>
                <strong>Rating:</strong> {esc(explanation.get("sample_raw_rating_value"))}
                to {esc(explanation.get("sample_normalized_rating"))}<br>
                <strong>Date:</strong> {esc(explanation.get("sample_raw_date_value"))}
                to {esc(explanation.get("sample_normalized_review_date"))}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hitl_view() -> None:
    control_left, control_right = st.columns([1, 3])
    with control_left:
        status_filter = st.selectbox("Status", ["pending", "all"])
    with control_right:
        refresh_clicked = st.button("Refresh queue", type="primary", width="stretch")

    if refresh_clicked or "hitl_rows" not in st.session_state:
        with st.spinner("Loading review queue..."):
            try:
                st.session_state["hitl_rows"] = load_hitl_queue(
                    None if status_filter == "all" else status_filter
                )
            except Exception as error:
                st.error(f"HITL queue request failed: {error}")
                return

    rows = st.session_state.get("hitl_rows") or []
    metric_cols = st.columns(3)
    pending_count = sum(1 for row in rows if row.get("status") == "pending")
    with metric_cols[0]:
        render_metric_card("Queue Items", format_count(len(rows)), "current filter")
    with metric_cols[1]:
        render_metric_card("Pending", format_count(pending_count), "awaiting review")
    with metric_cols[2]:
        average_confidence = (
            sum(safe_float(row.get("confidence")) for row in rows) / len(rows)
            if rows
            else 0.0
        )
        render_metric_card("Avg Confidence", f"{average_confidence:.2f}", "guardrail score")

    if not rows:
        render_empty("No queued requests found.")
        return

    queue_df = pd.DataFrame(rows)
    if "flags" in queue_df.columns:
        queue_df["flags"] = queue_df["flags"].apply(lambda values: ", ".join(values or []))
    st.dataframe(queue_df, width="stretch", hide_index=True)

    st.markdown('<div class="rp-section-title">Queued Requests</div>', unsafe_allow_html=True)
    for index, row in enumerate(rows[:25], start=1):
        flags = row.get("flags") or []
        flag_badges = "".join(badge_html(str(flag), "#d97706") for flag in flags)
        confidence_badge = badge_html(
            f"Confidence {safe_float(row.get('confidence')):.2f}"
        )
        results_badge = badge_html(
            f"Results {row.get('n_results') or 0}",
            "#2563eb",
        )
        st.markdown(
            f"""
            <div class="rp-review-card">
                <div class="rp-review-head">
                    <div>
                        <div class="rp-review-title">Request {index}: {esc(row.get("query"))}</div>
                        <div class="rp-review-meta">{esc(row.get("created_at"))}</div>
                    </div>
                    {badge_html(str(row.get("status") or "unknown"), "#d97706")}
                </div>
                <div class="rp-badge-row">
                    {confidence_badge}
                    {results_badge}
                    {flag_badges}
                </div>
                <div class="rp-review-text">{esc(row.get("reason"))}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_pipeline_ops_view() -> None:
    if st.button("Refresh pipeline status", type="primary", width="stretch"):
        load_pipeline_status.clear()
        with st.spinner("Inspecting local pipeline artifacts..."):
            try:
                st.session_state["pipeline_payload"] = load_pipeline_status()
            except Exception as error:
                st.error(f"Pipeline status request failed: {error}")
                return

    payload = st.session_state.get("pipeline_payload")
    if not payload:
        render_empty("Refresh pipeline status to inspect local data artifacts, DAG files, logs, and vector-store readiness.")
        return

    artifacts = pd.DataFrame(payload.get("artifacts") or [])
    sources = pd.DataFrame(payload.get("sources") or [])
    dags = pd.DataFrame(payload.get("dag_files") or [])
    logs = pd.DataFrame(payload.get("log_files") or [])
    vector_store = payload.get("vector_store") or {}

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Artifacts", format_count(len(artifacts)), "tracked outputs")
    with metric_cols[1]:
        render_metric_card("Sources Ready", format_count(sources["available"].sum() if not sources.empty else 0), "local source paths")
    with metric_cols[2]:
        render_metric_card("Airflow DAGs", format_count(len(dags)), "repo DAG files")
    with metric_cols[3]:
        docs = vector_store.get("sqlite_documents")
        render_metric_card("Indexed Docs", format_count(docs) if docs is not None else "Unknown", "SQLite vector store")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="rp-section-title">Source Coverage</div>', unsafe_allow_html=True)
        if not sources.empty:
            st.dataframe(sources, width="stretch", hide_index=True)
        else:
            render_empty("No source coverage rows found.")
    with right:
        st.markdown('<div class="rp-section-title">Artifacts</div>', unsafe_allow_html=True)
        if not artifacts.empty:
            st.dataframe(artifacts, width="stretch", hide_index=True)
        else:
            render_empty("No artifacts found.")

    st.markdown('<div class="rp-section-title">Airflow and Logs</div>', unsafe_allow_html=True)
    dag_col, log_col = st.columns(2)
    with dag_col:
        if not dags.empty:
            st.dataframe(dags, width="stretch", hide_index=True)
        else:
            render_empty("No DAG files found in the repository.")
    with log_col:
        if not logs.empty:
            st.dataframe(logs, width="stretch", hide_index=True)
        else:
            render_empty("No log files found yet.")

    st.info(
        "Airflow orchestration is configured through the repository DAG files and Docker assets. "
        "This page reports local artifact readiness rather than controlling Airflow runs."
    )


def render_settings_about_view() -> None:
    try:
        health = load_health()
    except Exception:
        health = {}

    metric_cols = st.columns(4)
    with metric_cols[0]:
        render_metric_card("Frontend", "Streamlit", "src/frontend/app.py")
    with metric_cols[1]:
        render_metric_card("Backend", "FastAPI", current_api_base())
    with metric_cols[2]:
        render_metric_card("Guardrails", "On" if health.get("guardrails_enabled") else "Off", "query safety")
    with metric_cols[3]:
        render_metric_card("Cache", "On" if health.get("cache_enabled") else "Off", "Redis optional")

    st.markdown('<div class="rp-section-title">Architecture Summary</div>', unsafe_allow_html=True)
    st.markdown(
        """
        ReviewPulse AI ingests reviews from Amazon, eBay, Yelp, iFixit, YouTube, and optional Reddit POC data,
        normalizes them into a shared schema, enriches reviews with sentiment and aspects, builds embeddings,
        exposes typed FastAPI endpoints, and presents the results in this Streamlit dashboard.
        """
    )

    st.markdown('<div class="rp-section-title">Supported UI Features</div>', unsafe_allow_html=True)
    feature_rows = [
        {"feature": "Review Explorer", "backend": "/reviews/explore", "status": "implemented"},
        {"feature": "Sentiment Analytics", "backend": "/analytics/sentiment", "status": "implemented"},
        {"feature": "Product Comparison", "backend": "/products/compare", "status": "implemented"},
        {"feature": "RAG Chatbot", "backend": "/chat and /search/semantic", "status": "implemented"},
        {"feature": "Aspect Intelligence", "backend": "/aspects/summary", "status": "implemented"},
        {"feature": "HITL Queue", "backend": "/hitl/queue", "status": "read-only queue"},
        {"feature": "Pipeline Ops", "backend": "/pipeline/status", "status": "local artifact status"},
        {"feature": "Token/Cost Tracking", "backend": "not present", "status": "not implemented in repo"},
    ]
    st.dataframe(pd.DataFrame(feature_rows), width="stretch", hide_index=True)

    st.markdown('<div class="rp-section-title">Run Commands</div>', unsafe_allow_html=True)
    st.code(
        "poetry run uvicorn src.api.main:app --host 127.0.0.1 --port 8001\n"
        "poetry run streamlit run src/frontend/app.py --server.port 8501",
        language="powershell",
    )

    st.markdown('<div class="rp-section-title">Project Attribution</div>', unsafe_allow_html=True)
    st.write(
        "DAMG 7245 - Big Data and Intelligent Analytics, Northeastern University. "
        "See README.md for team roles and project references."
    )


inject_theme()
view_name = render_sidebar()
render_topbar(view_name)

if view_name == "Dashboard Home":
    render_dashboard_home()
elif view_name == "Review Explorer":
    render_review_explorer()
elif view_name == "Sentiment Analytics":
    render_sentiment_analytics_view()
elif view_name == "Product Comparison":
    render_product_comparison_view()
elif view_name == "RAG Chatbot":
    render_ask_view()
elif view_name == "Aspect Intelligence":
    render_aspect_intelligence_view()
elif view_name == "HITL Queue":
    render_hitl_view()
elif view_name == "Pipeline & Data Ops":
    render_pipeline_ops_view()
else:
    render_settings_about_view()
