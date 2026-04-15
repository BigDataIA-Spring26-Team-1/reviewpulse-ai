"""
ReviewPulse AI — Single Page Dashboard
======================================
Top: unified question flow
Bottom: API status + overview metrics + charts

Run:
    poetry run streamlit run src/frontend/app.py
"""

import streamlit as st
import requests
import pandas as pd

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="ReviewPulse AI", layout="wide")
st.title("ReviewPulse AI")
st.subheader("Cross-Platform Product Review Intelligence")


def load_health():
    return requests.get(f"{API_BASE}/health", timeout=10).json()


def load_stats():
    return requests.get(f"{API_BASE}/stats/sources", timeout=20).json()


def run_search(query, source_filter, n_results):
    params = {
        "query": query,
        "n_results": n_results,
    }
    if source_filter:
        params["source_filter"] = source_filter
    return requests.get(f"{API_BASE}/search/semantic", params=params, timeout=60).json()


def run_chat(query, source_filter, n_results):
    params = {
        "query": query,
        "n_results": n_results,
    }
    if source_filter:
        params["source_filter"] = source_filter
    return requests.get(f"{API_BASE}/chat", params=params, timeout=60).json()


def pretty_sentiment_label(label: str) -> str:
    if not label:
        return "Unknown"
    return str(label).replace("_", " ").title()


# =========================
# TOP: UNIFIED QUESTION FLOW
# =========================
st.markdown("## Ask About the Reviews")

query = st.text_input(
    "Ask a grounded question",
    value="What do reviews say about customer service?"
)

top_col1, top_col2 = st.columns(2)

with top_col1:
    source_filter = st.selectbox(
        "Source filter",
        ["", "amazon", "yelp", "reddit", "youtube"]
    )

with top_col2:
    n_results = st.slider(
        "Context size",
        min_value=1,
        max_value=10,
        value=5
    )

if st.button("Ask"):
    try:
        chat_response = run_chat(query, source_filter, n_results)
        search_results = run_search(query, source_filter, n_results)
    except Exception as e:
        st.error(f"Request failed: {e}")
        st.stop()

    answer = chat_response.get("answer", "")
    citations = chat_response.get("citations", [])

    st.markdown("### Answer")
    st.success(answer)

    st.markdown("### Citations")
    if not citations:
        st.info("No citations returned.")
    else:
        for i, item in enumerate(citations, start=1):
            with st.container():
                st.markdown(f"#### Citation {i}")
                st.write(f"Source: {item.get('source', '')}")
                st.write(f"Name: {item.get('display_name', '')}")
                st.write(f"Category: {item.get('display_category', '')}")
                st.write(f"Type: {item.get('entity_type', '')}")
                st.write(f"Sentiment: {pretty_sentiment_label(item.get('sentiment_label', ''))}")

                url = item.get("source_url", "")
                source = item.get("source", "")

                if url:
                    if source == "amazon":
                        st.caption("Amazon link may be a sample placeholder and may not resolve to a live page.")
                        st.write(url)
                    else:
                        st.markdown(f"[Open citation link]({url})")

                st.markdown("---")

    st.markdown("### Supporting Retrieved Reviews")
    if not search_results:
        st.warning("No supporting search results found.")
    else:
        for i, item in enumerate(search_results, start=1):
            with st.container():
                st.markdown(f"#### Result {i}")
                st.write(f"Source: {item.get('source', '')}")
                st.write(f"Name: {item.get('display_name', '')}")
                st.write(f"Category: {item.get('display_category', '')}")
                st.write(f"Type: {item.get('entity_type', '')}")
                st.write(f"Sentiment: {pretty_sentiment_label(item.get('sentiment_label', ''))}")
                st.write(f"Distance: {item.get('distance', 0):.4f}")

                url = item.get("source_url", "")
                source = item.get("source", "")

                if url:
                    if source == "amazon":
                        st.caption("Amazon link may be a sample placeholder and may not resolve to a live page.")
                        st.write(url)
                    else:
                        st.markdown(f"[Open source link]({url})")

                st.info(item.get("review_text", ""))
                st.markdown("---")


# =========================
# BOTTOM: API + OVERVIEW
# =========================
st.markdown("## API Status and Overview")

try:
    health = load_health()
    st.success(f"API status: {health.get('status', 'unknown')}")
except Exception as e:
    st.error(f"API not reachable: {e}")
    st.stop()

stats = load_stats()
source_counts = stats.get("source_counts", [])
sentiment_breakdown = stats.get("sentiment_breakdown", [])

df_counts = pd.DataFrame(source_counts) if source_counts else pd.DataFrame()
df_sentiment = pd.DataFrame(sentiment_breakdown) if sentiment_breakdown else pd.DataFrame()

if not df_counts.empty:
    total_reviews = int(df_counts["count"].sum())
    total_sources = int(df_counts["source"].nunique())
else:
    total_reviews = 0
    total_sources = 0

positive_reviews = 0
negative_reviews = 0

if not df_sentiment.empty:
    for _, row in df_sentiment.iterrows():
        if row["sentiment_label"] == "positive":
            positive_reviews += int(row["count"])
        elif row["sentiment_label"] == "negative":
            negative_reviews += int(row["count"])

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Reviews", f"{total_reviews:,}")
m2.metric("Sources", total_sources)
m3.metric("Positive Reviews", f"{positive_reviews:,}")
m4.metric("Negative Reviews", f"{negative_reviews:,}")

bottom_left, bottom_right = st.columns(2)

with bottom_left:
    st.markdown("### Source Counts")
    if not df_counts.empty:
        st.dataframe(df_counts, use_container_width=True)
        st.bar_chart(df_counts.set_index("source"))
    else:
        st.info("No source count data available.")

with bottom_right:
    st.markdown("### Sentiment Breakdown")
    if not df_sentiment.empty:
        df_sentiment_display = df_sentiment.copy()
        df_sentiment_display["sentiment_label"] = df_sentiment_display["sentiment_label"].apply(pretty_sentiment_label)
        st.dataframe(df_sentiment_display, use_container_width=True)
        pivot = df_sentiment.pivot(index="source", columns="sentiment_label", values="count").fillna(0)
        st.bar_chart(pivot)
    else:
        st.info("No sentiment data available.")