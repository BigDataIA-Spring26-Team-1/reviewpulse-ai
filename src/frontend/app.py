"""
ReviewPulse AI Streamlit dashboard.
 
Run:
    poetry run streamlit run src/frontend/app.py
"""
 
from __future__ import annotations
 
import pandas as pd
import requests
import streamlit as st
 
 
API_BASE = "http://127.0.0.1:8000"
 
st.set_page_config(page_title="ReviewPulse AI", layout="wide")
st.title("ReviewPulse AI")
st.subheader("Cross-Platform Product Review Intelligence")
 
 
def load_health() -> dict:
    return requests.get(f"{API_BASE}/health", timeout=10).json()
 
 
def load_stats() -> dict:
    return requests.get(f"{API_BASE}/stats/sources", timeout=20).json()
 
 
def load_data_profile() -> dict:
    return requests.get(f"{API_BASE}/data/profile", timeout=60).json()
 
 
def load_data_compare() -> dict:
    return requests.get(f"{API_BASE}/data/compare", timeout=60).json()
 
 
def load_data_quality() -> dict:
    return requests.get(f"{API_BASE}/data/quality", timeout=60).json()
 
 
def load_normalization_explanations(source: str | None = None) -> dict:
    params = {}
    if source:
        params["source"] = source
    return requests.get(f"{API_BASE}/data/normalize/explain", params=params, timeout=30).json()
 
 
def run_search(query: str, source_filter: str, n_results: int) -> list[dict]:
    params = {"query": query, "n_results": n_results}
    if source_filter:
        params["source_filter"] = source_filter
    return requests.get(f"{API_BASE}/search/semantic", params=params, timeout=60).json()
 
 
def run_chat(query: str, source_filter: str, n_results: int) -> dict:
    params = {"query": query, "n_results": n_results}
    if source_filter:
        params["source_filter"] = source_filter
    return requests.get(f"{API_BASE}/chat", params=params, timeout=60).json()
 
 
def pretty_sentiment_label(label: str) -> str:
    if not label:
        return "Unknown"
    return str(label).replace("_", " ").title()
 
 
def api_error(payload: dict) -> str | None:
    if isinstance(payload, dict):
        error = payload.get("error")
        if error:
            return str(error)
    return None
 
 
def render_search_result(item: dict, heading: str, link_label: str) -> None:
    with st.container():
        st.markdown(heading)
        st.write(f"Source: {item.get('source', '')}")
        st.write(f"Name: {item.get('display_name', '')}")
        st.write(f"Category: {item.get('display_category', '')}")
        st.write(f"Type: {item.get('entity_type', '')}")
        st.write(
            f"Sentiment: {pretty_sentiment_label(item.get('sentiment_label', ''))}"
        )
 
        if "distance" in item:
            st.write(f"Distance: {item.get('distance', 0):.4f}")
 
        url = item.get("source_url", "")
        source = item.get("source", "")
 
        if url:
            if source == "amazon":
                st.caption(
                    "Amazon link may be a sample placeholder and may not resolve to a live page."
                )
                st.write(url)
            else:
                st.markdown(f"[{link_label}]({url})")
 
        review_text = item.get("review_text")
        if review_text:
            st.info(review_text)
 
        st.markdown("---")
 
 
def render_ask_tab() -> None:
    st.markdown("## Ask About the Reviews")
 
    query = st.text_input(
        "Ask a grounded question",
        value="What do reviews say about customer service?",
    )
 
    top_col1, top_col2 = st.columns(2)
    with top_col1:
        source_filter = st.selectbox(
            "Source filter",
            ["", "amazon", "yelp", "ebay", "ifixit", "youtube", "reddit"],
        )
    with top_col2:
        n_results = st.slider(
            "Context size",
            min_value=1,
            max_value=10,
            value=5,
        )
 
    if st.button("Ask"):
        try:
            chat_response = run_chat(query, source_filter, n_results)
            search_results = run_search(query, source_filter, n_results)
        except Exception as error:
            st.error(f"Request failed: {error}")
            st.stop()
 
        answer = chat_response.get("answer", "")
        citations = chat_response.get("citations", [])
 
        st.markdown("### Answer")
        st.success(answer)
 
        st.markdown("### Citations")
        if not citations:
            st.info("No citations returned.")
        else:
            for index, item in enumerate(citations, start=1):
                render_search_result(
                    item,
                    heading=f"#### Citation {index}",
                    link_label="Open citation link",
                )
 
        st.markdown("### Supporting Retrieved Reviews")
        if not search_results:
            st.warning("No supporting search results found.")
        else:
            for index, item in enumerate(search_results, start=1):
                render_search_result(
                    item,
                    heading=f"#### Result {index}",
                    link_label="Open source link",
                )
 
    st.markdown("## API Status and Overview")
    try:
        health = load_health()
        st.success(f"API status: {health.get('status', 'unknown')}")
    except Exception as error:
        st.error(f"API not reachable: {error}")
        st.stop()
 
    stats = load_stats()
    stats_error = api_error(stats)
    if stats_error:
        st.info(stats_error)
        return
 
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
 
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Total Reviews", f"{total_reviews:,}")
    metric_2.metric("Sources", total_sources)
    metric_3.metric("Positive Reviews", f"{positive_reviews:,}")
    metric_4.metric("Negative Reviews", f"{negative_reviews:,}")
 
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
            df_sentiment_display["sentiment_label"] = df_sentiment_display[
                "sentiment_label"
            ].apply(pretty_sentiment_label)
            st.dataframe(df_sentiment_display, use_container_width=True)
            pivot = df_sentiment.pivot(
                index="source",
                columns="sentiment_label",
                values="count",
            ).fillna(0)
            st.bar_chart(pivot)
        else:
            st.info("No sentiment data available.")
 
 
def render_normalization_explanation(explanation: dict) -> None:
    st.markdown(f"#### {str(explanation.get('source', 'unknown')).title()}")
    st.write(f"Raw rating field: {explanation.get('raw_rating_field')}")
    st.write(f"Raw date field: {explanation.get('raw_date_field')}")
    st.write(f"Raw scale: {explanation.get('raw_scale')}")
    st.write(f"Formula: {explanation.get('formula')}")
    st.write(explanation.get("rating_notes", ""))
    st.caption(explanation.get("date_notes", ""))
 
    if explanation.get("sample_found"):
        identifiers = explanation.get("sample_identifiers") or {}
        if identifiers:
            st.json(identifiers)
 
        st.write(
            f"Sample raw rating value: {explanation.get('sample_raw_rating_value')}"
        )
        st.write(
            f"Sample normalized rating: {explanation.get('sample_normalized_rating')}"
        )
        st.write(
            "Sample raw date value: "
            f"{explanation.get('sample_raw_date_value')}"
        )
        st.write(
            "Sample normalized review_date: "
            f"{explanation.get('sample_normalized_review_date')}"
        )
        sample_path = explanation.get("sample_source_path")
        if sample_path:
            st.caption(f"Sample source file: {sample_path}")
    else:
        st.info("No local raw sample was found for this source. Showing the formula only.")
 
    st.markdown("---")
 
 
def render_data_insights_tab() -> None:
    st.markdown("## Data Insights")
 
    try:
        profile = load_data_profile()
        comparison = load_data_compare()
        quality = load_data_quality()
    except Exception as error:
        st.error(f"Data insights request failed: {error}")
        return
 
    profile_error = api_error(profile)
    comparison_error = api_error(comparison)
    quality_error = api_error(quality)
 
    if profile_error:
        st.info(profile_error)
        return
    if comparison_error:
        st.info(comparison_error)
        return
    if quality_error:
        st.info(quality_error)
        return
 
    dataset_path = profile.get("dataset_path")
    if dataset_path:
        st.caption(f"Using dataset: {dataset_path}")
 
    metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)
    metric_1.metric("Records", f"{int(profile.get('record_count', 0)):,}")
    metric_2.metric("Sources", int(profile.get("source_count", 0)))
    metric_3.metric("Distinct Products", f"{int(profile.get('distinct_products', 0)):,}")
    metric_4.metric("Distinct Reviewers", f"{int(profile.get('distinct_reviewers', 0)):,}")
    metric_5.metric(
        "Avg Review Length",
        f"{float(profile.get('average_review_length_words', 0.0)):.2f} words",
    )
 
    rating_summary = profile.get("rating_normalized", {})
    date_range = profile.get("date_range", {})
    summary_left, summary_right = st.columns(2)
    with summary_left:
        st.markdown("### Dataset Profile")
        st.write(f"Min normalized rating: {rating_summary.get('min')}")
        st.write(f"Mean normalized rating: {rating_summary.get('mean')}")
        st.write(f"Max normalized rating: {rating_summary.get('max')}")
        st.write(f"Date range start: {date_range.get('min')}")
        st.write(f"Date range end: {date_range.get('max')}")
 
    with summary_right:
        st.markdown("### Null Counts")
        null_counts = pd.DataFrame(
            [
                {"field": field_name, "null_count": null_count}
                for field_name, null_count in (profile.get("null_counts") or {}).items()
            ]
        )
        if not null_counts.empty:
            st.dataframe(null_counts, use_container_width=True)
        else:
            st.info("No null count data available.")
 
    st.markdown("### Source Comparison")
    comparison_rows = comparison.get("sources", [])
    df_compare = pd.DataFrame(comparison_rows) if comparison_rows else pd.DataFrame()
    if not df_compare.empty:
        st.dataframe(df_compare, use_container_width=True)
 
        comparison_left, comparison_middle, comparison_right = st.columns(3)
        with comparison_left:
            st.caption("Average review length by source")
            st.bar_chart(df_compare.set_index("source")[["average_review_length_words"]])
        with comparison_middle:
            rating_compare = df_compare[["source", "average_rating_normalized"]].fillna(0.0)
            st.caption("Average normalized rating by source")
            st.bar_chart(rating_compare.set_index("source"))
        with comparison_right:
            missing_compare = df_compare[["source", "missing_value_ratio"]].fillna(0.0)
            st.caption("Missing value ratio by source")
            st.bar_chart(missing_compare.set_index("source"))
    else:
        st.info("No source comparison data available.")
 
    st.markdown("### Data Quality")
    quality_metric_1, quality_metric_2, quality_metric_3, quality_metric_4 = st.columns(4)
    quality_metric_1.metric("Duplicate Ratio", f"{quality.get('duplicate_ratio', 0.0):.2%}")
    quality_metric_2.metric("Null Ratio", f"{quality.get('null_ratio', 0.0):.2%}")
    quality_metric_3.metric("Empty Text Ratio", f"{quality.get('empty_text_ratio', 0.0):.2%}")
    quality_metric_4.metric("Missing Rating Ratio", f"{quality.get('missing_rating_ratio', 0.0):.2%}")
    st.caption(quality.get("duplicate_method", ""))
 
    quality_rows = quality.get("per_source", [])
    df_quality = pd.DataFrame(quality_rows) if quality_rows else pd.DataFrame()
    if not df_quality.empty:
        st.dataframe(df_quality, use_container_width=True)
        quality_chart = df_quality[["source", "duplicate_ratio", "empty_text_ratio", "missing_rating_ratio"]]
        st.bar_chart(quality_chart.set_index("source"))
    else:
        st.info("No per-source quality summary available.")
 
    st.markdown("### Normalization Explainer")
    normalization_source = st.selectbox(
        "Normalization source",
        ["all", "amazon", "yelp", "ebay", "ifixit", "youtube"],
    )
 
    try:
        explanations_payload = load_normalization_explanations(
            None if normalization_source == "all" else normalization_source
        )
    except Exception as error:
        st.error(f"Normalization explanation request failed: {error}")
        return
 
    explanation_error = api_error(explanations_payload)
    if explanation_error:
        st.info(explanation_error)
        return
 
    explanations = explanations_payload.get("explanations", [])
    if not explanations:
        st.info("No normalization explanations available.")
    else:
        for explanation in explanations:
            render_normalization_explanation(explanation)
 
 
ask_tab, insights_tab = st.tabs(["Ask Reviews", "Data Insights"])
 
with ask_tab:
    render_ask_tab()
 
with insights_tab:
    render_data_insights_tab()
 
 