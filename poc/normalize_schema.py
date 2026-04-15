"""
ReviewPulse AI — POC: Schema Normalization Pipeline
====================================================
Takes raw data from Amazon, Yelp, Reddit, and YouTube
and normalizes everything into one unified schema.

This is the CORE data engineering work. Every source has a different
format, different field names, different rating scales, different
date formats. This script demonstrates the normalization logic
that would run as a Spark job in production.

Run: python poc/normalize_schema.py
Output: data/normalized_reviews.jsonl + console report
"""

import json
import os
from datetime import datetime, UTC
from typing import Optional

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


# ══════════════════════════════════════════════════════════════
# UNIFIED SCHEMA — every review from every source maps to this
# ══════════════════════════════════════════════════════════════

def create_unified_record(
    review_id: str,
    product_name: str,
    product_category: str,
    source: str,
    rating_normalized: Optional[float],
    review_text: str,
    review_date: Optional[str],
    reviewer_id: str,
    verified_purchase: Optional[bool],
    helpful_votes: Optional[int],
    source_url: str,
    display_name: Optional[str] = None,
    display_category: Optional[str] = None,
    entity_type: Optional[str] = None,
) -> dict:
    if display_name is None:
        display_name = product_name
    if display_category is None:
        display_category = product_category
    if entity_type is None:
        entity_type = f"{source}_entry"

    return {
        "review_id": review_id,
        "product_name": product_name,
        "product_category": product_category,
        "source": source,
        "rating_normalized": rating_normalized,
        "review_text": review_text,
        "review_date": review_date,
        "reviewer_id": reviewer_id,
        "verified_purchase": verified_purchase,
        "helpful_votes": helpful_votes,
        "source_url": source_url,
        "display_name": display_name,
        "display_category": display_category,
        "entity_type": entity_type,
        "text_length_words": len(review_text.split()) if review_text else 0,
    }


# ══════════════════════════════════════════════════════════════
# SHARED LOADERS
# ══════════════════════════════════════════════════════════════

def load_jsonl(path: str, limit: Optional[int] = None) -> list[dict]:
    """
    Load newline-delimited JSON records from a file.
    Skips malformed lines instead of crashing.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_yelp_business_lookup(path: str, limit: Optional[int] = None) -> dict:
    """
    Build a lookup: business_id -> business metadata
    """
    business_lookup = {}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                business_id = row.get("business_id")
                if business_id:
                    business_lookup[business_id] = row
            except json.JSONDecodeError:
                continue
    return business_lookup


# ══════════════════════════════════════════════════════════════
# SOURCE-SPECIFIC NORMALIZERS
# ══════════════════════════════════════════════════════════════

def normalize_amazon(raw: dict) -> dict:
    """
    Amazon Reviews 2023 normalization.

    Key decisions:
    - rating is float 1.0-5.0 → normalize to 0-1: (rating - 1) / 4
    - timestamp is epoch MILLISECONDS → convert to ISO 8601
    - title and text may be null → use empty string
    - asin is product ID → use as product_name
    - add user-facing metadata for cleaner display
    """
    rating = raw.get("rating")
    rating_norm = (rating - 1.0) / 4.0 if rating is not None else None

    ts = raw.get("timestamp")
    date_str = None
    if ts:
        try:
            date_str = datetime.fromtimestamp(ts / 1000, UTC).isoformat()
        except (ValueError, OSError, TypeError):
            date_str = None

    text = raw.get("text") or ""
    title = raw.get("title") or ""
    full_text = f"{title}. {text}".strip(". ") if title else text

    asin = raw.get("asin", "unknown")
    parent_asin = raw.get("parent_asin", asin)

    return create_unified_record(
        review_id=f"amazon_{asin}_{raw.get('user_id', '')[:8]}",
        product_name=parent_asin,
        product_category="electronics",
        source="amazon",
        rating_normalized=round(rating_norm, 4) if rating_norm is not None else None,
        review_text=full_text,
        review_date=date_str,
        reviewer_id=raw.get("user_id", "unknown"),
        verified_purchase=raw.get("verified_purchase"),
        helpful_votes=raw.get("helpful_vote", 0),
        source_url=f"https://amazon.com/dp/{asin}",
        display_name=f"Amazon Electronics Item {asin}",
        display_category="Electronics Product",
        entity_type="product_review",
    )


def normalize_yelp(raw: dict, business_lookup: Optional[dict] = None) -> dict:
    """
    Yelp Open Dataset normalization.

    Key decisions:
    - stars is INTEGER 1-5 -> normalize to 0-1
    - date is string YYYY-MM-DD -> ISO 8601 style
    - no verified_purchase -> None
    - no helpful_votes -> None
    - use business metadata if available
    - add user-friendly metadata fields for UI and chat
    """
    stars = raw.get("stars")
    rating_norm = (stars - 1) / 4.0 if stars is not None else None

    date_str = raw.get("date")
    if date_str:
        date_str = date_str + "T00:00:00"

    business = {}
    if business_lookup:
        business = business_lookup.get(raw.get("business_id"), {})

    business_name = business.get("name", raw.get("business_id", "unknown"))

    categories = business.get("categories")
    if isinstance(categories, str) and categories.strip():
        product_category = categories
        display_category = categories.split(",")[0].strip()
    else:
        product_category = "local_business"
        display_category = "Local Business"

    return create_unified_record(
        review_id=f"yelp_{raw.get('review_id', '')}",
        product_name=business_name,
        product_category=product_category,
        source="yelp",
        rating_normalized=round(rating_norm, 4) if rating_norm is not None else None,
        review_text=raw.get("text", ""),
        review_date=date_str,
        reviewer_id=raw.get("user_id", "unknown"),
        verified_purchase=None,
        helpful_votes=None,
        source_url=f"https://yelp.com/biz/{raw.get('business_id', '')}",
        display_name=business_name,
        display_category=display_category,
        entity_type="business_review",
    )


def normalize_reddit(raw: dict) -> dict:
    """
    Reddit normalization.

    Key decisions:
    - Reddit has NO star rating → rating_normalized = None
    - We store the post score separately but do NOT convert it to a rating
    - created_utc is epoch SECONDS → convert to ISO 8601
    - no verified purchase → None
    - add user-friendly metadata fields for UI and chat
    """
    ts = raw.get("created_utc")
    date_str = None
    if ts:
        try:
            date_str = datetime.fromtimestamp(ts, UTC).isoformat()
        except (ValueError, OSError, TypeError):
            date_str = None

    title = raw.get("title", "")
    text = raw.get("text", "")
    full_text = f"{title}. {text}".strip(". ") if title else text

    subreddit = raw.get("subreddit", "general")
    title_text = raw.get("title", "Reddit Discussion")

    return create_unified_record(
        review_id=f"reddit_{raw.get('source_id', '')}",
        product_name="unknown",
        product_category=subreddit,
        source="reddit",
        rating_normalized=None,
        review_text=full_text,
        review_date=date_str,
        reviewer_id=raw.get("author", "unknown"),
        verified_purchase=None,
        helpful_votes=raw.get("score", 0),
        source_url=raw.get("url", ""),
        display_name=title_text,
        display_category=subreddit,
        entity_type="forum_post",
    )


def normalize_youtube(raw: dict) -> dict:
    """
    YouTube transcript normalization.

    Key decisions:
    - YouTube has NO rating → rating_normalized = None
    - transcript is unstructured speech
    - no verified purchase → None
    - add user-friendly metadata fields for UI and chat
    """
    ts = raw.get("created_utc")
    date_str = None
    if ts:
        try:
            date_str = datetime.fromtimestamp(ts, UTC).isoformat()
        except (ValueError, OSError, TypeError):
            date_str = None

    title_text = raw.get("title", "YouTube Review")

    return create_unified_record(
        review_id=f"youtube_{raw.get('source_id', '')}",
        product_name="unknown",
        product_category="unknown",
        source="youtube",
        rating_normalized=None,
        review_text=raw.get("text", ""),
        review_date=date_str,
        reviewer_id=raw.get("channel", "unknown"),
        verified_purchase=None,
        helpful_votes=None,
        source_url=raw.get("url", ""),
        display_name=title_text,
        display_category="Video Review",
        entity_type="video_transcript",
    )

# ══════════════════════════════════════════════════════════════
# SAMPLE FALLBACKS
# ══════════════════════════════════════════════════════════════

def generate_yelp_sample():
    """
    Yelp sample fallback if real files are missing.
    """
    import random
    random.seed(123)
    records = []
    for i in range(100):
        records.append({
            "review_id": f"yelp_review_{i:04d}",
            "user_id": f"yelp_user_{random.randint(1000, 9999)}",
            "business_id": f"biz_{random.randint(100, 999)}",
            "stars": random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.08, 0.15, 0.30, 0.42])[0],
            "text": (
                f"This place was {'great' if random.random() > 0.3 else 'disappointing'}. "
                f"The service was {'quick' if random.random() > 0.4 else 'slow'}. "
                f"Food quality was {'excellent' if random.random() > 0.3 else 'average'}. "
                f"Would {'definitely' if random.random() > 0.3 else 'probably not'} come back."
            ),
            "date": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        })
    return records


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

def run():
    print("=" * 60)
    print("REVIEWPULSE AI — SCHEMA NORMALIZATION PIPELINE")
    print("=" * 60)

    all_normalized = []
    source_counts = {}

    # 1. Normalize Amazon data
    amazon_path = os.path.join(OUTPUT_DIR, "amazon_electronics_sample.jsonl")
    if os.path.exists(amazon_path):
        print(f"\n  [1/4] Normalizing Amazon data from {amazon_path}...")
        count = 0
        with open(amazon_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                normalized = normalize_amazon(raw)
                all_normalized.append(normalized)
                count += 1
        source_counts["amazon"] = count
        print(f"    Normalized {count} Amazon reviews")
    else:
        print(f"\n  [1/4] Amazon sample not found at {amazon_path}")
        print("    Run eda_amazon.py first to generate the sample.")
        print("    Generating inline sample...")
        import random
        random.seed(42)
        for i in range(500):
            raw = {
                "rating": float(random.choices([1, 2, 3, 4, 5], weights=[0.08, 0.05, 0.09, 0.20, 0.58])[0]),
                "title": f"Product review {i}",
                "text": f"This product is {'great' if random.random() > 0.3 else 'okay'}. Quality is {'excellent' if random.random() > 0.4 else 'average'}.",
                "asin": f"B0{random.randint(10000, 99999)}",
                "parent_asin": f"B0{random.randint(10000, 99999)}",
                "user_id": f"USER_{random.randint(10000, 99999)}",
                "timestamp": int(datetime(2023, random.randint(1, 12), random.randint(1, 28)).timestamp() * 1000),
                "helpful_vote": random.choice([0, 0, 0, 1, 2, 5]),
                "verified_purchase": random.random() > 0.15,
            }
            all_normalized.append(normalize_amazon(raw))
        source_counts["amazon"] = 500

    # 2. Normalize Yelp data
    print(f"\n  [2/4] Normalizing Yelp data...")

    yelp_review_path = os.path.join(OUTPUT_DIR, "yelp", "yelp_academic_dataset_review.json")
    yelp_business_path = os.path.join(OUTPUT_DIR, "yelp", "yelp_academic_dataset_business.json")

    if os.path.exists(yelp_review_path):
        print("    Found real Yelp review file")

        business_lookup = {}
        if os.path.exists(yelp_business_path):
            print("    Found Yelp business file")
            business_lookup = load_yelp_business_lookup(yelp_business_path)

        yelp_raw = load_jsonl(yelp_review_path, limit=10000)
        for raw in yelp_raw:
            all_normalized.append(normalize_yelp(raw, business_lookup))

        source_counts["yelp"] = len(yelp_raw)
        print(f"    Normalized {len(yelp_raw)} real Yelp reviews")
    else:
        print("    Real Yelp files not found, falling back to generated sample")
        yelp_raw = generate_yelp_sample()
        for raw in yelp_raw:
            all_normalized.append(normalize_yelp(raw))
        source_counts["yelp"] = len(yelp_raw)
        print(f"    Normalized {len(yelp_raw)} synthetic Yelp reviews")

    # 3. Normalize Reddit data
    reddit_path = os.path.join(OUTPUT_DIR, "reddit_reviews.jsonl")
    if os.path.exists(reddit_path):
        print(f"\n  [3/4] Normalizing Reddit data from {reddit_path}...")
        count = 0
        with open(reddit_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                all_normalized.append(normalize_reddit(raw))
                count += 1
        source_counts["reddit"] = count
        print(f"    Normalized {count} Reddit posts")
    else:
        print(f"\n  [3/4] Reddit data not found. Run reddit_connector.py first.")
        source_counts["reddit"] = 0

    # 4. Normalize YouTube data
    youtube_path = os.path.join(OUTPUT_DIR, "youtube_reviews.jsonl")
    if os.path.exists(youtube_path):
        print(f"\n  [4/4] Normalizing YouTube data from {youtube_path}...")
        count = 0
        with open(youtube_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    continue
                all_normalized.append(normalize_youtube(raw))
                count += 1
        source_counts["youtube"] = count
        print(f"    Normalized {count} YouTube transcripts")
    else:
        print(f"\n  [4/4] YouTube data not found. Run youtube_extractor.py first.")
        source_counts["youtube"] = 0

    # Save normalized output
    output_path = os.path.join(OUTPUT_DIR, "normalized_reviews.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in all_normalized:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # REPORT
    print(f"\n{'=' * 60}")
    print("NORMALIZATION REPORT")
    print(f"{'=' * 60}")
    print(f"Total normalized records: {len(all_normalized)}")

    print("\nRecords by source:")
    for src, cnt in source_counts.items():
        print(f"  {src}: {cnt}")

    if all_normalized:
        print(f"\nUnified schema fields: {list(all_normalized[0].keys())}")

    print("\nRating analysis by source:")
    for src in source_counts:
        src_records = [r for r in all_normalized if r["source"] == src]
        rated = [r for r in src_records if r["rating_normalized"] is not None]
        null_rated = len(src_records) - len(rated)
        if rated:
            avg = sum(r["rating_normalized"] for r in rated) / len(rated)
            print(f"  {src}: {len(rated)} rated (avg={avg:.3f}), {null_rated} NULL ratings")
        else:
            print(f"  {src}: ALL {null_rated} reviews have NULL rating (no star system)")

    print("\nNull field analysis:")
    for field in ["rating_normalized", "verified_purchase", "helpful_votes", "review_date"]:
        nulls = sum(1 for r in all_normalized if r.get(field) is None)
        pct = (nulls / len(all_normalized) * 100) if all_normalized else 0
        print(f"  {field}: {nulls} nulls ({pct:.1f}%)")

    print("\nText length by source:")
    for src in source_counts:
        src_records = [r for r in all_normalized if r["source"] == src]
        if src_records:
            avg_words = sum(r["text_length_words"] for r in src_records) / len(src_records)
            print(f"  {src}: avg {avg_words:.0f} words")

    print(f"\nOutput saved to: {output_path}")
    print("\n✓ Schema normalization complete.")
    print("\nKEY INSIGHT: Reddit and YouTube have NULL ratings because those")
    print("platforms have no star rating system. This is a real schema")
    print("challenge that our pipeline handles explicitly — we do NOT")
    print("fake a rating for these sources.")


if __name__ == "__main__":
    run()
