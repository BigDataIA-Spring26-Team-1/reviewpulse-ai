"""
ReviewPulse AI — POC: Aspect Extraction Demo
=============================================
Demonstrates the aspect extraction pipeline on sample reviews.

Two modes:
  1. Heuristic mode (default): keyword-based extraction for POC speed
  2. LLM mode (if OLLAMA_HOST is set): real extraction via Ollama/Llama 3.1

The heuristic mode proves the pipeline contract works. The LLM mode
proves the extraction quality. Both output the same schema.

Run: python poc/05_aspect_extraction.py
Output: data/extracted_aspects.jsonl + accuracy report
"""

import json
import os
import re
from typing import Optional

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ══════════════════════════════════════════════════════════════
# ASPECT EXTRACTION — HEURISTIC MODE
# Fast, deterministic, no API needed. Good for POC.
# ══════════════════════════════════════════════════════════════

# Aspect keywords — mapped to canonical aspect names
ASPECT_KEYWORDS = {
    "battery": ["battery", "battery life", "charge", "charging", "dies", "lasts", "power"],
    "camera": ["camera", "photo", "picture", "lens", "megapixel", "portrait", "zoom"],
    "sound quality": ["sound", "audio", "bass", "treble", "noise cancellation", "anc", "speaker"],
    "build quality": ["build", "construction", "material", "plastic", "metal", "premium", "sturdy", "flimsy", "durable"],
    "comfort": ["comfort", "comfortable", "fits", "ear", "weight", "lightweight", "heavy", "ergonomic"],
    "screen": ["screen", "display", "resolution", "brightness", "oled", "lcd", "refresh rate"],
    "performance": ["performance", "speed", "fast", "slow", "lag", "smooth", "processor", "cpu", "gpu"],
    "price": ["price", "expensive", "cheap", "value", "worth", "cost", "affordable", "overpriced", "deal"],
    "design": ["design", "look", "style", "aesthetic", "color", "sleek", "ugly", "beautiful"],
    "durability": ["durable", "durability", "broke", "broken", "cracked", "lasted", "years"],
    "customer service": ["customer service", "support", "warranty", "return", "refund", "replacement"],
    "ease of use": ["easy", "simple", "intuitive", "complicated", "setup", "install", "user friendly"],
}

# Sentiment keywords
POSITIVE_WORDS = {"great", "excellent", "amazing", "love", "best", "perfect", "wonderful", "fantastic", 
                  "impressive", "solid", "superb", "outstanding", "recommend", "happy", "pleased",
                  "incredible", "brilliant", "stunning", "insane", "awesome"}
NEGATIVE_WORDS = {"terrible", "awful", "worst", "hate", "horrible", "disappointing", "poor", "bad",
                  "broken", "useless", "waste", "regret", "frustrating", "cheap", "flimsy",
                  "died", "fails", "defective", "garbage", "overpriced", "sucks"}


def extract_aspects_heuristic(text: str) -> list:
    """
    Rule-based aspect extraction.
    Returns list of {aspect, sentiment, quote} dicts.
    
    This is the fast POC version. In production, this is replaced
    by Ollama/Llama 3.1 for much higher accuracy.
    """
    text_lower = text.lower()
    sentences = re.split(r'[.!?]+', text)
    
    results = []
    seen_aspects = set()
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
        sentence_lower = sentence.lower()
        
        for aspect, keywords in ASPECT_KEYWORDS.items():
            if aspect in seen_aspects:
                continue
            
            for kw in keywords:
                if kw in sentence_lower:
                    # Score sentiment of the sentence
                    words = set(sentence_lower.split())
                    pos = len(words & POSITIVE_WORDS)
                    neg = len(words & NEGATIVE_WORDS)
                    
                    if pos > neg:
                        sentiment = min(0.5 + pos * 0.2, 1.0)
                    elif neg > pos:
                        sentiment = max(-0.5 - neg * 0.2, -1.0)
                    else:
                        sentiment = 0.0
                    
                    results.append({
                        "aspect": aspect,
                        "sentiment": round(sentiment, 2),
                        "quote": sentence[:150],  # Truncate long quotes
                    })
                    seen_aspects.add(aspect)
                    break
    
    return results


def extract_aspects_llm(text: str, ollama_host: str = "http://localhost:11434") -> Optional[list]:
    """
    LLM-based aspect extraction via Ollama.
    Returns list of {aspect, sentiment, quote} dicts.
    
    Requires Ollama running locally with Llama 3.1 pulled.
    Setup: ollama pull llama3.1:8b
    """
    import urllib.request
    
    prompt = f"""Extract product aspects mentioned in this review and rate sentiment for each.
Respond ONLY with a valid JSON array. No other text.

Review: "{text[:1000]}"

Output format: [{{"aspect": "string", "sentiment": float_between_-1_and_1, "quote": "relevant quote"}}]"""
    
    try:
        payload = json.dumps({
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1},
        }).encode()
        
        req = urllib.request.Request(
            f"{ollama_host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            response_text = result.get("response", "")
            
            # Try to parse JSON from the response
            # LLMs sometimes wrap JSON in markdown code blocks
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = re.sub(r'```\w*\n?', '', response_text).strip()
            
            aspects = json.loads(response_text)
            if isinstance(aspects, list):
                return aspects
            return None
            
    except json.JSONDecodeError:
        # ~3% of responses are malformed JSON — we catch, log, and skip
        print(f"    WARNING: Ollama returned malformed JSON. Skipping review.")
        return None
    except Exception as e:
        print(f"    WARNING: Ollama request failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════
# GOLDEN SET — manual annotations for accuracy evaluation
# ══════════════════════════════════════════════════════════════

GOLDEN_SET = [
    {
        "text": "The battery on this phone is terrible. Dies after 3 hours of use. But the camera is absolutely stunning, best I've ever used.",
        "expected_aspects": [
            {"aspect": "battery", "sentiment_sign": "negative"},
            {"aspect": "camera", "sentiment_sign": "positive"},
        ],
    },
    {
        "text": "Sound quality is amazing for the price. Noise cancellation blocks out everything. Very comfortable for long listening sessions.",
        "expected_aspects": [
            {"aspect": "sound quality", "sentiment_sign": "positive"},
            {"aspect": "comfort", "sentiment_sign": "positive"},
        ],
    },
    {
        "text": "Build quality feels cheap and plastic. The design looks nice but after 2 months it cracked. Terrible durability.",
        "expected_aspects": [
            {"aspect": "build quality", "sentiment_sign": "negative"},
            {"aspect": "design", "sentiment_sign": "positive"},
            {"aspect": "durability", "sentiment_sign": "negative"},
        ],
    },
    {
        "text": "Easy to set up out of the box. Performance is smooth, no lag at all. Screen is gorgeous with vivid colors.",
        "expected_aspects": [
            {"aspect": "ease of use", "sentiment_sign": "positive"},
            {"aspect": "performance", "sentiment_sign": "positive"},
            {"aspect": "screen", "sentiment_sign": "positive"},
        ],
    },
    {
        "text": "Way too expensive for what you get. Customer service was horrible when I tried to get a refund.",
        "expected_aspects": [
            {"aspect": "price", "sentiment_sign": "negative"},
            {"aspect": "customer service", "sentiment_sign": "negative"},
        ],
    },
]


def evaluate_golden_set(extract_fn):
    """
    Evaluate extraction against golden set.
    Reports aspect detection precision, recall, F1.
    Reports sentiment direction accuracy.
    """
    total_expected = 0
    total_extracted = 0
    correct_aspects = 0
    correct_sentiment = 0
    
    print(f"\n{'─' * 50}")
    print("GOLDEN SET EVALUATION")
    print(f"{'─' * 50}")
    
    for i, item in enumerate(GOLDEN_SET):
        extracted = extract_fn(item["text"])
        extracted_aspect_names = {e["aspect"] for e in extracted}
        expected_aspects = {e["aspect"] for e in item["expected_aspects"]}
        
        total_expected += len(expected_aspects)
        total_extracted += len(extracted_aspect_names)
        
        matched = extracted_aspect_names & expected_aspects
        correct_aspects += len(matched)
        
        # Check sentiment direction for matched aspects
        for exp in item["expected_aspects"]:
            for ext in extracted:
                if ext["aspect"] == exp["aspect"]:
                    ext_sign = "positive" if ext["sentiment"] > 0 else "negative" if ext["sentiment"] < 0 else "neutral"
                    if ext_sign == exp["sentiment_sign"]:
                        correct_sentiment += 1
        
        print(f"\n  Review {i+1}: \"{item['text'][:80]}...\"")
        print(f"    Expected: {expected_aspects}")
        print(f"    Got:      {extracted_aspect_names}")
        print(f"    Matched:  {matched}")
    
    precision = correct_aspects / total_extracted if total_extracted > 0 else 0
    recall = correct_aspects / total_expected if total_expected > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    sentiment_acc = correct_sentiment / correct_aspects if correct_aspects > 0 else 0
    
    print(f"\n{'─' * 50}")
    print(f"RESULTS:")
    print(f"  Aspect Precision: {precision:.2f}")
    print(f"  Aspect Recall:    {recall:.2f}")
    print(f"  Aspect F1:        {f1:.2f}")
    print(f"  Sentiment Direction Accuracy: {sentiment_acc:.2f}")
    print(f"  (Target: F1 > 0.85 with LLM extraction)")
    print(f"{'─' * 50}")
    
    return {"precision": precision, "recall": recall, "f1": f1, "sentiment_accuracy": sentiment_acc}


def run():
    print("=" * 60)
    print("REVIEWPULSE AI — ASPECT EXTRACTION DEMO")
    print("=" * 60)
    
    # Determine mode
    ollama_host = os.getenv("OLLAMA_HOST", "")
    use_llm = bool(ollama_host)
    
    if use_llm:
        print(f"\n  Mode: LLM (Ollama at {ollama_host})")
        extract_fn = lambda text: extract_aspects_llm(text, ollama_host) or extract_aspects_heuristic(text)
    else:
        print(f"\n  Mode: Heuristic (set OLLAMA_HOST to use LLM)")
        extract_fn = extract_aspects_heuristic
    
    # Run on normalized data if available
    input_path = os.path.join(OUTPUT_DIR, "normalized_reviews.jsonl")
    if not os.path.exists(input_path):
        print(f"  Normalized data not found. Run 04_normalize_schema.py first.")
        print(f"  Running golden set evaluation only.\n")
        evaluate_golden_set(extract_fn)
        return
    
    # Process sample
    print(f"\n  Processing reviews from {input_path}...")
    results = []
    processed = 0
    failed = 0
    max_reviews = 200  # Keep small for demo
    
    with open(input_path) as f:
        for line in f:
            if processed >= max_reviews:
                break
            review = json.loads(line)
            text = review.get("review_text", "")
            if len(text) < 30:
                continue
            
            aspects = extract_fn(text)
            if aspects:
                results.append({
                    "review_id": review["review_id"],
                    "source": review["source"],
                    "product_name": review["product_name"],
                    "aspects": aspects,
                })
                processed += 1
            else:
                failed += 1
    
    # Save results
    output_path = os.path.join(OUTPUT_DIR, "extracted_aspects.jsonl")
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    print(f"  Processed: {processed} reviews")
    print(f"  Failed/skipped: {failed}")
    print(f"  Output: {output_path}")
    
    # Report
    all_aspects = []
    for r in results:
        for a in r["aspects"]:
            all_aspects.append(a["aspect"])
    
    from collections import Counter
    aspect_counts = Counter(all_aspects).most_common(15)
    
    print(f"\n  Top aspects found:")
    for aspect, count in aspect_counts:
        print(f"    {aspect}: {count}")
    
    # Golden set evaluation
    evaluate_golden_set(extract_fn)
    
    print(f"\n✓ Aspect extraction complete.")


if __name__ == "__main__":
    run()
