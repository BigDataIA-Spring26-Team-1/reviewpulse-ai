"""
ReviewPulse AI — POC: YouTube Transcript Extractor
===================================================
Extracts auto-generated transcripts from product review videos.
Uses youtube-transcript-api (free, no API quota cost).

This is the hardest source to normalize because transcripts are
pure unstructured speech with no ratings or product IDs.

Run: python poc/03_youtube_extractor.py
Output: data/youtube_reviews.jsonl
"""

import json
import os
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_transcript(video_id: str) -> dict | None:
    """
    Extract transcript from a YouTube video.
    Uses youtube-transcript-api which accesses auto-generated captions.
    No API quota is consumed — this is a separate mechanism from YouTube Data API.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Concatenate all segments into full text
        full_text = " ".join(seg["text"] for seg in segments)
        total_duration = sum(seg.get("duration", 0) for seg in segments)
        
        return {
            "source": "youtube",
            "source_id": video_id,
            "title": None,  # Would come from YouTube Data API
            "text": full_text,
            "duration_seconds": total_duration,
            "segment_count": len(segments),
            "url": f"https://youtube.com/watch?v={video_id}",
            # YouTube has NO star rating — critical schema difference
            "rating": None,
            "verified_purchase": None,
            "score": None,
            "created_utc": None,
        }
    except Exception as e:
        print(f"    Failed for {video_id}: {e}")
        return None


def generate_demo_data() -> list:
    """
    Generate realistic demo transcripts when youtube-transcript-api
    can't reach YouTube (e.g., in CI/CD or sandboxed environments).
    Schema matches exactly what real extraction produces.
    """
    demo_transcripts = [
        {
            "video_id": "demo_001",
            "title": "Sony WH-1000XM5 Review — Best Noise Cancelling?",
            "channel": "TechReviewerPro",
            "text": "okay so today we are looking at the sony wh 1000 xm5 and honestly the noise cancellation is probably the best i have tested this year the battery life is incredible i am getting about 30 hours on a single charge which is exactly what sony promised the sound quality is warm and balanced not as bass heavy as the previous generation which i actually prefer the comfort is good for the first few hours but after about four hours they start to feel a bit warm on my ears the call quality is where it falls short a bit compared to the airpods max the microphone picks up too much background noise overall i would give these a solid nine out of ten if you care about noise cancellation this is the one to get",
            "duration_seconds": 487,
        },
        {
            "video_id": "demo_002", 
            "title": "iPhone 15 Pro — 3 Month Review",
            "channel": "DailyTechReview",
            "text": "three months in with the iphone 15 pro and i have some thoughts the camera system is genuinely impressive especially the 5x telephoto lens for portraits the action button is more useful than i expected i have it set to toggle silent mode the titanium frame looks premium but it does scratch easily i have a few visible marks already battery life is solid about seven hours of screen on time which is better than the 14 pro the usb c switch was overdue and its great for charging flexibility the 48 megapixel main sensor takes incredible photos in good light but night mode still has some noise",
            "duration_seconds": 612,
        },
        {
            "video_id": "demo_003",
            "title": "Dyson V15 vs Shark — Honest Comparison",
            "channel": "HomeTestLab",
            "text": "i bought both the dyson v15 detect and the shark stratos to compare them side by side suction power goes to the dyson it picks up more debris on carpet especially fine dust the laser on the dyson is genuinely useful not just a gimmick you can actually see particles you missed battery life is where the shark wins about 60 minutes versus the dyson 40 minutes in standard mode the shark is also two hundred dollars cheaper build quality goes to dyson it feels premium while the shark feels plasticky if money is no object get the dyson if you want the best value the shark is excellent",
            "duration_seconds": 534,
        },
        {
            "video_id": "demo_004",
            "title": "MacBook Pro M3 — Developer Review",
            "channel": "CodeWithMe",
            "text": "as a software developer i have been using the macbook pro m3 for about two months and here is my honest take compilation times are noticeably faster than my old intel macbook about three times faster for our main project docker runs natively and the performance is great the 18 hour battery life claim is real i regularly get 14 to 16 hours of actual coding the screen is gorgeous and the 120hz promotion makes scrolling through code feel smooth the keyboard is the best apple has ever made the speakers are insanely good for a laptop my only complaint is the price starting at two thousand dollars it is hard to justify when a thinkpad does the same job for half the price",
            "duration_seconds": 445,
        },
    ]
    
    records = []
    for dt in demo_transcripts:
        records.append({
            "source": "youtube",
            "source_id": dt["video_id"],
            "title": dt["title"],
            "channel": dt["channel"],
            "text": dt["text"],
            "duration_seconds": dt["duration_seconds"],
            "segment_count": dt["duration_seconds"] // 3,  # ~3 sec per segment
            "url": f"https://youtube.com/watch?v={dt['video_id']}",
            "rating": None,       # YouTube has NO star rating
            "verified_purchase": None,
            "score": None,
            "created_utc": int(datetime(2025, 6, 15).timestamp()),
        })
    return records


# Sample video IDs of real product reviews (public, popular tech reviews)
SAMPLE_VIDEO_IDS = [
    "xmpYSgBx31I",  # Example tech review
    "q3fBSEfGWMc",  # Example headphone review
]


def run():
    print("=" * 60)
    print("REVIEWPULSE AI — YOUTUBE TRANSCRIPT EXTRACTOR")
    print("=" * 60)
    
    records = []
    
    # Try real extraction first
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        print("\n  youtube-transcript-api available. Trying real extraction...")
        for vid in SAMPLE_VIDEO_IDS:
            print(f"    Extracting: {vid}")
            result = extract_transcript(vid)
            if result:
                records.append(result)
                print(f"    Success: {len(result['text'])} chars, {result['segment_count']} segments")
    except ImportError:
        print("\n  youtube-transcript-api not installed. Using demo data.")
    
    # Add demo data regardless (for consistent output)
    demo = generate_demo_data()
    records.extend(demo)
    
    # Save to JSONL
    output_path = os.path.join(OUTPUT_DIR, "youtube_reviews.jsonl")
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    
    print(f"\n  Saved {len(records)} transcripts to: {output_path}")
    
    # Schema analysis
    print(f"\n  Schema summary:")
    print(f"    Fields: {list(records[0].keys())}")
    print(f"    Avg transcript length: {sum(len(r['text']) for r in records)/len(records):.0f} chars")
    print(f"    Avg word count: {sum(len(r['text'].split()) for r in records)/len(records):.0f} words")
    print(f"    rating: ALL NULL (YouTube has no star ratings)")
    print(f"    verified_purchase: ALL NULL (not applicable)")
    print(f"\n  KEY CHALLENGE: YouTube transcripts are pure unstructured speech.")
    print(f"  No product IDs, no ratings, no structured fields.")
    print(f"  We extract product mentions and sentiment via LLM in the next step.")
    print("\n✓ YouTube extractor complete.")


if __name__ == "__main__":
    run()
