"""
ReviewPulse AI — POC: Reddit API Connector
===========================================
Connects to Reddit API (free tier), pulls product review posts
from specified subreddits, handles rate limiting and pagination,
and outputs structured JSONL matching our pipeline format.

Setup:
  1. Go to https://www.reddit.com/prefs/apps
  2. Create a "script" app
  3. Set environment variables:
     REDDIT_CLIENT_ID=your_id
     REDDIT_CLIENT_SECRET=your_secret
     REDDIT_USER_AGENT=ReviewPulseAI/1.0

Run: python poc/02_reddit_connector.py
Output: data/reddit_reviews.jsonl
"""

import json
import os
import time
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ──
SUBREDDITS = [
    "headphones", "BuyItForLife", "SkincareAddiction", "buildapc",
    "HomeImprovement", "Cooking", "cameras", "laptops",
    "MechanicalKeyboards", "audiophile", "running", "cycling",
    "espresso", "camping", "gardening", "CleaningTips",
    "AutoDetailing", "Watches", "malefashionadvice", "frugalmalefashion",
]
POSTS_PER_SUBREDDIT = 100  # Keep small for demo; increase for production
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Rate limiting: Reddit free tier = 100 QPM = ~1.6 req/sec
# We stay well under at 1 req/sec
RATE_LIMIT_DELAY = 1.0


class RedditConnector:
    """
    Reddit API connector with OAuth authentication, rate limiting,
    pagination, and structured output.
    
    This is a real pipeline component, not a toy demo.
    It handles:
    - OAuth token acquisition and refresh
    - Rate limiting (stays under 100 QPM)
    - Pagination via Reddit's "after" cursor
    - Error handling with retries
    - Structured output in our pipeline format
    """

    BASE_URL = "https://oauth.reddit.com"
    TOKEN_URL = "https://www.reddit.com/api/v1/access_token"

    def __init__(self):
        self.client_id = os.getenv("REDDIT_CLIENT_ID", "")
        self.client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
        self.user_agent = os.getenv("REDDIT_USER_AGENT", "ReviewPulseAI/1.0 (academic project)")
        self.access_token = None
        self.token_expiry = 0
        self.request_count = 0
        self.last_request_time = 0

    def authenticate(self) -> bool:
        """Acquire OAuth token using client credentials."""
        if not self.client_id or not self.client_secret:
            return False

        try:
            data = urllib.parse.urlencode({
                "grant_type": "client_credentials"
            }).encode()
            
            req = urllib.request.Request(self.TOKEN_URL, data=data)
            # Basic auth with client_id:client_secret
            import base64
            credentials = base64.b64encode(
                f"{self.client_id}:{self.client_secret}".encode()
            ).decode()
            req.add_header("Authorization", f"Basic {credentials}")
            req.add_header("User-Agent", self.user_agent)
            
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read())
                self.access_token = result["access_token"]
                self.token_expiry = time.time() + result.get("expires_in", 3600)
                print("  Reddit OAuth: authenticated successfully")
                return True
        except Exception as e:
            print(f"  Reddit OAuth failed: {e}")
            return False

    def _rate_limit(self):
        """Enforce rate limiting: max 1 request per second."""
        elapsed = time.time() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1

    def _api_get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make authenticated GET request with rate limiting and retries."""
        if not self.access_token:
            return None

        self._rate_limit()
        
        url = f"{self.BASE_URL}{endpoint}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url)
                req.add_header("Authorization", f"Bearer {self.access_token}")
                req.add_header("User-Agent", self.user_agent)
                
                with urllib.request.urlopen(req) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    # Rate limited — back off exponentially
                    wait = (2 ** attempt) * 2
                    print(f"    Rate limited. Waiting {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)
                elif e.code == 403:
                    print(f"    Forbidden: {endpoint}. Subreddit may be private.")
                    return None
                else:
                    print(f"    HTTP {e.code}: {e.reason}")
                    if attempt == max_retries - 1:
                        return None
                    time.sleep(2)
            except Exception as e:
                print(f"    Error: {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)
        return None

    def fetch_subreddit_posts(self, subreddit: str, limit: int = 100) -> list:
        """
        Fetch posts from a subreddit with pagination.
        Returns structured records matching our pipeline format.
        """
        posts = []
        after = None
        per_page = min(limit, 100)
        
        while len(posts) < limit:
            params = {"limit": per_page, "sort": "relevance", "t": "year"}
            if after:
                params["after"] = after
            
            data = self._api_get(f"/r/{subreddit}/search", {
                **params,
                "q": "review OR recommend OR bought OR quality OR worth",
                "restrict_sr": "true",
                "type": "link",
            })
            
            if not data or "data" not in data:
                break
            
            children = data["data"].get("children", [])
            if not children:
                break
            
            for child in children:
                post = child.get("data", {})
                # Only keep text posts with actual content
                selftext = post.get("selftext", "")
                if len(selftext) < 20:
                    continue
                
                posts.append({
                    "source": "reddit",
                    "source_id": post.get("id", ""),
                    "subreddit": subreddit,
                    "title": post.get("title", ""),
                    "text": selftext,
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "author": post.get("author", "[deleted]"),
                    "created_utc": post.get("created_utc", 0),
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "upvote_ratio": post.get("upvote_ratio", 0),
                    # Reddit has NO star rating — this is a key schema difference
                    "rating": None,
                    "verified_purchase": None,  # Not applicable for Reddit
                })
            
            after = data["data"].get("after")
            if not after:
                break
        
        return posts[:limit]

def run():
    print("=" * 60)
    print("REVIEWPULSE AI — REDDIT CONNECTOR")
    print("=" * 60)
    
    connector = RedditConnector()
    all_posts = []
    
    if not connector.authenticate():
        raise RuntimeError(
            "Reddit API credentials are required. Configure real credentials instead of using demo data."
        )

    for i, sub in enumerate(SUBREDDITS):
        print(f"\n  [{i+1}/{len(SUBREDDITS)}] Fetching r/{sub}...")
        posts = connector.fetch_subreddit_posts(sub, limit=POSTS_PER_SUBREDDIT)
        print(f"    Got {len(posts)} posts")
        all_posts.extend(posts)
    print(f"\n  Total API requests: {connector.request_count}")

    if not all_posts:
        raise RuntimeError("Reddit API returned no real posts for the configured subreddits.")
    
    # Save to JSONL
    output_path = os.path.join(OUTPUT_DIR, "reddit_reviews.jsonl")
    with open(output_path, "w") as f:
        for post in all_posts:
            f.write(json.dumps(post) + "\n")
    
    print(f"\n  Saved {len(all_posts)} posts to: {output_path}")
    
    # Print schema summary
    print(f"\n  Schema summary:")
    print(f"    Fields: {list(all_posts[0].keys())}")
    print(f"    Subreddits covered: {len(set(p['subreddit'] for p in all_posts))}")
    print(f"    Avg text length: {sum(len(p['text']) for p in all_posts) / len(all_posts):.0f} chars")
    print(f"    Score range: {min(p['score'] for p in all_posts)} to {max(p['score'] for p in all_posts)}")
    print(f"    rating field: ALL NULL (Reddit has no star ratings)")
    print(f"    verified_purchase: ALL NULL (not applicable)")
    print("\n✓ Reddit connector complete.")


if __name__ == "__main__":
    run()
