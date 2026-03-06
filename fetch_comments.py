#!/usr/bin/env python3
"""Fetch YouTube comments for ads in the dataset.

Usage:
    # Using youtube-comment-downloader (no API key needed):
    python fetch_comments.py

    # Using YouTube Data API (more reliable, needs free API key):
    YOUTUBE_API_KEY=your_key python fetch_comments.py

    # Limit comments per video:
    python fetch_comments.py --max-comments 50
"""

import argparse
import os
import re
import sys
import time

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def extract_video_id(url):
    """Extract YouTube video ID from a URL."""
    if not isinstance(url, str):
        return None
    match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
    return match.group(1) if match else None


def fetch_with_downloader(video_id, max_comments=100):
    """Fetch comments using youtube-comment-downloader (no API key)."""
    try:
        from youtube_comment_downloader import YoutubeCommentDownloader

        downloader = YoutubeCommentDownloader()
        comments = []
        for comment in downloader.get_comments_from_url(
            f"https://www.youtube.com/watch?v={video_id}"
        ):
            comments.append({
                "video_id": video_id,
                "author": comment.get("author", ""),
                "text": comment.get("text", ""),
                "likes": comment.get("votes", 0),
                "time": comment.get("time", ""),
            })
            if len(comments) >= max_comments:
                break
        return comments
    except ImportError:
        print("  Install: pip install youtube-comment-downloader")
        return []
    except Exception as e:
        print(f"  Error: {e}")
        return []


def fetch_with_api(video_id, api_key, max_comments=100):
    """Fetch comments using YouTube Data API v3."""
    try:
        from googleapiclient.discovery import build

        youtube = build("youtube", "v3", developerKey=api_key)
        comments = []
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=min(max_comments, 100),
            order="relevance",
        )
        while request and len(comments) < max_comments:
            response = request.execute()
            for item in response.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id": video_id,
                    "author": snippet.get("authorDisplayName", ""),
                    "text": snippet.get("textDisplay", ""),
                    "likes": snippet.get("likeCount", 0),
                    "time": snippet.get("publishedAt", ""),
                })
            request = youtube.commentThreads().list_next(request, response)
        return comments[:max_comments]
    except Exception as e:
        print(f"  API error: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Fetch YouTube comments for Super Bowl ads"
    )
    parser.add_argument(
        "--max-comments", type=int, default=50,
        help="Maximum comments to fetch per video (default: 50)",
    )
    args = parser.parse_args()

    ads_path = os.path.join(DATA_DIR, "superbowl_ads.csv")
    if not os.path.exists(ads_path):
        print(f"Dataset not found at {ads_path}. Run the Streamlit app first.")
        sys.exit(1)

    ads_df = pd.read_csv(ads_path)
    api_key = os.environ.get("YOUTUBE_API_KEY")
    method = "YouTube Data API" if api_key else "youtube-comment-downloader"
    print(f"Method: {method}")
    print(f"Max comments per video: {args.max_comments}\n")

    all_comments = []
    urls = ads_df[ads_df["youtube_url"].notna()]["youtube_url"].unique()

    for i, url in enumerate(urls):
        video_id = extract_video_id(str(url))
        if not video_id:
            continue

        row = ads_df[ads_df["youtube_url"] == url].iloc[0]
        brand = row["brand"]
        year = int(row["year"])
        print(f"[{i + 1}/{len(urls)}] {brand} ({year}) ...", end=" ")

        comments = (
            fetch_with_api(video_id, api_key, args.max_comments)
            if api_key
            else fetch_with_downloader(video_id, args.max_comments)
        )

        for c in comments:
            c["brand"] = brand
            c["year"] = year

        all_comments.extend(comments)
        print(f"{len(comments)} comments")
        time.sleep(1)  # rate-limit

    if all_comments:
        output = os.path.join(DATA_DIR, "youtube_comments.csv")
        pd.DataFrame(all_comments).to_csv(output, index=False)
        print(f"\nSaved {len(all_comments)} total comments to {output}")
    else:
        print("\nNo comments collected. Check your internet or API key.")


if __name__ == "__main__":
    main()
