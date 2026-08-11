import os
import json
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import feedparser
import requests
from bs4 import BeautifulSoup
from mistralai.client import Mistral


ROOT = Path(__file__).resolve().parent.parent

SOURCES_FILE = ROOT / "sources.json"
DATABASE_DIR = ROOT / "database"
OUTPUT_DIR = ROOT / "collector_output"


MODEL = os.getenv(
    "MISTRAL_MODEL",
    "mistral-large-latest"
)

WORKER_ID = int(
    os.getenv("WORKER_ID", "0")
)

TOTAL_WORKERS = int(
    os.getenv("TOTAL_WORKERS", "5")
)

MAX_ARTICLES = int(
    os.getenv("MAX_ARTICLES", "10")
)

MIN_SCORE = 7

MAX_CONTENT_CHARS = 18000
MIN_CONTENT_CHARS = 500

REQUEST_TIMEOUT = 20


CATEGORIES = [
    "AI",
    "OS",
    "Linux",
    "Windows",
    "Programming",
    "Cybersecurity",
    "Hardware",
    "Cloud",
    "DevOps",
    "Networking",
    "Science",
    "Space",
    "Technology",
    "OpenSource",
    "Gaming",
    "Mobile"
]


API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is not configured."
    )


client = Mistral(
    api_key=API_KEY
)


HEADERS = {
    "User-Agent":
        "Mozilla/5.0 "
        "(compatible; AIKnowledgeCollector/1.0)"
}


# ============================================================
# NORMALIZE URL
# ============================================================

def normalize_url(url):

    try:

        parts = urlsplit(url)

        query = []

        for key, value in parse_qsl(
            parts.query,
            keep_blank_values=True
        ):

            key_lower = key.lower()

            if key_lower.startswith("utm_"):
                continue

            if key_lower in [
                "fbclid",
                "gclid",
                "ref",
                "source"
            ]:
                continue

            query.append(
                (key, value)
            )

        return urlunsplit((
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            urlencode(query),
            ""
        ))

    except Exception:
        return url.strip()


# ============================================================
# NORMALIZE TITLE
# ============================================================

def normalize_title(title):

    title = title.lower()

    title = re.sub(
        r"[^\w\s]",
        " ",
        title,
        flags=re.UNICODE
    )

    title = re.sub(
        r"\s+",
        " ",
        title
    )

    return title.strip()


# ============================================================
# HASH
# ============================================================

def make_hash(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# LOAD SOURCES
# ============================================================

def load_sources():

    with open(
        SOURCES_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD EXISTING IDS
# ============================================================

def load_existing():

    ids = set()
    urls = set()
    titles = set()
    hashes = set()

    if not DATABASE_DIR.exists():
        return ids, urls, titles, hashes

    for file in DATABASE_DIR.rglob(
        "*.jsonl"
    ):

        try:

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:

                    try:

                        item = json.loads(
                            line
                        )

                    except Exception:
                        continue

                    if item.get("id"):
                        ids.add(
                            item["id"]
                        )

                    if item.get("url"):
                        urls.add(
                            normalize_url(
                                item["url"]
                            )
                        )

                    if item.get("title"):
                        titles.add(
                            normalize_title(
                                item["title"]
                            )
                        )

                    if item.get(
                        "content_hash"
                    ):
                        hashes.add(
                            item[
                                "content_hash"
                            ]
                        )

        except Exception:
            continue

    return ids, urls, titles, hashes


# ============================================================
# FETCH
# ============================================================

def fetch_article(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "noscript",
            "svg",
            "iframe"
        ]):

            tag.decompose()

        article = soup.find(
            "article"
        )

        if article:

            text = article.get_text(
                " ",
                strip=True
            )

        else:

            text = soup.get_text(
                " ",
                strip=True
            )

        text = clean_text(text)

        return text[:MAX_CONTENT_CHARS]

    except Exception as e:

        print(
            f"[FETCH ERROR] {url}: {e}"
        )

        return ""


# ============================================================
# MISTRAL
# ============================================================

def analyze_article(
    title,
    url,
    content
):

    prompt = f"""
Bạn là AI quản lý một kho kiến thức
công nghệ chất lượng cao.

Phân tích bài viết.

YÊU CẦU:

- Chỉ giữ nội dung thực sự hữu ích.
- Loại quảng cáo.
- Loại spam.
- Loại clickbait.
- Loại nội dung quá mỏng.
- Loại tin giải trí không có giá trị kỹ thuật.
- Không bịa thông tin.
- Tóm tắt bằng tiếng Việt.
- Chấm điểm 0-10.
- Chọn đúng một category.
- Tạo tags.
- Liệt kê key points.

ƯU TIÊN:

AI
Machine Learning
LLM
Linux
Windows
Programming
Open Source
Cybersecurity
Cloud
DevOps
Networking
Hardware
Science
Technology

CATEGORY:

{", ".join(CATEGORIES)}

Chỉ trả JSON:

{{
  "useful": true,
  "category": "AI",
  "title": "...",
  "summary": "...",
  "score": 9,
  "tags": ["AI", "LLM"],
  "key_points": [
    "...",
    "...",
    "..."
  ]
}}

TITLE:
{title}

URL:
{url}

CONTENT:
{content}
"""

    try:

        response = client.chat.complete(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content":
                        "Bạn là AI curator "
                        "chuyên lọc kiến thức. "
                        "Chỉ trả JSON hợp lệ."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            response_format={
                "type": "json_object"
            },

            temperature=0.1,

            max_tokens=1200
        )

        raw = (
            response
            .choices[0]
            .message
            .content
        )

        return json.loads(raw)

    except Exception as e:

        print(
            f"[MISTRAL ERROR] {e}"
        )

        return None


# ============================================================
# SAVE TEMP RESULT
# ============================================================

def save_result(
    result,
    title,
    url,
    source,
    content
):

    category = result.get(
        "category",
        "Technology"
    )

    if category not in CATEGORIES:

        category = "Technology"

    normalized_url = normalize_url(
        url
    )

    normalized_title = normalize_title(
        title
    )

    content_hash = make_hash(
        content
    )

    record = {

        "id": make_hash(
            normalized_url
        )[:16],

        "title": result.get(
            "title",
            title
        ),

        "url": normalized_url,

        "source": source,

        "category": category,

        "summary": result.get(
            "summary",
            ""
        ),

        "score": result.get(
            "score",
            0
        ),

        "tags": result.get(
            "tags",
            []
        ),

        "key_points": result.get(
            "key_points",
            []
        ),

        "content_hash": content_hash,

        "title_hash": make_hash(
            normalized_title
        ),

        "collected_at":
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR /
        f"worker-{WORKER_ID}.jsonl"
    )

    with open(
        output_file,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

    print(
        f"[ACCEPT] "
        f"{category} | "
        f"{result.get('score', 0)}/10 | "
        f"{title}"
    )


# ============================================================
# COLLECT
# ============================================================

def collect():

    sources = load_sources()

    existing_ids, existing_urls, existing_titles, existing_hashes = (
        load_existing()
    )

    # Mỗi worker phụ trách một phần nguồn
    assigned_sources = [
        source
        for index, source in enumerate(sources)
        if index % TOTAL_WORKERS == WORKER_ID
    ]

    print(
        f"Worker {WORKER_ID + 1}/{TOTAL_WORKERS}"
    )

    print(
        f"Assigned sources: "
        f"{len(assigned_sources)}"
    )

    processed = 0
    accepted = 0

    for source in assigned_sources:

        if processed >= MAX_ARTICLES:
            break

        name = source.get(
            "name",
            "Unknown"
        )

        rss_url = source.get(
            "rss"
        )

        if not rss_url:
            continue

        print()
        print(
            f"[SOURCE] {name}"
        )

        try:

            feed = feedparser.parse(
                rss_url
            )

        except Exception as e:

            print(
                f"[RSS ERROR] {e}"
            )

            continue

        for entry in feed.entries:

            if processed >= MAX_ARTICLES:
                break

            title = clean_text(
                entry.get(
                    "title",
                    ""
                )
            )

            url = entry.get(
                "link",
                ""
            )

            if not title or not url:
                continue

            normalized_url = normalize_url(
                url
            )

            title_key = normalize_title(
                title
            )

            url_id = make_hash(
                normalized_url
            )[:16]

            # ========================================
            # DUPLICATE CHECK #1
            # ========================================

            if url_id in existing_ids:
                continue

            if normalized_url in existing_urls:
                continue

            if title_key in existing_titles:
                continue

            processed += 1

            print()
            print(
                f"[READ] {title}"
            )

            content = fetch_article(
                url
            )

            if len(content) < MIN_CONTENT_CHARS:

                print(
                    "[SKIP] Too short"
                )

                continue

            content_hash = make_hash(
                content
            )

            # ========================================
            # DUPLICATE CHECK #2
            # ========================================

            if content_hash in existing_hashes:

                print(
                    "[DUPLICATE] Same content"
                )

                continue

            result = analyze_article(
                title,
                url,
                content
            )

            if not result:
                continue

            useful = result.get(
                "useful",
                False
            )

            try:

                score = float(
                    result.get(
                        "score",
                        0
                    )
                )

            except Exception:

                score = 0

            if not useful:

                print(
                    "[REJECT] Not useful"
                )

                continue

            if score < MIN_SCORE:

                print(
                    f"[REJECT] "
                    f"Score {score}/10"
                )

                continue

            save_result(
                result,
                title,
                url,
                name,
                content
            )

            accepted += 1

            time.sleep(1)

    print()
    print(
        f"[WORKER {WORKER_ID}] "
        f"Processed={processed} "
        f"Accepted={accepted}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 60
    )

    print(
        "AI KNOWLEDGE COLLECTOR"
    )

    print(
        f"Worker: {WORKER_ID + 1}/{TOTAL_WORKERS}"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        "=" * 60
    )

    collect()
