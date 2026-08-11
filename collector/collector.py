import os
import json
import hashlib
import re
import time
from pathlib import Path
from urllib.parse import (
    urlsplit,
    urlunsplit,
    parse_qsl,
    urlencode,
)

import feedparser
import requests
from bs4 import BeautifulSoup
from mistralai.client import Mistral


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

SOURCES_FILE = ROOT / "sources.json"
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

MAX_CONTENT_CHARS = 18000
MIN_CONTENT_CHARS = 500

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
    "Mobile",
]


# ============================================================
# MISTRAL
# ============================================================

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY is missing"
    )

client = Mistral(
    api_key=API_KEY
)


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent":
        "AI-Knowledge-Collector/1.0"
}


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url):

    try:

        parts = urlsplit(url)

        query = []

        for key, value in parse_qsl(
            parts.query,
            keep_blank_values=True,
        ):

            key_lower = key.lower()

            if key_lower.startswith("utm_"):
                continue

            if key_lower in {
                "fbclid",
                "gclid",
                "ref",
                "source",
            }:
                continue

            query.append(
                (key, value)
            )

        return urlunsplit(
            (
                parts.scheme.lower(),
                parts.netloc.lower(),
                parts.path.rstrip("/"),
                urlencode(query),
                "",
            )
        )

    except Exception:

        return url.strip()


# ============================================================
# TITLE NORMALIZATION
# ============================================================

def normalize_title(title):

    title = title.lower()

    title = re.sub(
        r"[^\w\s]",
        " ",
        title,
        flags=re.UNICODE,
    )

    title = re.sub(
        r"\s+",
        " ",
        title,
    )

    return title.strip()


# ============================================================
# SHA256
# ============================================================

def sha256(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# ============================================================
# SOURCES
# ============================================================

def load_sources():

    with open(
        SOURCES_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# FETCH ARTICLE
# ============================================================

def fetch_article(url):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=20,
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser",
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
            "iframe",
        ]):

            tag.decompose()

        article = soup.find(
            "article"
        )

        if article:

            text = article.get_text(
                " ",
                strip=True,
            )

        else:

            text = soup.get_text(
                " ",
                strip=True,
            )

        text = clean_text(
            text
        )

        return text[
            :MAX_CONTENT_CHARS
        ]

    except Exception as e:

        print(
            f"[FETCH ERROR] {url}: {e}"
        )

        return ""


# ============================================================
# AI ANALYSIS - FIRST PASS
# ============================================================

def analyze_article(
    title,
    url,
    content,
):

    prompt = f"""
Bạn là AI curator chuyên xây dựng
kho kiến thức công nghệ.

Hãy phân tích bài viết dưới đây.

MỤC TIÊU:

- Chỉ giữ nội dung thực sự hữu ích.
- Loại spam.
- Loại clickbait.
- Loại quảng cáo.
- Loại nội dung quá mỏng.
- Loại nội dung chỉ mang tính giải trí.
- Không bịa thông tin.
- Không thêm thông tin không có trong bài.
- Tóm tắt chính xác.
- Chọn đúng category.

CATEGORY:

{", ".join(CATEGORIES)}

Chỉ trả JSON hợp lệ:

{{
  "useful": true,
  "category": "AI",
  "title": "...",
  "summary": "...",
  "score": 9,
  "tags": ["AI"],
  "key_points": [
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
                    "content": prompt,
                },
            ],

            response_format={
                "type": "json_object"
            },

            temperature=0.1,

            max_tokens=1200,
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
    content,
):

    category = result.get(
        "category",
        "Technology",
    )

    if category not in CATEGORIES:

        category = "Technology"

    normalized_url = normalize_url(
        url
    )

    normalized_title = normalize_title(
        title
    )

    content_hash = sha256(
        content
    )

    record = {

        "id":
            sha256(
                normalized_url
            )[:16],

        "title":
            result.get(
                "title",
                title,
            ),

        "url":
            normalized_url,

        "source":
            source,

        "category":
            category,

        "summary":
            result.get(
                "summary",
                "",
            ),

        "score":
            result.get(
                "score",
                0,
            ),

        "tags":
            result.get(
                "tags",
                [],
            ),

        "key_points":
            result.get(
                "key_points",
                [],
            ),

        "content_hash":
            content_hash,

        "title_hash":
            sha256(
                normalized_title
            ),

        "collected_at":
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        OUTPUT_DIR
        / f"worker-{WORKER_ID}.jsonl"
    )

    with open(
        output,
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    print(
        f"[ACCEPT] "
        f"{category} | "
        f"{record['score']}/10 | "
        f"{title}"
    )


# ============================================================
# COLLECT
# ============================================================

def collect():

    sources = load_sources()

    assigned_sources = [
        source
        for index, source
        in enumerate(sources)
        if index % TOTAL_WORKERS
        == WORKER_ID
    ]

    print(
        f"Worker "
        f"{WORKER_ID + 1}/"
        f"{TOTAL_WORKERS}"
    )

    print(
        f"Sources: "
        f"{len(assigned_sources)}"
    )

    processed = 0

    for source in assigned_sources:

        if processed >= MAX_ARTICLES:
            break

        name = source.get(
            "name",
            "Unknown",
        )

        rss = source.get(
            "rss"
        )

        if not rss:
            continue

        print(
            f"\n[SOURCE] {name}"
        )

        try:

            feed = feedparser.parse(
                rss
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
                    "",
                )
            )

            url = entry.get(
                "link",
                "",
            )

            if not title or not url:
                continue

            normalized_url = (
                normalize_url(url)
            )

            processed += 1

            print(
                f"\n[READ] {title}"
            )

            content = fetch_article(
                normalized_url
            )

            if len(content) < MIN_CONTENT_CHARS:

                print(
                    "[SKIP] Content too short"
                )

                continue

            result = analyze_article(
                title,
                normalized_url,
                content,
            )

            if not result:
                continue

            if not result.get(
                "useful",
                False,
            ):

                print(
                    "[REJECT] Not useful"
                )

                continue

            try:

                score = float(
                    result.get(
                        "score",
                        0,
                    )
                )

            except Exception:

                score = 0

            if score < 7:

                print(
                    f"[REJECT] "
                    f"Score={score}"
                )

                continue

            save_result(
                result,
                title,
                normalized_url,
                name,
                content,
            )

            time.sleep(1)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "AI KNOWLEDGE COLLECTOR"
    )

    print(
        f"Worker: "
        f"{WORKER_ID + 1}/"
        f"{TOTAL_WORKERS}"
    )

    print(
        f"Model: {MODEL}"
    )

    print("=" * 60)

    collect()
