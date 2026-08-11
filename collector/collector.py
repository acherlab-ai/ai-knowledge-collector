import os
import json
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from mistralai import Mistral


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIR = ROOT / "database"
SOURCES_FILE = ROOT / "sources.json"

MODEL = os.getenv(
    "MISTRAL_MODEL",
    "mistral-large-latest"
)

MAX_ARTICLES = int(
    os.getenv("MAX_ARTICLES", "30")
)

MAX_CONTENT = 18000

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


# ============================================================
# MISTRAL
# ============================================================

API_KEY = os.getenv("MISTRAL_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY chưa được thiết lập."
    )

client = Mistral(api_key=API_KEY)


# ============================================================
# UTILS
# ============================================================

def load_sources():
    with open(
        SOURCES_FILE,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def make_id(url):
    return hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]


def already_exists(url):
    article_id = make_id(url)

    if not DATABASE_DIR.exists():
        return False

    for file in DATABASE_DIR.rglob("*.jsonl"):
        try:
            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                for line in f:
                    if article_id in line:
                        return True

        except Exception:
            continue

    return False


# ============================================================
# ARTICLE FETCHER
# ============================================================

def fetch_article(url):

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(compatible; AIKnowledgeCollector/1.0)"
        )
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        # Remove useless elements
        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "noscript"
        ]):
            tag.decompose()

        text = clean_text(
            soup.get_text(" ")
        )

        return text[:MAX_CONTENT]

    except Exception as e:

        print(
            f"[FETCH ERROR] {url}: {e}"
        )

        return ""


# ============================================================
# MISTRAL ANALYSIS
# ============================================================

def analyze(title, url, content):

    categories = ", ".join(CATEGORIES)

    prompt = f"""
Bạn là hệ thống quản lý kho kiến thức công nghệ.

Hãy phân tích bài viết dưới đây.

NHIỆM VỤ:

1. Xác định bài viết có hữu ích hay không.
2. Tóm tắt nội dung bằng tiếng Việt.
3. Chấm điểm hữu ích từ 0 đến 10.
4. Chọn đúng MỘT category.
5. Tạo 3-8 tags.
6. Liệt kê các ý chính.
7. Không được bịa thông tin.
8. Nếu bài là quảng cáo, spam,
   nội dung quá mỏng hoặc không có giá trị
   thì useful=false.

CATEGORY ĐƯỢC PHÉP:

{categories}

TRẢ VỀ JSON:

{{
    "useful": true,
    "category": "AI",
    "title": "...",
    "summary": "...",
    "score": 8,
    "tags": [
        "LLM",
        "GPU",
        "Machine Learning"
    ],
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

ARTICLE:
{content}
"""

    try:

        response = client.chat.complete(

            model=MODEL,

            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là AI chuyên "
                        "phân loại và tóm tắt "
                        "kiến thức. "
                        "Chỉ trả về JSON hợp lệ."
                    )
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

        result = (
            response
            .choices[0]
            .message
            .content
        )

        return json.loads(result)

    except Exception as e:

        print(
            f"[MISTRAL ERROR] {e}"
        )

        return None


# ============================================================
# SAVE DATABASE
# ============================================================

def save_article(
    result,
    url,
    source
):

    category = result.get(
        "category",
        "Technology"
    )

    if category not in CATEGORIES:
        category = "Technology"

    today = datetime.now(
        timezone.utc
    ).strftime("%Y-%m-%d")

    category_dir = (
        DATABASE_DIR /
        category
    )

    category_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    file = (
        category_dir /
        f"{today}.jsonl"
    )

    record = {

        "id": make_id(url),

        "title": result.get(
            "title",
            ""
        ),

        "url": url,

        "source": source,

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

        "collected_at":
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    with open(
        file,
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
        f"[SAVED] "
        f"{category} | "
        f"{record['score']}/10 | "
        f"{record['title']}"
    )


# ============================================================
# COLLECT
# ============================================================

def collect():

    sources = load_sources()

    processed = 0
    saved = 0

    for source in sources:

        if processed >= MAX_ARTICLES:
            break

        name = source["name"]
        rss_url = source["rss"]

        print()
        print(
            f"========== {name} =========="
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

            title = entry.get(
                "title",
                ""
            )

            url = entry.get(
                "link",
                ""
            )

            if not title or not url:
                continue

            if already_exists(url):

                print(
                    f"[DUPLICATE] {title}"
                )

                continue

            processed += 1

            print()
            print(
                f"[READ] {title}"
            )

            content = fetch_article(url)

            if len(content) < 500:

                print(
                    "[SKIP] Nội dung quá ngắn"
                )

                continue

            result = analyze(
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

            score = result.get(
                "score",
                0
            )

            if not useful:

                print(
                    "[REJECT] Không hữu ích"
                )

                continue

            if score < 6:

                print(
                    f"[REJECT] Score {score}/10"
                )

                continue

            save_article(
                result,
                url,
                name
            )

            saved += 1

    return saved


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=========================================="
    )
    print(
        "      AI KNOWLEDGE COLLECTOR"
    )
    print(
        "=========================================="
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Max articles: {MAX_ARTICLES}"
    )

    print()

    total = collect()

    print()
    print(
        "=========================================="
    )
    print(
        f"Finished. Saved: {total}"
    )
    print(
        "=========================================="
    )
