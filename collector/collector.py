import os
import json
import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests
from bs4 import BeautifulSoup
from mistralai.client import Mistral


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATABASE_DIR = ROOT / "database"
SOURCES_FILE = ROOT / "sources.json"


# ============================================================
# CONFIG
# ============================================================

MODEL = os.getenv(
    "MISTRAL_MODEL",
    "mistral-large-latest"
)

MAX_ARTICLES = int(
    os.getenv("MAX_ARTICLES", "30")
)

MAX_CONTENT_CHARS = 18000

MIN_CONTENT_CHARS = 500

MIN_SCORE = 6

REQUEST_TIMEOUT = 20

SLEEP_BETWEEN_REQUESTS = 1


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
        "MISTRAL_API_KEY is not configured."
    )

client = Mistral(
    api_key=API_KEY
)


# ============================================================
# HTTP
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(compatible; AIKnowledgeCollector/1.0)"
    )
}


# ============================================================
# LOAD SOURCES
# ============================================================

def load_sources():

    if not SOURCES_FILE.exists():

        raise FileNotFoundError(
            f"Missing {SOURCES_FILE}"
        )

    with open(
        SOURCES_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    if not isinstance(data, list):

        raise ValueError(
            "sources.json must contain a JSON array."
        )

    return data


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# ARTICLE ID
# ============================================================

def make_id(url):

    return hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]


# ============================================================
# CHECK DUPLICATE
# ============================================================

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

        except Exception as e:

            print(
                f"[WARNING] Cannot read {file}: {e}"
            )

    return False


# ============================================================
# FETCH ARTICLE
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

        # Remove useless HTML
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

        # Prefer article content
        article = soup.find("article")

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

    except requests.RequestException as e:

        print(
            f"[FETCH ERROR] {url}"
        )

        print(
            f"           {e}"
        )

        return ""

    except Exception as e:

        print(
            f"[PARSE ERROR] {url}: {e}"
        )

        return ""


# ============================================================
# MISTRAL ANALYSIS
# ============================================================

def analyze_article(
    title,
    url,
    content
):

    category_text = ", ".join(
        CATEGORIES
    )

    prompt = f"""
Bạn là AI quản lý một kho kiến thức
công nghệ tự động.

Hãy phân tích bài viết dưới đây.

NHIỆM VỤ:

1. Xác định bài viết có hữu ích hay không.
2. Tóm tắt chính xác bằng tiếng Việt.
3. Chấm điểm hữu ích từ 0 đến 10.
4. Chọn MỘT category phù hợp nhất.
5. Tạo từ 3 đến 8 tags.
6. Liệt kê các ý chính quan trọng.
7. Không được bịa thông tin.
8. Không suy diễn những điều bài viết không nói.
9. Nếu bài chủ yếu là quảng cáo, spam,
   clickbait hoặc nội dung quá mỏng:
   useful=false.
10. Nếu bài có thông tin kỹ thuật,
    nghiên cứu, cập nhật hoặc hướng dẫn
    có giá trị thì useful=true.

CATEGORY ĐƯỢC PHÉP:

{category_text}

CHỈ TRẢ VỀ JSON.

FORMAT:

{{
    "useful": true,
    "category": "AI",
    "title": "Tiêu đề",
    "summary": "Tóm tắt bằng tiếng Việt",
    "score": 8,
    "tags": [
        "LLM",
        "Machine Learning"
    ],
    "key_points": [
        "Ý chính 1",
        "Ý chính 2",
        "Ý chính 3"
    ]
}}

TITLE:
{title}

URL:
{url}

ARTICLE CONTENT:
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
                        "phân tích, tóm tắt và "
                        "phân loại dữ liệu. "
                        "Luôn trả về JSON hợp lệ."
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

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:

            print(
                "[AI ERROR] Empty response."
            )

            return None

        result = json.loads(content)

        return result

    except json.JSONDecodeError as e:

        print(
            f"[JSON ERROR] {e}"
        )

        return None

    except Exception as e:

        print(
            f"[MISTRAL ERROR] {e}"
        )

        return None


# ============================================================
# VALIDATE AI RESULT
# ============================================================

def validate_result(result):

    if not isinstance(
        result,
        dict
    ):

        return False

    required = [
        "useful",
        "category",
        "title",
        "summary",
        "score",
        "tags",
        "key_points"
    ]

    for field in required:

        if field not in result:
            return False

    category = result["category"]

    if category not in CATEGORIES:

        result["category"] = "Technology"

    try:

        score = float(
            result["score"]
        )

    except:

        result["score"] = 0

    if score < 0:
        result["score"] = 0

    if score > 10:
        result["score"] = 10

    return True


# ============================================================
# SAVE ARTICLE
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
    ).strftime(
        "%Y-%m-%d"
    )

    category_dir = (
        DATABASE_DIR /
        category
    )

    category_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
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
        output_file,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
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
# PROCESS SOURCE
# ============================================================

def process_source(
    source,
    processed
):

    name = source.get(
        "name",
        "Unknown"
    )

    rss_url = source.get(
        "rss"
    )

    if not rss_url:

        print(
            f"[SKIP] {name}: missing RSS URL"
        )

        return processed, 0

    print()
    print(
        "=" * 60
    )

    print(
        f"SOURCE: {name}"
    )

    print(
        "=" * 60
    )

    try:

        feed = feedparser.parse(
            rss_url
        )

    except Exception as e:

        print(
            f"[RSS ERROR] {e}"
        )

        return processed, 0

    saved = 0

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

        if already_exists(url):

            print(
                f"[DUPLICATE] {title}"
            )

            continue

        processed += 1

        print()
        print(
            f"[READ {processed}/{MAX_ARTICLES}]"
        )

        print(
            title
        )

        article = fetch_article(
            url
        )

        if len(article) < MIN_CONTENT_CHARS:

            print(
                "[SKIP] Content too short."
            )

            continue

        result = analyze_article(
            title,
            url,
            article
        )

        if not result:

            continue

        if not validate_result(
            result
        ):

            print(
                "[SKIP] Invalid AI result."
            )

            continue

        useful = result.get(
            "useful",
            False
        )

        score = float(
            result.get(
                "score",
                0
            )
        )

        if not useful:

            print(
                "[REJECT] AI marked as useless."
            )

            continue

        if score < MIN_SCORE:

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

        time.sleep(
            SLEEP_BETWEEN_REQUESTS
        )

    return processed, saved


# ============================================================
# MAIN COLLECTOR
# ============================================================

def collect():

    sources = load_sources()

    print(
        f"Sources: {len(sources)}"
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Max articles: {MAX_ARTICLES}"
    )

    print()

    processed = 0
    total_saved = 0

    for source in sources:

        if processed >= MAX_ARTICLES:
            break

        processed, saved = process_source(
            source,
            processed
        )

        total_saved += saved

    return processed, total_saved


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=" * 60
    )

    print(
        "        AI KNOWLEDGE COLLECTOR"
    )

    print(
        "=" * 60
    )

    print(
        f"Model: {MODEL}"
    )

    print(
        f"Database: {DATABASE_DIR}"
    )

    print(
        "=" * 60
    )

    try:

        processed, saved = collect()

        print()
        print(
            "=" * 60
        )

        print(
            "FINISHED"
        )

        print(
            f"Articles processed: {processed}"
        )

        print(
            f"Articles saved:     {saved}"
        )

        print(
            "=" * 60
        )

    except Exception as e:

        print()
        print(
            "=" * 60
        )

        print(
            "COLLECTOR FAILED"
        )

        print(
            str(e)
        )

        print(
            "=" * 60
        )

        raise


if __name__ == "__main__":
    main()
