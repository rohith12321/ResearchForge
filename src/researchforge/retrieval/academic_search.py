from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET
import time


# ---------------------------------------------------------
# API endpoints
# ---------------------------------------------------------

ARXIV_API_URL = "https://export.arxiv.org/api/query"

SEMANTIC_SCHOLAR_API_URL = (
    "https://api.semanticscholar.org/graph/v1/paper/search"
)

WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"

ATOM_NAMESPACE = {
    "atom": "http://www.w3.org/2005/Atom"
}

USER_AGENT = "ResearchForge/1.0 (academic research assistant)"

# arXiv rate limiting: enforce a minimum gap between consecutive requests.
_last_arxiv_request_time = 0.0
ARXIV_REQUEST_INTERVAL = 3.0


# ---------------------------------------------------------
# Common result structures
# ---------------------------------------------------------

@dataclass
class Paper:
    """
    Result returned by the arXiv search function.

    Kept separate to preserve compatibility with the
    original arXiv implementation.
    """
    paper_id: str
    title: str
    authors: list[str]
    published: str
    abstract: str
    pdf_url: str


@dataclass
class SearchResult:
    """
    Common result format used when combining all sources.
    Some fields may be unavailable for certain sources.
    """
    source: str
    title: str
    url: str
    abstract: str
    authors: list[str]
    published: str
    paper_id: str = ""
    pdf_url: str = ""


# ---------------------------------------------------------
# General helper functions
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove extra whitespace from text."""
    return re.sub(r"\s+", " ", text).strip()


def make_request(url: str, retries: int = 3) -> bytes:
    """
    Send an HTTP GET request with retries for temporary failures.
    Uses exponential backoff for rate limits, server errors and
    network failures.
    """
    request = Request(
        url,
        headers={"User-Agent": USER_AGENT}
    )

    retryable_status_codes = {408, 429, 500, 502, 503, 504}

    for attempt in range(retries + 1):
        try:
            with urlopen(request, timeout=20) as response:
                return response.read()

        except HTTPError as error:
            if error.code not in retryable_status_codes:
                raise RuntimeError(
                    f"API returned HTTP {error.code}: {error.reason}"
                ) from error

            if attempt == retries:
                raise RuntimeError(
                    f"API request failed after {retries + 1} attempts. "
                    f"Last HTTP error: {error.code}"
                ) from error

            retry_after = error.headers.get("Retry-After")

            if retry_after and retry_after.isdigit():
                wait_time = min(int(retry_after), 30)
            else:
                wait_time = min(2 ** (attempt + 1), 8)

            print(
                f"HTTP {error.code}. Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)

        except (URLError, TimeoutError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Request failed after {retries + 1} attempts: {error}"
                ) from error

            wait_time = min(2 ** (attempt + 1), 8)

            print(
                f"Network error: {error}. "
                f"Retrying in {wait_time} seconds..."
            )
            time.sleep(wait_time)

    raise RuntimeError("Request failed after retries.")


class HTMLTextExtractor(HTMLParser):
    """Extract readable text from HTML snippets."""

    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return clean_text(" ".join(self.text_parts))


def strip_html(html: str) -> str:
    """Convert an HTML snippet into plain text."""
    parser = HTMLTextExtractor()
    parser.feed(html)
    return unescape(parser.get_text())


# =========================================================
# 1. arXiv SEARCH
# =========================================================

# Stop-words to strip from natural-language queries before
# building arXiv / Wikipedia search terms.
_STOP_WORDS = {
    # Question words
    "what", "which", "how", "why", "when", "where", "who", "whom",
    "whose", "whether",
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    # Prepositions / conjunctions
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "into", "about", "through", "between", "and", "or", "but",
    "nor", "so", "yet",
    # Auxiliary / copula verbs
    "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "can", "could", "will", "would", "shall", "should", "may",
    "might", "must",
    # Common filler words in research questions
    "their", "its", "it", "they", "them", "we", "us", "you",
    "he", "she", "his", "her", "our", "your",
    "also", "not", "no", "as", "if", "than", "then", "some",
    "any", "all", "each", "both", "such", "more", "most", "other",
    "different", "various", "based", "using", "used",
}


def _extract_key_terms(query: str) -> list[str]:
    """
    Extract meaningful content words from a natural-language query.

    - Strips punctuation (preserving hyphens so compound terms like
      'meta-learning' and 'few-shot' stay intact).
    - Removes stop-words and very short tokens (< 3 chars).
    """
    # Remove punctuation except hyphens
    cleaned = re.sub(r"[^\w\s-]", " ", query.lower())
    words = cleaned.split()

    content_words = [
        w for w in words
        if w not in _STOP_WORDS and len(w) >= 3
    ]

    if not content_words:
        # Fallback: return all non-trivial words
        content_words = [w for w in words if len(w) >= 3] or words

    return content_words


def _prioritize_terms(
    content_words: list[str],
    original_query: str,
    max_terms: int = 3
) -> list[str]:
    """
    Select the most topically important terms from a list of content words.

    Priority order:
    1. Hyphenated compound terms  (meta-learning, few-shot, model-agnostic)
    2. Acronyms – words that appear fully uppercase in the original query
       (MAML, NLP, RAG, BERT, etc.)
    3. Longer terms (more specific / technical vocabulary)
    """
    acronyms = {
        m.lower()
        for m in re.findall(r'\b[A-Z]{2,}\b', original_query)
    }

    def _sort_key(word: str):
        if '-' in word:
            return (0, -len(word))   # hyphenated compounds first
        if word in acronyms:
            return (1, -len(word))   # acronyms second
        return (2, -len(word))       # then by descending length

    return sorted(content_words, key=_sort_key)[:max_terms]


def simplify_query(query: str) -> str:
    """
    Reduce a natural-language question to its 2-3 most important search terms.

    Returns a space-separated string suitable for any search API.

    Example:
        "How does MAML function for few-shot learning tasks?"
        -> "few-shot maml learning"
    """
    if not query or not query.strip():
        return query.strip()

    content_words = _extract_key_terms(query)
    selected = _prioritize_terms(content_words, query, max_terms=3)
    return " ".join(selected)


def build_arxiv_query(
    query: str,
    field: str = "all",
    exact_phrase: bool = False
) -> str:
    """
    Convert a natural-language search query into arXiv query syntax.

    Strategy:
    - Short queries (<=4 content words after stop-word removal): treated
      as one exact phrase.
    - Longer queries: the 3 most important content words are selected
      (prioritising hyphenated compounds, then acronyms, then length)
      and AND-joined as individual terms.

    Examples:
        "meta-learning"  ->  all:"meta-learning"

        "few-shot learning MAML"  ->  all:"few-shot learning maml"

        "How does MAML function for few-shot image classification?"
        ->  all:"few-shot" AND all:"maml" AND all:"image"

        exact_phrase=True  ->  all:"<original query>"
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    valid_fields = {"all", "ti", "abs"}

    if field not in valid_fields:
        raise ValueError(
            f"Invalid field '{field}'. Choose from {valid_fields}."
        )

    query = query.strip()

    if exact_phrase:
        return f'{field}:"{query}"'

    content_words = _extract_key_terms(query)

    # Short query: treat the meaningful words as one exact phrase.
    if len(content_words) <= 4:
        phrase = " ".join(content_words)
        return f'{field}:"{phrase}"'

    # Longer query: pick the 3 most important terms and AND them.
    selected = _prioritize_terms(content_words, query, max_terms=3)
    return " AND ".join(f'{field}:"{term}"' for term in selected)


def search_arxiv(
    query: str,
    max_results: int = 10,
    field: str = "all",
    exact_phrase: bool = False,
    sort_by: str = "relevance",
    sort_order: str = "descending"
) -> list[Paper]:
    """
    Search arXiv and return a list of Paper objects.

    field:
        all  -> search all fields
        ti   -> search title
        abs  -> search abstract

    sort_by:
        relevance, lastUpdatedDate, submittedDate

    sort_order:
        ascending, descending
    """
    if max_results <= 0:
        raise ValueError("max_results must be positive.")

    valid_sort_fields = {
        "relevance",
        "lastUpdatedDate",
        "submittedDate"
    }

    valid_sort_orders = {"ascending", "descending"}

    if sort_by not in valid_sort_fields:
        raise ValueError("Invalid arXiv sort field.")

    if sort_order not in valid_sort_orders:
        raise ValueError("Invalid arXiv sort order.")

    search_query = build_arxiv_query(
        query,
        field=field,
        exact_phrase=exact_phrase
    )

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order
    }

    url = f"{ARXIV_API_URL}?{urlencode(params)}"

    # Enforce a minimum interval between consecutive arXiv requests
    # to avoid rate-limiting (HTTP 429).
    global _last_arxiv_request_time

    elapsed = time.monotonic() - _last_arxiv_request_time

    if elapsed < ARXIV_REQUEST_INTERVAL:
        wait_time = ARXIV_REQUEST_INTERVAL - elapsed
        print(
            f"Waiting {wait_time:.1f}s before next arXiv request..."
        )
        time.sleep(wait_time)

    response_data = make_request(url)
    _last_arxiv_request_time = time.monotonic()

    try:
        root = ET.fromstring(response_data)
    except ET.ParseError as error:
        raise RuntimeError("Could not parse arXiv XML response.") from error

    papers = []

    for entry in root.findall("atom:entry", ATOM_NAMESPACE):

        def get_element_text(tag):
            element = entry.find(tag, ATOM_NAMESPACE)

            if element is None or element.text is None:
                return ""

            return clean_text(element.text)

        paper_id = get_element_text("atom:id")
        title = get_element_text("atom:title")
        abstract = get_element_text("atom:summary")
        published = get_element_text("atom:published")

        authors = []

        for author in entry.findall("atom:author", ATOM_NAMESPACE):
            name = author.find("atom:name", ATOM_NAMESPACE)

            if name is not None and name.text:
                authors.append(clean_text(name.text))

        pdf_url = ""

        for link in entry.findall("atom:link", ATOM_NAMESPACE):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        papers.append(
            Paper(
                paper_id=paper_id,
                title=title,
                authors=authors,
                published=published,
                abstract=abstract,
                pdf_url=pdf_url
            )
        )

    return papers


# =========================================================
# 2. SEMANTIC SCHOLAR SEARCH
# =========================================================

def search_semantic_scholar(
    query: str,
    max_results: int = 10
) -> list[SearchResult]:
    """
    Search Semantic Scholar for academic papers.

    Uses the Semantic Scholar Academic Graph API.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if max_results <= 0:
        raise ValueError("max_results must be positive.")

    params = {
        "query": query.strip(),
        "limit": max_results,
        "fields": (
            "title,authors,year,abstract,url,"
            "openAccessPdf,publicationDate,externalIds"
        )
    }

    url = f"{SEMANTIC_SCHOLAR_API_URL}?{urlencode(params)}"

    response_data = make_request(url)

    try:
        data = json.loads(response_data)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Could not parse Semantic Scholar JSON response."
        ) from error

    results = []

    for paper in data.get("data", []):

        title = paper.get("title") or ""

        authors = [
            author.get("name", "")
            for author in paper.get("authors", [])
            if author.get("name")
        ]

        publication_date = (
            paper.get("publicationDate")
            or str(paper.get("year") or "")
        )

        open_access_pdf = paper.get("openAccessPdf") or {}

        results.append(
            SearchResult(
                source="Semantic Scholar",
                title=title,
                url=paper.get("url") or "",
                abstract=paper.get("abstract") or "",
                authors=authors,
                published=publication_date,
                paper_id=paper.get("paperId") or "",
                pdf_url=open_access_pdf.get("url") or ""
            )
        )

    return results


# =========================================================
# 3. WIKIPEDIA SEARCH
# =========================================================

def search_wikipedia(
    query: str,
    max_results: int = 10
) -> list[SearchResult]:
    """
    Search English Wikipedia for relevant articles.

    Returns article titles, URLs and introductory snippets.

    Wikipedia is intended to provide background context,
    not to replace academic papers.
    """
    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if max_results <= 0:
        raise ValueError("max_results must be positive.")

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query.strip(),
        "srlimit": max_results,
        "format": "json",
        "utf8": 1
    }

    url = f"{WIKIPEDIA_API_URL}?{urlencode(params)}"

    response_data = make_request(url)

    try:
        data = json.loads(response_data)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "Could not parse Wikipedia JSON response."
        ) from error

    search_results = data.get("query", {}).get("search", [])

    results = []

    for article in search_results:

        title = article.get("title", "")

        # Wikipedia article URLs use underscores instead of spaces.
        article_url = (
            "https://en.wikipedia.org/wiki/"
            + quote(title.replace(" ", "_"), safe="()_")
        )

        snippet = strip_html(article.get("snippet", ""))

        results.append(
            SearchResult(
                source="Wikipedia",
                title=title,
                url=article_url,
                abstract=snippet,
                authors=[],
                published=article.get("timestamp", "")
            )
        )

    return results


# =========================================================
# 4. COMBINED SEARCH
# =========================================================

def search_all(
    query: str,
    max_results: int = 10
) -> list[SearchResult]:

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    if max_results <= 0:
        raise ValueError("max_results must be positive.")

    combined_results = []

    # 1. arXiv (build_arxiv_query handles simplification internally).
    try:
        arxiv_results = search_arxiv(query, max_results)

        for paper in arxiv_results:
            combined_results.append(
                SearchResult(
                    source="arXiv",
                    title=paper.title,
                    url=paper.paper_id,
                    abstract=paper.abstract,
                    authors=paper.authors,
                    published=paper.published,
                    paper_id=paper.paper_id,
                    pdf_url=paper.pdf_url
                )
            )

        print(f"arXiv: {len(arxiv_results)} results")

    except Exception as error:
        print(f"arXiv search failed: {error}")

    # 2. Wikipedia — use simplified keywords so long natural-language
    #    questions don't return 0 results.  Wikipedia's srsearch works
    #    best with short keyword phrases, not full questions.
    try:
        wiki_query = simplify_query(query)
        wikipedia_results = search_wikipedia(wiki_query, max_results)

        combined_results.extend(wikipedia_results)

        print(f"Wikipedia: {len(wikipedia_results)} results")

    except Exception as error:
        print(f"Wikipedia search failed: {error}")

    return combined_results


# =========================================================
# 5. SIMPLE OUTPUT DISPLAY
# =========================================================

def print_results(results: list[SearchResult]) -> None:
    """Print search results in a readable format."""

    for index, result in enumerate(results, start=1):

        print(f"\n{'=' * 70}")
        print(f"Result {index} | {result.source}")
        print(f"{'=' * 70}")

        print(f"Title: {result.title}")

        if result.authors:
            print(f"Authors: {', '.join(result.authors)}")

        if result.published:
            print(f"Published: {result.published}")

        print(f"Abstract/Snippet: {result.abstract}")
        print(f"URL: {result.url}")

        if result.pdf_url:
            print(f"PDF: {result.pdf_url}")


# =========================================================
# 6. RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    query = "retrieval augmented generation"

    results = search_all(
        query=query,
        max_results=5
    )

    print_results(results)
