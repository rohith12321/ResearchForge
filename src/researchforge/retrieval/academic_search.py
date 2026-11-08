
from dataclasses import dataclass
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree as ET


ARXIV_API_URL = "https://export.arxiv.org/api/query"

ATOM_NAMESPACE = {
    "atom": "http://www.w3.org/2005/Atom"
}


@dataclass
class Paper:
    paper_id: str
    title: str
    authors: list[str]
    published: str
    abstract: str
    pdf_url: str


def build_arxiv_query(
    query: str,
    field: str = "all",
    exact_phrase: bool = False
) -> str:
    """
    Construct an arXiv API search query.

    Supported fields:
        all: Search all supported fields.
        ti: Search paper titles.
        abs: Search abstracts.

    Args:
        query: Search phrase or terms.
        field: Field to search.
        exact_phrase: Whether to search the entire phrase exactly.

    Returns:
        Formatted arXiv search query.
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

    terms = query.split()

    return " AND ".join(
        f'{field}:"{term}"' for term in terms
    )


def search_arxiv(
    query: str,
    max_results: int = 10,
    field: str = "all",
    exact_phrase: bool = False,
    sort_by: str = "relevance",
    sort_order: str = "descending",
) -> list[Paper]:
    """
    Search arXiv and return structured paper metadata.

    Args:
        query: Research topic or search phrase.
        max_results: Maximum number of papers to retrieve.
        field: Search field - all, ti, or abs.
        exact_phrase: Whether to search for the exact phrase.
        sort_by: relevance, lastUpdatedDate, or submittedDate.
        sort_order: ascending or descending.

    Returns:
        List of Paper objects.

    Raises:
        ValueError: For invalid arguments.
        urllib.error.URLError: If the API request fails.
        ET.ParseError: If the API returns invalid XML.
    """

    if max_results <= 0:
        raise ValueError("max_results must be positive.")

    valid_sort_fields = {
        "relevance",
        "lastUpdatedDate",
        "submittedDate",
    }

    if sort_by not in valid_sort_fields:
        raise ValueError(
            f"Invalid sort_by value: {sort_by}"
        )

    if sort_order not in {"ascending", "descending"}:
        raise ValueError(
            "sort_order must be ascending or descending."
        )

    search_query = build_arxiv_query(
        query=query,
        field=field,
        exact_phrase=exact_phrase,
    )

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": sort_by,
        "sortOrder": sort_order,
    }

    url = f"{ARXIV_API_URL}?{urlencode(params)}"

    with urlopen(url, timeout=30) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)

    papers = []

    for entry in root.findall("atom:entry", ATOM_NAMESPACE):

        paper_id = entry.findtext(
            "atom:id",
            default="",
            namespaces=ATOM_NAMESPACE
        )

        title = entry.findtext(
            "atom:title",
            default="",
            namespaces=ATOM_NAMESPACE
        )

        abstract = entry.findtext(
            "atom:summary",
            default="",
            namespaces=ATOM_NAMESPACE
        )

        published = entry.findtext(
            "atom:published",
            default="",
            namespaces=ATOM_NAMESPACE
        )

        authors = [
            author.findtext(
                "atom:name",
                default="",
                namespaces=ATOM_NAMESPACE
            )
            for author in entry.findall(
                "atom:author",
                ATOM_NAMESPACE
            )
        ]

        pdf_url = ""

        for link in entry.findall("atom:link", ATOM_NAMESPACE):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        papers.append(
            Paper(
                paper_id=paper_id.strip(),
                title=" ".join(title.split()),
                authors=authors,
                published=published,
                abstract=" ".join(abstract.split()),
                pdf_url=pdf_url,
            )
        )

    return papers
