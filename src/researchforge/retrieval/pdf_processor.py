import pymupdf
import requests
from pathlib import Path
from dataclasses import dataclass


@dataclass
class ProcessedPaper:
    paper_id: str
    title: str
    text: str
    pdf_path: str


def download_pdf(pdf_url: str, save_path: str) -> str:
    """
    Download a research paper PDF and save it locally.
    """
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)

    return str(path)


def extract_text(pdf_path: str) -> str:
    """
    Extract text from all pages of a PDF.
    """
    document = pymupdf.open(pdf_path)

    pages = []

    try:
        for page in document:
            pages.append(page.get_text("text"))
    finally:
        document.close()

    return "\n".join(pages).strip()


def process_paper(
    paper_id: str,
    title: str,
    pdf_url: str,
    download_dir: str = "data/raw"
) -> ProcessedPaper:
    """
    Download a paper and extract its text.
    """
    filename = paper_id.rstrip("/").split("/")[-1]
    filename = filename.replace(":", "_") + ".pdf"

    pdf_path = Path(download_dir) / filename

    if not pdf_path.exists():
        download_pdf(pdf_url, str(pdf_path))

    text = extract_text(str(pdf_path))

    if not text:
        raise ValueError(
            f"No extractable text found in PDF: {title}"
        )

    return ProcessedPaper(
        paper_id=paper_id,
        title=title,
        text=text,
        pdf_path=str(pdf_path)
    )
