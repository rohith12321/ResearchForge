
import unittest
from io import BytesIO
from unittest.mock import patch

from src.researchforge.retrieval.academic_search import (
    Paper,
    SearchResult,
    build_arxiv_query,
    search_arxiv,
    search_semantic_scholar,
    search_wikipedia,
    search_all,
    print_results
)


SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">

    <entry>
        <id>http://arxiv.org/abs/2401.12345</id>

        <title>
            Retrieval Augmented Generation for Scientific Research
        </title>

        <summary>
            This paper studies retrieval augmented generation
            for scientific literature.
        </summary>

        <published>2024-01-20T12:00:00Z</published>

        <author>
            <name>Alice Smith</name>
        </author>

        <author>
            <name>Bob Jones</name>
        </author>

        <link
            href="https://arxiv.org/pdf/2401.12345"
            title="pdf"
            type="application/pdf"
        />

    </entry>

</feed>
"""


class TestAcademicSearch(unittest.TestCase):

    # ------------------------------------------
    # Original search functionality tests
    # ------------------------------------------

    @patch("src.researchforge.academic_search.urlopen")
    def test_search_returns_paper(self, mock_urlopen):

        mock_urlopen.return_value = BytesIO(SAMPLE_XML)

        papers = search_arxiv("retrieval augmented generation")

        self.assertEqual(len(papers), 1)

        paper = papers[0]

        self.assertEqual(
            paper.paper_id,
            "http://arxiv.org/abs/2401.12345"
        )

        self.assertEqual(
            paper.title,
            "Retrieval Augmented Generation for Scientific Research"
        )

        self.assertEqual(
            paper.authors,
            ["Alice Smith", "Bob Jones"]
        )

        self.assertEqual(
            paper.published,
            "2024-01-20T12:00:00Z"
        )

        self.assertEqual(
            paper.abstract,
            "This paper studies retrieval augmented generation for scientific literature."
        )

        self.assertEqual(
            paper.pdf_url,
            "https://arxiv.org/pdf/2401.12345"
        )

    @patch("src.researchforge.academic_search.urlopen")
    def test_empty_search_results(self, mock_urlopen):

        empty_xml = b"""
        <feed xmlns="http://www.w3.org/2005/Atom">
        </feed>
        """

        mock_urlopen.return_value = BytesIO(empty_xml)

        papers = search_arxiv("some nonexistent research topic")

        self.assertEqual(papers, [])

    def test_empty_query_raises_error(self):

        with self.assertRaises(ValueError):
            search_arxiv("")

        with self.assertRaises(ValueError):
            search_arxiv("   ")

    def test_invalid_max_results_raises_error(self):

        with self.assertRaises(ValueError):
            search_arxiv("machine learning", 0)

        with self.assertRaises(ValueError):
            search_arxiv("machine learning", -5)

    @patch("src.researchforge.academic_search.urlopen")
    def test_malformed_xml_raises_error(self, mock_urlopen):

        mock_urlopen.return_value = BytesIO(b"<invalid xml")

        with self.assertRaises(Exception):
            search_arxiv("machine learning")

    # ------------------------------------------
    # Query construction tests
    # ------------------------------------------

    def test_general_query_construction(self):

        result = build_arxiv_query(
            "retrieval augmented generation"
        )

        self.assertEqual(
            result,
            'all:"retrieval" AND all:"augmented" AND all:"generation"'
        )

    def test_exact_phrase_query(self):

        result = build_arxiv_query(
            "retrieval augmented generation",
            exact_phrase=True
        )

        self.assertEqual(
            result,
            'all:"retrieval augmented generation"'
        )

    def test_title_search(self):

        result = build_arxiv_query(
            "transformers",
            field="ti"
        )

        self.assertEqual(
            result,
            'ti:"transformers"'
        )

    def test_abstract_search(self):

        result = build_arxiv_query(
            "diffusion models",
            field="abs"
        )

        self.assertEqual(
            result,
            'abs:"diffusion" AND abs:"models"'
        )

    def test_invalid_search_field(self):

        with self.assertRaises(ValueError):
            build_arxiv_query(
                "transformers",
                field="invalid"
            )

    # ------------------------------------------
    # Search configuration tests
    # ------------------------------------------

    def test_invalid_sort_order(self):

        with self.assertRaises(ValueError):
            search_arxiv(
                "transformers",
                sort_order="random"
            )

    def test_invalid_sort_field(self):

        with self.assertRaises(ValueError):
            search_arxiv(
                "transformers",
                sort_by="invalid"
            )


if __name__ == "__main__":
    unittest.main()
