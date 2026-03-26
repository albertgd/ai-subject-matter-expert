"""
Tests for scrapers — offline/unit tests only (no real HTTP calls).
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


class TestBaseScraper:
    def setup_method(self):
        from src.scrapers.base_scraper import BaseScraper

        class ConcreteScraper(BaseScraper):
            SOURCE_NAME = "test_source"

            def scrape(self, max_cases=10):
                return []

        with tempfile.TemporaryDirectory() as tmpdir:
            self.tmpdir = Path(tmpdir)

        self.ScraperClass = ConcreteScraper

    def test_save_and_load_case(self):
        from src.scrapers.base_scraper import BaseScraper

        class ConcreteScraper(BaseScraper):
            SOURCE_NAME = "test_source"

            def scrape(self, max_cases=10):
                return []

        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = ConcreteScraper(output_dir=Path(tmpdir))

            case = {
                "source_id": "test_001",
                "source_name": "TestSource",
                "title": "Smith v. Jones",
                "url": "https://example.com/cases/1",
                "date": "2023-01-15",
                "court": "District Court",
                "text": "The court held that custody should be shared. " * 20,
                "metadata": {"opinion_id": "001"},
            }

            path = scraper.save_case(case)
            assert path.exists()

            loaded = scraper.load_saved()
            assert len(loaded) == 1
            assert loaded[0]["source_id"] == "test_001"
            assert loaded[0]["title"] == "Smith v. Jones"

    def test_already_scraped(self):
        from src.scrapers.base_scraper import BaseScraper

        class ConcreteScraper(BaseScraper):
            SOURCE_NAME = "test_source2"

            def scrape(self, max_cases=10):
                return []

        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = ConcreteScraper(output_dir=Path(tmpdir))

            assert not scraper.already_scraped("case_123")

            case = {
                "source_id": "case_123",
                "source_name": "Test",
                "title": "Test Case",
                "url": "",
                "date": "",
                "court": "",
                "text": "Some text.",
                "metadata": {},
            }
            scraper.save_case(case)
            assert scraper.already_scraped("case_123")

    def test_save_all(self):
        from src.scrapers.base_scraper import BaseScraper

        class ConcreteScraper(BaseScraper):
            SOURCE_NAME = "test_source3"

            def scrape(self, max_cases=10):
                return []

        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = ConcreteScraper(output_dir=Path(tmpdir))
            cases = [
                {
                    "source_id": f"case_{i}",
                    "source_name": "Test",
                    "title": f"Case {i}",
                    "url": f"https://example.com/{i}",
                    "date": "2023-01-01",
                    "court": "Test Court",
                    "text": f"Content of case {i}. " * 20,
                    "metadata": {},
                }
                for i in range(5)
            ]
            saved = scraper.save_all(cases)
            assert saved == 5

            loaded = scraper.load_saved()
            assert len(loaded) == 5


class TestCourtListenerScraper:
    def test_strip_html(self):
        from src.scrapers.courtlistener_scraper import CourtListenerScraper
        html = "<p>The court <strong>held</strong> that custody should be shared.</p>"
        text = CourtListenerScraper._strip_html(html)
        assert "<p>" not in text
        assert "<strong>" not in text
        assert "held" in text
        assert "custody" in text

    def test_build_case_from_hit_with_empty_text(self):
        """Case with empty snippet and no fetchable text should return None."""
        from src.scrapers.courtlistener_scraper import CourtListenerScraper

        with tempfile.TemporaryDirectory() as tmpdir:
            scraper = CourtListenerScraper(output_dir=Path(tmpdir))

            # Mock _fetch_opinion_text to return empty
            scraper._fetch_opinion_text = lambda oid: ""

            hit = {
                "id": "99999",
                "caseName": "Test v. Case",
                "absolute_url": "/opinion/99999/test-v-case/",
                "dateFiled": "2023-01-01",
                "court": "test-court",
                "snippet": "",  # Empty snippet
            }

            result = scraper._build_case_from_hit(hit)
            assert result is None  # No text → should return None


class TestWebScraper:
    def test_url_to_id(self):
        from src.scrapers.web_scraper import WebScraper
        url = "https://law.justia.com/cases/federal/district-courts/FSupp/123/456"
        id_ = WebScraper._url_to_id(url)
        assert len(id_) <= 100
        assert "/" not in id_

    def test_is_justia_case_url(self):
        from src.scrapers.web_scraper import WebScraper
        assert WebScraper._is_justia_case_url("https://law.justia.com/cases/california/2023/case.html")
        assert not WebScraper._is_justia_case_url("https://google.com")
        assert not WebScraper._is_justia_case_url("https://law.justia.com/topics/")

    def test_is_bailii_case_url(self):
        from src.scrapers.web_scraper import WebScraper
        assert WebScraper._is_bailii_case_url("/ew/cases/EWCA/Civ/2023/123.html")
        assert not WebScraper._is_bailii_case_url("/cgi-bin/search.pl")
        assert not WebScraper._is_bailii_case_url("/about.html")

    def test_extract_bailii_court(self):
        from src.scrapers.web_scraper import WebScraper
        url = "https://www.bailii.org/ew/cases/EWCA/Civ/2023/123.html"
        court = WebScraper._extract_bailii_court(url)
        assert "Court of Appeal" in court
