"""
StockM v1.0 - Phase 10, Lesson 3
Document Loaders
=================

Concrete DocumentLoader implementations (the Stage 1 of the RAG pipeline).
Each loader owns ONE source and normalizes its raw content into our Document
schema. Three loaders ship here:

  1. SampleDataLoader   - bundled synthetic docs, needs no network/API key.
                          Lets the whole RAG pipeline run end-to-end on day one.
  2. RSSFeedLoader      - generic RSS/Atom feed parser (MoneyControl, Reuters,
                          economic times, etc.). The workhorse for live news.
  3. FilingsDirLoader   - reads already-downloaded filings/PRs from a directory
                          (PDF/TXT/HTML), for exchange filings & annual reports.

Why a SampleDataLoader first?
  Professional RAG teams always start with a fixture dataset so the embedding
  -> store -> retrieve -> generate loop is verifiable WITHOUT external deps.
  External loaders then drop in unchanged. This is the same lesson the
  feature-engineering phase taught: never couple your first end-to-end test
  to a live API.

All loaders share a tiny helper module (_loader_utils) for the boring-but-
critical bits: stable doc_id hashing (dedup), trivial text cleanup, and UTC
timestamp normalization. Keeping these out of each loader is DRY; keeping
them private (leading underscore) signals they're not part of the public API.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import RAGSettings, get_yaml_sources
from .contracts import DocumentLoader
from .schemas import Document, DocumentCategory, utc_now

logger = logging.getLogger("stockm.rag.loader")

# ---------------------------------------------------------------------------
# Shared helpers (module-level functions, not a class - they're stateless)
# ---------------------------------------------------------------------------


def make_doc_id(*parts: str) -> str:
    """Stable, deterministic document id from content-defining parts.

    We hash (url + text) or (source + published + title) so that re-ingesting
    the SAME article returns the SAME doc_id. The vector store's add() is
    idempotent on id, so duplicate ingestion becomes an upsert, not a
    duplicate (Lesson 4/13). SHA1 truncated to 16 hex chars: collisions are
    astronomically unlikely for our scale (<10M docs) and the id is short
    enough to log readably.
    """
    raw = "::".join(p.strip() for p in parts if p)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def clean_text(text: str) -> str:
    """Minimal normalization at the loader stage.

    We do NOT do aggressive cleaning here - the chunker (Lesson 4) owns
    chunking-grade cleaning. The loader only fixes things that break doc_id
    stability and parsing: collapsing whitespace, stripping null bytes, and
    trimming. Why minimal? Because over-cleaning at ingest loses information
    (e.g. paragraph breaks that the chunker needs to find boundaries) that
    you can never recover later.
    """
    if not text:
        return ""
    # Drop null/control bytes that sneak in from PDFs and break JSON encoding.
    text = text.replace("\x00", "")
    # Collapse runs of whitespace (including weird unicode spaces) to a single
    # space, but PRESERVE newlines - paragraph structure matters for chunking.
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ newlines (PDF/HTML artifacts) to 2, preserving paragraph gaps.
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def to_utc(dt: datetime | None) -> datetime | None:
    """Normalize any datetime to timezone-aware UTC.

    Feed parsers return naive datetimes (interpreted as the feed's local time,
    which is unknowable), aware datetimes, or strings. We coerce to UTC here
    once so every downstream stage can assume UTC. Naive datetimes are assumed
    UTC (the safest default for international financial feeds); if a source
    is known to be local, the loader converts before calling this.
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def normalize_company(raw: str | None) -> str | None:
    """Normalize a company identifier to uppercase, no exchange suffix.

    'reliance industries' -> 'RELIANCE', 'INFY.NS' -> 'INFY'. This makes
    metadata filtering case- and suffix-insensitive: a query for company=INFY
    matches docs tagged INFY.NS. We keep it simple (first token, uppercased)
    rather than a full ticker resolver - that's a Phase 11 concern.
    """
    if not raw:
        return None
    raw = raw.strip().upper()
    # Strip exchange suffixes (.NS, .BO) so RELIANCE.NS and RELIANCE.BO match.
    raw = raw.split(".")[0]
    # For long company names, take the first significant token (the ticker
    # convention). 'RELIANCE INDUSTRIES' -> 'RELIANCE'. This is a heuristic;
    # loaders that know the exact ticker should pass it directly.
    if " " in raw:
        raw = raw.split()[0]
    return raw or None


# ---------------------------------------------------------------------------
# 1. SampleDataLoader - bundled fixtures, zero external deps
# ---------------------------------------------------------------------------

# A small but realistic set of Indian-market financial docs. These cover every
# DocumentCategory so retrieval/filter testing works out of the box. Real news
# has the same structure; only the text/source changes.
_SAMPLE_DOCS: list[dict[str, Any]] = [
    {
        "source": "moneycontrol",
        "url": "https://www.moneycontrol.com/news/reliance-q2-results-2025",
        "company": "RELIANCE",
        "category": DocumentCategory.EARNINGS_REPORT,
        "published_at": datetime(2025, 10, 15, 9, 30, tzinfo=timezone.utc),
        "text": (
            "Reliance Industries reported Q2 FY26 consolidated net profit of "
            "Rs 18,540 crore, beating street estimates of Rs 17,200 crore. "
            "Revenue from operations rose 14% year-on-year to Rs 2.55 lakh crore. "
            "The oil-to-chemicals segment saw strong margins driven by stable "
            "crude prices. Jio Platforms added 28 million subscribers, with ARPU "
            "expanding to Rs 201 from Rs 181. Retail segment EBITDA grew 19%. "
            "Chairman Mukesh Ambani said the company will accelerate its new "
            "energy giga-factory rollout in FY27. The board declared an interim "
            "dividend of Rs 7 per share."
        ),
    },
    {
        "source": "moneycontrol",
        "url": "https://www.moneycontrol.com/news/infosys-guidance-2025",
        "company": "INFY",
        "category": DocumentCategory.EARNINGS_REPORT,
        "published_at": datetime(2025, 10, 17, 12, 0, tzinfo=timezone.utc),
        "text": (
            "Infosys raised its FY26 revenue growth guidance to 4.5-5.5% from "
            "the earlier 3-4%, citing improving demand environment in North "
            "America. Q2 net profit came in at Rs 6,506 crore, up 11% YoY. "
            "Large deal total contract value reached $2.4 billion. CEO Salil "
            "Parekh said the company is seeing discretionary IT spend return. "
            "Operating margin improved to 21.1% from 20.5% sequentially. The "
            "company announced a special dividend of Rs 20 per share."
        ),
    },
    {
        "source": "bse_india",
        "url": "https://www.bseindia.com/corporates/tcs-buyback-2025",
        "company": "TCS",
        "category": DocumentCategory.COMPANY_ANNOUNCEMENT,
        "published_at": datetime(2025, 10, 10, 14, 0, tzinfo=timezone.utc),
        "text": (
            "Tata Consultancy Services announced a share buyback of Rs 17,000 "
            "crore at Rs 4,150 per share, a 15% premium to the current market "
            "price. The buyback will be via the tender offer route for up to "
            "4.09 crore equity shares. The record date is set for October 25. "
            "Management stated the buyback reflects confidence in the company's "
            "cash generation and long-term outlook."
        ),
    },
    {
        "source": "reuters",
        "url": "https://www.reuters.com/article/rbi-rate-decision-2025",
        "company": None,
        "category": DocumentCategory.ECONOMIC_NEWS,
        "published_at": datetime(2025, 10, 8, 6, 0, tzinfo=timezone.utc),
        "text": (
            "The Reserve Bank of India kept the repo rate unchanged at 6.5% for "
            "the sixth consecutive policy meeting, maintaining its neutral stance. "
            "Governor Sanjay Malhotra cited persistent food inflation and global "
            "uncertainty as reasons for caution. The RBI revised its FY26 GDP "
            "growth forecast down to 6.8% from 7.0%. Markets had priced in a 20% "
            "probability of a rate cut. Bank Nifty fell 0.8% on the announcement."
        ),
    },
    {
        "source": "economic_times",
        "url": "https://economictimes.com/tcs-analyst-downgrade-2025",
        "company": "TCS",
        "category": DocumentCategory.ANALYST_REPORT,
        "published_at": datetime(2025, 10, 18, 8, 0, tzinfo=timezone.utc),
        "text": (
            "Morgan Stanley downgraded Tata Consultancy Services to equal-weight "
            "from overweight, citing near-term valuation concerns and slowing "
            "growth in the BFSI segment. The brokerage cut its target price to "
            "Rs 4,200 from Rs 4,500. However, CLSA maintained a buy rating with "
            "a target of Rs 4,600, arguing the buyback and dividend yield provide "
            "downside support. Nomura highlighted risks from delayed client "
            "decision-making in Europe."
        ),
    },
    {
        "source": "reliance_press",
        "url": "https://www.ril.com/press-releases/new-energy-2025",
        "company": "RELIANCE",
        "category": DocumentCategory.PRESS_RELEASE,
        "published_at": datetime(2025, 10, 20, 11, 0, tzinfo=timezone.utc),
        "text": (
            "Reliance Industries commissioned its first solar giga-factory "
            "module line with 10 GW annual capacity at Jamnagar. The company "
            "plans to scale to 100 GW integrated solar energy capacity by 2030. "
            "The facility will produce high-efficiency PV modules using "
            "indigenous technology. This marks a strategic pivot from the "
            "company's traditional oil-to-chemicals focus toward clean energy, "
            "aligned with India's 500 GW renewable target by 2030."
        ),
    },
    {
        "source": "moneycontrol",
        "url": "https://www.moneycontrol.com/news/infy-institutional-buying-2025",
        "company": "INFY",
        "category": DocumentCategory.FINANCIAL_NEWS,
        "published_at": datetime(2025, 10, 19, 15, 30, tzinfo=timezone.utc),
        "text": (
            "Infosys shares saw heavy institutional buying on the NSE, with "
            "foreign institutional investors net purchasing Rs 1,840 crore worth "
            "of stock. The stock gained 4.2% to close at Rs 1,890. Block deal "
            "data showed Morgan Stanley and Goldman Sachs among the buyers. "
            "The rally followed the guidance raise announced on Thursday. "
            "Derivative data showed call writers unwinding at the 1,900 strike."
        ),
    },
    {
        "source": "bse_india",
        "url": "https://www.bseindia.com/corporates/infy-dividend-2025",
        "company": "INFY",
        "category": DocumentCategory.COMPANY_ANNOUNCEMENT,
        "published_at": datetime(2025, 10, 17, 17, 0, tzinfo=timezone.utc),
        "text": (
            "Infosys Limited informed the exchanges that the board has approved "
            "an interim dividend of Rs 21 per share and a special dividend of "
            "Rs 20 per share. The total dividend payout amounts to Rs 41 per "
            "share. The record date for the dividend is November 2, 2025. "
            "Fixed record date ensures eligible shareholders receive payment."
        ),
    },
]


class SampleDataLoader(DocumentLoader):
    """Zero-dependency loader over the bundled _SAMPLE_DOCS fixtures.

    Use this for development, testing, and the first end-to-end RAG run. It
    exercises every DocumentCategory and proves the pipeline works before any
    external source is wired. Professional teams call this a "golden path"
    fixture - it should never be deleted, only expanded.
    """

    def __init__(self, settings: RAGSettings | None = None) -> None:
        self._settings = settings

    @property
    def source_name(self) -> str:
        return "sample_data"

    def load(self, *args: Any, **kwargs: Any) -> list[Document]:
        docs: list[Document] = []
        for spec in _SAMPLE_DOCS:
            text = clean_text(spec["text"])
            doc_id = make_doc_id(spec["url"], text)
            docs.append(Document(
                doc_id=doc_id,
                text=text,
                source=spec["source"],
                url=spec["url"],
                company=normalize_company(spec.get("company")),
                category=spec["category"],
                published_at=to_utc(spec.get("published_at")),
                language="en",
            ))
        logger.info("SampleDataLoader produced %d documents", len(docs))
        return docs


# ---------------------------------------------------------------------------
# 2. RSSFeedLoader - generic RSS/Atom parser (live news)
# ---------------------------------------------------------------------------


class RSSFeedLoader(DocumentLoader):
    """Parse any RSS/Atom feed into Documents.

    Configured via the 'rss' list in configs/rag_config.yaml, e.g.:
        sources:
          rss:
            - name: moneycontrol_markets
              url: https://www.moneycontrol.com/rss/latest-news.xml
              company: null            # feed-wide company hint, or null
              category: financial_news
            - name: reliance_news
              url: https://example.com/reliance.rss
              company: RELIANCE
              category: financial_news

    Requires the optional 'feedparser' package. If not installed, load()
    raises a clear ImportError - the SampleDataLoader needs no such dep, so
    the pipeline still runs without RSS.

    Design notes:
    - One RSSFeedLoader instance handles ALL feeds in the config (each feed is
      a source-spec, not a separate loader class). This is acceptable because
      RSS feeds share one parsing API (feedparser); they differ only in URL
      and metadata hints, which are data, not behavior. The Open/Closed rule
      is about behavior extensibility, not data variance.
    - published_at is parsed from the feed's entry date and forced to UTC.
    - company can be set per-feed OR per-entry (entry <title>/<description>
      can mention a ticker); we take the feed-level hint as a default and let
      the chunker/generator do finer entity resolution later.
    """

    def __init__(
        self,
        settings: RAGSettings | None = None,
        feeds: list[dict[str, Any]] | None = None,
    ) -> None:
        self._settings = settings
        # Feeds come from YAML by default; tests can inject a list directly.
        if feeds is None:
            yaml_sources = get_yaml_sources()
            feeds = yaml_sources.get("rss", []) if isinstance(yaml_sources, dict) else []
        self._feeds = feeds or []

    @property
    def source_name(self) -> str:
        return "rss"

    def load(self, *args: Any, **kwargs: Any) -> list[Document]:
        if not self._feeds:
            logger.warning("RSSFeedLoader: no feeds configured (check configs/rag_config.yaml)")
            return []

        try:
            import feedparser  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "RSSFeedLoader requires 'feedparser'. Install with: pip install feedparser"
            ) from exc

        docs: list[Document] = []
        for feed_spec in self._feeds:
            url = feed_spec.get("url")
            if not url:
                continue
            feed_source = feed_spec.get("name") or urlparse(url).netloc
            company = normalize_company(feed_spec.get("company"))
            try:
                category = DocumentCategory(feed_spec.get("category", "financial_news"))
            except ValueError:
                category = DocumentCategory.FINANCIAL_NEWS

            try:
                parsed = feedparser.parse(url)
            except Exception as exc:  # noqa: BLE001 - network errors shouldn't kill the batch
                logger.warning("RSS feed %s failed: %s", url, exc)
                continue

            for entry in getattr(parsed, "entries", []):
                title = getattr(entry, "title", "") or ""
                # Prefer 'content' (full body) over 'summary' (truncated).
                content = ""
                if getattr(entry, "content", None):
                    content = entry.content[0].get("value", "")
                if not content:
                    content = getattr(entry, "summary", "") or ""
                text = clean_text(f"{title}. {content}")
                if len(text) < 20:  # skip empty/garbage entries
                    continue
                link = getattr(entry, "link", "") or url
                published = getattr(entry, "published_parsed", None)
                published_dt = None
                if published is not None:
                    try:
                        import time as _time
                        published_dt = datetime.fromtimestamp(
                            _time.mktime(published), tz=timezone.utc
                        )
                    except Exception:  # noqa: BLE001
                        published_dt = None
                doc_id = make_doc_id(link, text)
                docs.append(Document(
                    doc_id=doc_id,
                    text=text,
                    source=feed_source,
                    url=link,
                    company=company,
                    category=category,
                    published_at=to_utc(published_dt),
                    language="en",
                ))
            logger.debug("RSS feed %s: %d entries", feed_source,
                          len(getattr(parsed, "entries", [])))

        logger.info("RSSFeedLoader produced %d documents from %d feeds",
                    len(docs), len(self._feeds))
        return docs


# ---------------------------------------------------------------------------
# 3. FilingsDirLoader - local PDF/TXT/HTML filings
# ---------------------------------------------------------------------------


class FilingsDirLoader(DocumentLoader):
    """Load already-downloaded filings from a directory tree.

    Supports .txt, .md, .html, and .pdf. PDF requires the optional
    'pypdf' package. The directory structure encodes metadata:
        data/documents/filings/<company>/<category>/<file>.pdf
    e.g.  data/documents/filings/RELIANCE/annual_report/AR_2024.pdf

    This loader is for exchange filings (BSE/NSE), annual/quarterly reports,
    and press releases that arrive as files, not feeds. The path-based
    metadata convention means you can drop a filing in the right folder and
    have it indexed on the next run - no config edit needed.

    Why parse PDFs here and not in the chunker? Because the loader's job is
    'bytes -> Document.text', and a PDF is one document (one doc_id). The
    chunker then splits that text. Keeping PDF parsing in the loader keeps the
    chunker format-agnostic (it only ever sees plain text).
    """

    def __init__(
        self,
        settings: RAGSettings | None = None,
        root_dir: Path | str | None = None,
    ) -> None:
        self._settings = settings
        if root_dir is None:
            # Default to data/documents/filings under the project root.
            root_dir = (settings.documents_dir if settings else
                        Path("data/documents")) / "filings"
        self._root = Path(root_dir)

    @property
    def source_name(self) -> str:
        return "filings_dir"

    def load(self, *args: Any, **kwargs: Any) -> list[Document]:
        if not self._root.exists():
            logger.info("FilingsDirLoader: root %s does not exist; skipping", self._root)
            return []

        docs: list[Document] = []
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix not in {".txt", ".md", ".html", ".htm", ".pdf"}:
                continue
            try:
                text = self._extract_text(path, suffix)
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
                logger.warning("Failed to read %s: %s", path, exc)
                continue
            text = clean_text(text)
            if len(text) < self._min_len():
                continue

            # Path encodes company/category: <root>/<company>/<category>/<file>
            company, category = self._metadata_from_path(path)
            # published_at from file mtime as a fallback (filings rarely embed
            # reliable dates in the filename; a real pipeline would parse the
            # document body for the filing date - Lesson 4 enhancement).
            published = to_utc(datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc))
            doc_id = make_doc_id(str(path), text[:1000])
            docs.append(Document(
                doc_id=doc_id,
                text=text,
                source="filings_dir",
                url=None,  # local file; no canonical URL
                company=company,
                category=category,
                published_at=published,
                language="en",
                extra={"path": str(path)},
            ))
        logger.info("FilingsDirLoader produced %d documents from %s", len(docs), self._root)
        return docs

    def _extract_text(self, path: Path, suffix: str) -> str:
        """Dispatch on file extension. Each branch is small and replaceable."""
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix in {".html", ".htm"}:
            raw = path.read_text(encoding="utf-8", errors="replace")
            # Minimal HTML tag strip. We don't add BeautifulSoup as a dep for
            # this; the chunker does heavier cleaning. Good enough for indexing.
            return re.sub(r"<[^>]+>", " ", raw)
        if suffix == ".pdf":
            try:
                from pypdf import PdfReader  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "FilingsDirLoader PDF support requires 'pypdf'. "
                    "Install with: pip install pypdf"
                ) from exc
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        return ""

    def _metadata_from_path(self, path: Path) -> tuple[str | None, DocumentCategory]:
        """Derive company + category from the path relative to root.

        <root>/<company>/<category>/<file> -> (company, category).
        Missing/unknown segments fall back gracefully; we never raise on a
        path that doesn't match the convention - we just tag it UNKNOWN.
        """
        rel = path.relative_to(self._root)
        parts = rel.parts
        company = normalize_company(parts[0]) if len(parts) >= 2 else None
        category = DocumentCategory.UNKNOWN
        if len(parts) >= 3:
            raw_cat = parts[1].lower().replace("-", "_")
            try:
                category = DocumentCategory(raw_cat)
            except ValueError:
                category = DocumentCategory.UNKNOWN
        return company, category

    def _min_len(self) -> int:
        """Minimum text length to index a filing. 50 chars filters empty/garbage."""
        return 50
