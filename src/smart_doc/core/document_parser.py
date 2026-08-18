"""Enhanced document parsing with structure detection and metadata extraction."""

import re
from dataclasses import dataclass, field
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader


@dataclass
class DocumentMetadata:
    """Metadata extracted from a document."""
    source: str = ""
    page_count: int = 0
    total_chars: int = 0
    total_words: int = 0
    language: str = "unknown"
    sections: list = field(default_factory=list)
    has_tables: bool = False
    has_figures: bool = False
    title: str = ""


@dataclass
class ParsedChunk:
    """A parsed chunk with enhanced metadata."""
    content: str
    metadata: dict = field(default_factory=dict)
    section_title: str = ""
    chunk_index: int = 0
    is_heading: bool = False
    semantic_type: str = "paragraph"  # paragraph, heading, list, table, code


# ─── Language Detection ───────────────────────────────────────────────────────

def _detect_language(text: str) -> str:
    """Simple language detection based on character patterns."""
    # Indonesian markers
    id_markers = ["yang", "dan", "ini", "itu", "dengan", "untuk", "pada", "adalah",
                  "akan", "tidak", "dari", "dalam", "oleh", "atau", "juga"]
    # English markers
    en_markers = ["the", "and", "is", "this", "that", "with", "for", "are",
                  "was", "not", "from", "in", "by", "or", "also"]

    text_lower = text.lower()
    words = text_lower.split()

    id_count = sum(1 for w in words if w in id_markers)
    en_count = sum(1 for w in words if w in en_markers)

    if id_count > en_count:
        return "id"
    elif en_count > id_count:
        return "en"
    return "unknown"


# ─── Section Detection ────────────────────────────────────────────────────────

HEADING_PATTERNS = [
    re.compile(r"^(PASAL|BAB|BAGIAN|SECTION|CHAPTER|ARTICLE)\s+\d+", re.IGNORECASE),
    re.compile(r"^(UNDANG-UNDANG|PERATURAN|KEPUTusan|REGULATION)\s+", re.IGNORECASE),
    re.compile(r"^\d+[\.\)]\s+[A-Z][A-Za-z\s]+"),  # "1. Introduction"
    re.compile(r"^#{1,6}\s+"),  # Markdown headings
    re.compile(r"^[A-Z][A-Z\s]{5,}$"),  # ALL CAPS lines (likely headings)
]

LIST_PATTERN = re.compile(r"^[\-\*\•\d+\.]\s+")
TABLE_PATTERN = re.compile(r"\|.*\|.*\|")
CODE_PATTERN = re.compile(r"^(def |class |import |from |if |for |while )")


def _detect_semantic_type(line: str) -> str:
    """Detect the semantic type of a line of text."""
    line_stripped = line.strip()

    if not line_stripped:
        return "empty"

    for pattern in HEADING_PATTERNS:
        if pattern.search(line_stripped):
            return "heading"

    if LIST_PATTERN.match(line_stripped):
        return "list"

    if TABLE_PATTERN.search(line_stripped):
        return "table"

    if CODE_PATTERN.match(line_stripped):
        return "code"

    return "paragraph"


def _extract_sections(text: str) -> list[str]:
    """Extract section titles from text."""
    sections = []
    for line in text.split("\n"):
        if _detect_semantic_type(line) == "heading":
            sections.append(line.strip())
    return sections


# ─── Document Parsing ─────────────────────────────────────────────────────────

def parse_document(file_path: str) -> tuple[list, DocumentMetadata]:
    """Parse a document and extract content with enhanced metadata.

    Args:
        file_path: Path to the document file.

    Returns:
        Tuple of (list of LangChain documents, DocumentMetadata).
    """
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    documents = loader.load()

    # Build metadata
    all_text = "\n".join([doc.page_content for doc in documents])
    metadata = DocumentMetadata(
        source=file_path,
        page_count=len(documents),
        total_chars=len(all_text),
        total_words=len(all_text.split()),
        language=_detect_language(all_text),
        sections=_extract_sections(all_text),
        has_tables=bool(TABLE_PATTERN.search(all_text)),
        title=documents[0].metadata.get("title", "") if documents else "",
    )

    return documents, metadata
