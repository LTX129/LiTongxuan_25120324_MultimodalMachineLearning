import logging
import re
from pathlib import Path
from typing import List, Optional

from pypdf import PdfReader

LOGGER = logging.getLogger(__name__)

_REF_PAT = re.compile(r"\b(references|bibliography)\b", re.IGNORECASE)


def _should_stop_at_references(text: str) -> bool:
    return bool(_REF_PAT.search(text or ""))


def extract_text_chunks(
    pdf_path: Path,
    chunk_size: int,
    max_pages: Optional[int] = None,
    stop_at_references: bool = False,
) -> List[str]:
    """
    Extract text then chunk by tokens. Improvements:
    - allow max_pages (for classification)
    - optionally stop when encountering References/Bibliography (for classification)
    """
    reader = PdfReader(str(pdf_path))
    pages = reader.pages
    if max_pages is not None:
        pages = pages[:max_pages]

    buffer: List[str] = []
    for page_idx, page in enumerate(pages):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to read page %s in %s: %s", page_idx, pdf_path, exc)
            text = ""

        if stop_at_references and _should_stop_at_references(text):
            # keep the part before references header (rough but effective)
            parts = _REF_PAT.split(text, maxsplit=1)
            if parts:
                buffer.append(parts[0])
            break

        buffer.append(text)

    full_text = "\n".join(buffer).strip()
    if not full_text:
        return []

    tokens = full_text.split()
    chunks: List[str] = []
    for start in range(0, len(tokens), chunk_size):
        chunk_tokens = tokens[start : start + chunk_size]
        if chunk_tokens:
            chunks.append(" ".join(chunk_tokens))
    return chunks