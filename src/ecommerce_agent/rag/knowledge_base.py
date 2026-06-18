"""Load local after-sales knowledge documents."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"


@dataclass(frozen=True)
class KnowledgeDocument:
    doc_id: str
    title: str
    source: str
    content: str
    keywords: tuple[str, ...]


def _extract_keywords(text: str) -> tuple[str, ...]:
    keywords: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("适用"):
            continue
        _, _, tail = stripped.partition("：")
        keywords.extend(word.strip() for word in tail.replace("、", "，").replace("。", "，").split("，"))
    return tuple(word for word in keywords if word)


def _split_markdown_sections(path: Path) -> list[KnowledgeDocument]:
    text = path.read_text(encoding="utf-8")
    sections: list[KnowledgeDocument] = []
    current_title = ""
    current_lines: list[str] = []

    def flush() -> None:
        if not current_title or not current_lines:
            return
        doc_id = current_title.split(maxsplit=1)[0]
        content = "\n".join(current_lines).strip()
        sections.append(
            KnowledgeDocument(
                doc_id=doc_id,
                title=current_title,
                source=path.name,
                content=content,
                keywords=_extract_keywords(content),
            )
        )

    for line in text.splitlines():
        if line.startswith("## "):
            flush()
            current_title = line[3:].strip()
            current_lines = [line]
        elif current_title:
            current_lines.append(line)
    flush()
    return sections


@lru_cache(maxsize=1)
def load_knowledge_base() -> tuple[KnowledgeDocument, ...]:
    docs: list[KnowledgeDocument] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        docs.extend(_split_markdown_sections(path))
    return tuple(docs)

