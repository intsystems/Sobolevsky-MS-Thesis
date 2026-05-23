import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class DatasetItem:
    """Container for one article and its corresponding reference file."""
    article_id: str
    original_path: Path
    reference_path: Path
    sentences: List[str]


@dataclass
class DatasetSplit:
    """Container for a dataset split with original documents and labeled references."""
    name: str
    original_dir: Path
    reference_dir: Path


def read_text(path: str | Path) -> str:
    """Read a text file with UTF-8 fallback error handling."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def extract_numbered_sentences(text: str) -> Optional[List[str]]:
    """Extract sentences from already-numbered text if possible."""
    pattern = re.compile(
        r"(?:^|\n)\s*(\d+)\s*[\.\)]\s+(.*?)(?=(?:\n\s*\d+\s*[\.\)]\s+)|\Z)",
        flags=re.S,
    )
    matches = pattern.findall(text)

    if len(matches) < 2:
        return None

    indexed = []
    for idx, sent in matches:
        sent = re.sub(r"\s+", " ", sent).strip()
        if sent:
            indexed.append((int(idx), sent))

    indexed.sort(key=lambda x: x[0])
    return [sent for _, sent in indexed]


def sent_tokenize_fallback(text: str) -> List[str]:
    """Split text into sentences using NLTK if available, otherwise regex fallback."""
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        return [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
    except Exception:
        return [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
            if s.strip()
        ]
    

def split_story_source_and_highlights(text: str) -> str:
    """Return the source-document part before the first @highlight marker."""
    parts = re.split(r"\n\s*@highlight\s*\n", text, maxsplit=1, flags=re.I)
    return parts[0] if parts else text


def read_document_sentences(path: str | Path) -> List[str]:
    """
    Read a .story source document as line-based sentence/paragraph units.

    The MindMap dataset source article usually stores each sentence or heading
    on a separate non-empty line before the @highlight section. We preserve this
    structure instead of applying automatic sentence tokenization, because regex
    or NLTK splitting can break abbreviations such as U.S., No. 1, Dr., etc.
    """
    text = read_text(path)
    source_text = split_story_source_and_highlights(text)

    lines = []

    for raw_line in source_text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()

        if not line:
            continue

        lines.append(line)

    return lines


def get_dataset_splits(data_dir: str | Path = "data") -> Dict[str, DatasetSplit]:
    """Return the predefined dev/test dataset split configuration."""
    data_dir = Path(data_dir)

    return {
        "dev": DatasetSplit(
            name="dev",
            original_dir=data_dir / "dev_full_original",
            reference_dir=data_dir / "a_labeling_dev",
        ),
        "test": DatasetSplit(
            name="test",
            original_dir=data_dir / "test_full_original",
            reference_dir=data_dir / "a_labeling_test",
        ),
    }


def find_text_files(directory: str | Path) -> List[Path]:
    """Find story/text files inside a directory."""
    directory = Path(directory)

    files = sorted(directory.rglob("*.story"))
    if files:
        return files

    return sorted(directory.rglob("*.txt"))


def build_reference_index(reference_dir: str | Path) -> Dict[str, Path]:
    """Build a mapping from file stem to reference annotation path."""
    reference_dir = Path(reference_dir)
    files = sorted(reference_dir.rglob("*"))

    return {
        path.stem: path
        for path in files
        if path.is_file() and not path.name.startswith(".")
    }


def load_dataset_split(split: DatasetSplit) -> List[DatasetItem]:
    """Load one dataset split and match original articles with reference files."""
    original_files = find_text_files(split.original_dir)
    reference_index = build_reference_index(split.reference_dir)

    items = []

    for original_path in original_files:
        article_id = original_path.stem
        reference_path = reference_index.get(article_id)

        if reference_path is None:
            continue

        sentences = read_document_sentences(original_path)

        if not sentences:
            continue

        items.append(
            DatasetItem(
                article_id=article_id,
                original_path=original_path,
                reference_path=reference_path,
                sentences=sentences,
            )
        )

    return items


def format_numbered_document(sentences: List[str]) -> str:
    """Format sentences as an indexed document for prompting."""
    return "\n".join(f"{idx}. {sent}" for idx, sent in enumerate(sentences))