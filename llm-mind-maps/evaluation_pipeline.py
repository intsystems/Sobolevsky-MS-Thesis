import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from rouge import Rouge


@dataclass
class RougeScores:
    """ROUGE-1, ROUGE-2, ROUGE-L and their mean."""
    rouge_1: float
    rouge_2: float
    rouge_l: float
    rouge_mean: float


@dataclass
class EvaluationResult:
    """Evaluation result for one generated mind map."""
    article_id: str
    num_reference_pairs: int
    scores: RougeScores


def read_text(path: str | Path) -> str:
    """Read a text file with UTF-8 fallback error handling."""
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    """Normalize whitespace in text."""
    return re.sub(r"\s+", " ", text).strip()


def rouge_similarity(text_a: str, text_b: str, rouge_type: str) -> float:
    """Compute one ROUGE F-score type between two text fragments."""
    text_a = normalize_text(text_a)
    text_b = normalize_text(text_b)

    if not text_a and not text_b:
        return 1.0

    if not text_a or not text_b:
        return 0.0

    rouge = Rouge()

    try:
        scores = rouge.get_scores(text_a, text_b)[0]
        return float(scores[rouge_type]["f"])
    except Exception:
        return 0.0


def rouge_1_similarity(text_a: str, text_b: str) -> float:
    """Compute ROUGE-1 similarity for two text fragments."""
    return rouge_similarity(text_a, text_b, "rouge-1")


def rouge_2_similarity(text_a: str, text_b: str) -> float:
    """Compute ROUGE-2 similarity for two text fragments."""
    return rouge_similarity(text_a, text_b, "rouge-2")


def rouge_l_similarity(text_a: str, text_b: str) -> float:
    """Compute ROUGE-L similarity for two text fragments."""
    return rouge_similarity(text_a, text_b, "rouge-l")


def mean_rouge_similarity(text_a: str, text_b: str) -> float:
    """Compute the mean of ROUGE-1, ROUGE-2 and ROUGE-L similarities."""
    return (
        rouge_1_similarity(text_a, text_b)
        + rouge_2_similarity(text_a, text_b)
        + rouge_l_similarity(text_a, text_b)
    ) / 3.0


def score_pair(
    generated_pair: List[str],
    reference_pair: List[str],
    similarity_fn: Callable[[str, str], float],
) -> float:
    """Score one generated parent-child pair against one reference pair."""
    parent_score = similarity_fn(reference_pair[0], generated_pair[0])
    child_score = similarity_fn(reference_pair[1], generated_pair[1])
    return (parent_score + child_score) / 2.0


def greedy_pair_matching_score(
    generated_pairs: List[List[str]],
    reference_pairs: List[List[str]],
    similarity_fn: Callable[[str, str], float],
) -> float:
    """Compute CMGN-style greedy matching score for one similarity function."""
    available_pairs = [pair[:] for pair in generated_pairs]

    if len(available_pairs) < 2 or len(reference_pairs) < 2:
        return 0.0

    total_score = 0.0

    for reference_pair in reference_pairs[1:]:
        best_score = -1.0
        best_index = None

        for index, generated_pair in enumerate(available_pairs[1:], start=1):
            current_score = score_pair(
                generated_pair=generated_pair,
                reference_pair=reference_pair,
                similarity_fn=similarity_fn,
            )

            if current_score > best_score:
                best_score = current_score
                best_index = index

        if best_index is not None:
            total_score += best_score
            del available_pairs[best_index]

    return total_score / len(reference_pairs)


def evaluate_pairs(
    generated_pairs: List[List[str]],
    reference_pairs: List[List[str]],
) -> RougeScores:
    """Evaluate generated mind-map pairs against reference pairs."""
    rouge_1 = greedy_pair_matching_score(
        generated_pairs=generated_pairs,
        reference_pairs=reference_pairs,
        similarity_fn=rouge_1_similarity,
    )

    rouge_2 = greedy_pair_matching_score(
        generated_pairs=generated_pairs,
        reference_pairs=reference_pairs,
        similarity_fn=rouge_2_similarity,
    )

    rouge_l = greedy_pair_matching_score(
        generated_pairs=generated_pairs,
        reference_pairs=reference_pairs,
        similarity_fn=rouge_l_similarity,
    )

    rouge_mean = greedy_pair_matching_score(
        generated_pairs=generated_pairs,
        reference_pairs=reference_pairs,
        similarity_fn=mean_rouge_similarity,
    )

    return RougeScores(
        rouge_1=rouge_1,
        rouge_2=rouge_2,
        rouge_l=rouge_l,
        rouge_mean=rouge_mean,
    )


def extract_highlight_block(text: str) -> str:
    """Extract the mind-map block inside <highlight> tags if present."""
    match = re.search(
        r"<highlight>\s*(.*?)\s*</highlight>",
        text,
        flags=re.S | re.I,
    )

    if match:
        return match.group(1)

    return text


def normalize_tag_text(text: str) -> str:
    """Clean text extracted from a mind-map tag."""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&quot;", '"')
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = re.sub(r"\s+", " ", text)
    return text.strip().strip('"').strip()


def parse_mindmap_tags(text: str) -> dict[str, str]:
    """
    Parse tagged mind-map nodes from <T...>...</T...> blocks.

    The parser is intentionally tolerant because some dataset files contain
    malformed closing tags.
    """
    text = extract_highlight_block(text)

    pattern = re.compile(
        r"<(?P<tag>T[\d.]+)>\s*(?P<body>.*?)(?=<T[\d.]+>|</highlight>|\Z)",
        flags=re.S | re.I,
    )

    nodes = {}

    for match in pattern.finditer(text):
        tag = match.group("tag").strip()
        body = match.group("body")

        closing_tag_pattern = rf"</?\s*{re.escape(tag)}\s*>"
        body = re.sub(closing_tag_pattern, " ", body, flags=re.I)

        value = normalize_tag_text(body)

        if value:
            nodes[tag] = value

    return nodes


def get_parent_tag(tag: str) -> str | None:
    """Return the parent tag according to dotted mind-map numbering."""
    if "." not in tag:
        return None

    return tag.rsplit(".", 1)[0]


def sort_mindmap_tags(tags: list[str]) -> list[str]:
    """Sort mind-map tags by their numeric hierarchy order."""
    def key(tag: str) -> list[int]:
        return [int(part) for part in tag[1:].split(".") if part.isdigit()]

    return sorted(tags, key=key)


def parse_reference_pairs(reference_path: str | Path) -> List[List[str]]:
    """
    Parse a .story reference mind map into parent-child sentence pairs.

    The expected format is a <highlight> block containing tags such as:
    <T1>root</T1>
    <T1.1>child</T1.1>
    <T1.1.1>grandchild</T1.1.1>
    """
    text = read_text(reference_path)
    nodes = parse_mindmap_tags(text)

    if not nodes:
        raise ValueError(
            f"Could not parse any <T...> mind-map nodes from {reference_path}."
        )

    sorted_tags = sort_mindmap_tags(list(nodes.keys()))

    root_tag = sorted_tags[0]
    root_text = nodes[root_tag]

    pairs = [["", root_text]]

    for tag in sorted_tags:
        if tag == root_tag:
            continue

        parent_tag = get_parent_tag(tag)

        if parent_tag is None:
            continue

        parent_text = nodes.get(parent_tag)
        child_text = nodes.get(tag)

        if parent_text and child_text:
            pairs.append([parent_text, child_text])

    return pairs


def evaluate_generated_mindmap(
    article_id: str,
    generated_pairs: List[List[str]],
    reference_path: str | Path,
) -> EvaluationResult:
    """Evaluate one generated mind map against one reference annotation file."""
    reference_pairs = parse_reference_pairs(reference_path)

    scores = evaluate_pairs(
        generated_pairs=generated_pairs,
        reference_pairs=reference_pairs,
    )

    return EvaluationResult(
        article_id=article_id,
        num_reference_pairs=len(reference_pairs),
        scores=scores,
    )


def aggregate_results(results: List[EvaluationResult]) -> Dict[str, float]:
    """Average evaluation scores over multiple articles."""
    if not results:
        return {
            "rouge_1": 0.0,
            "rouge_2": 0.0,
            "rouge_l": 0.0,
            "rouge_mean": 0.0,
        }

    return {
        "rouge_1": sum(r.scores.rouge_1 for r in results) / len(results),
        "rouge_2": sum(r.scores.rouge_2 for r in results) / len(results),
        "rouge_l": sum(r.scores.rouge_l for r in results) / len(results),
        "rouge_mean": sum(r.scores.rouge_mean for r in results) / len(results),
    }