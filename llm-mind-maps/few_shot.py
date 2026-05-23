import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from data_processing import DatasetItem
from evaluation_pipeline import (
    get_parent_tag,
    parse_mindmap_tags,
    read_text,
    sort_mindmap_tags,
)


@dataclass
class FewShotExample:
    """One few-shot example with source sentences and target mind-map JSON."""
    article_id: str
    sentences: List[str]
    target: Dict[str, Any]


def normalize_for_matching(text: str) -> str:
    """Normalize text for approximate reference-to-source matching."""
    return " ".join(text.lower().strip().split())


def find_best_sentence_id(
    reference_text: str,
    sentences: List[str],
    min_similarity: float = 0.86,
) -> Optional[int]:
    """Find the best matching source sentence ID for a reference node."""
    reference_norm = normalize_for_matching(reference_text)

    best_idx = None
    best_score = 0.0

    for idx, sentence in enumerate(sentences):
        sentence_norm = normalize_for_matching(sentence)

        if reference_norm == sentence_norm:
            return idx

        score = SequenceMatcher(None, reference_norm, sentence_norm).ratio()

        if score > best_score:
            best_score = score
            best_idx = idx

    if best_score >= min_similarity:
        return best_idx

    return None


def reference_to_target_json(item: DatasetItem) -> Optional[Dict[str, Any]]:
    """
    Convert a tagged reference mind map into sentence-ID JSON.

    Unmatched reference nodes are skipped. If a skipped node has matched
    descendants, those descendants are attached to the nearest matched ancestor.
    """
    text = read_text(item.reference_path)
    tag_to_text = parse_mindmap_tags(text)

    if not tag_to_text:
        return None

    sorted_tags = sort_mindmap_tags(list(tag_to_text.keys()))

    tag_to_id = {}
    for tag in sorted_tags:
        sentence_id = find_best_sentence_id(tag_to_text[tag], item.sentences)
        if sentence_id is not None:
            tag_to_id[tag] = sentence_id

    if len(set(tag_to_id.values())) < 2:
        return None

    selected = []
    for tag in sorted_tags:
        sentence_id = tag_to_id.get(tag)
        if sentence_id is not None and sentence_id not in selected:
            selected.append(sentence_id)

    def nearest_matched_parent(tag: str) -> Optional[int]:
        parent_tag = get_parent_tag(tag)

        while parent_tag is not None:
            if parent_tag in tag_to_id:
                return tag_to_id[parent_tag]

            parent_tag = get_parent_tag(parent_tag)

        return None

    edges = []
    child_has_parent = set()

    for tag in sorted_tags:
        child_id = tag_to_id.get(tag)

        if child_id is None:
            continue

        parent_id = nearest_matched_parent(tag)

        if parent_id is None:
            continue

        if parent_id == child_id:
            continue

        if child_id in child_has_parent:
            continue

        edges.append([parent_id, child_id])
        child_has_parent.add(child_id)

    possible_roots = [node for node in selected if node not in child_has_parent]

    if possible_roots:
        root = possible_roots[0]
    else:
        root = selected[0]

    clean_edges = []
    clean_children = set()

    for parent, child in edges:
        if child == root:
            continue

        if child in clean_children:
            continue

        clean_edges.append([parent, child])
        clean_children.add(child)

    for node in selected:
        if node == root:
            continue

        if node not in clean_children:
            clean_edges.append([root, node])
            clean_children.add(node)

    selected = sorted(set([root] + [node for edge in clean_edges for node in edge]))

    return {
        "root": root,
        "selected": selected,
        "edges": clean_edges,
    }


def build_few_shot_examples(
    dev_items: List[DatasetItem],
    count: int,
    exclude_article_ids: Optional[Set[str]] = None,
) -> List[FewShotExample]:
    """Build up to count few-shot examples from the development set."""
    if count < 0 or count > 3:
        raise ValueError("Few-shot count must be between 0 and 3.")
    
    if count == 0:
        return []

    exclude_article_ids = exclude_article_ids or set()
    examples = []

    for item in dev_items:
        if item.article_id in exclude_article_ids:
            continue

        target = reference_to_target_json(item)

        if target is None:
            continue

        examples.append(
            FewShotExample(
                article_id=item.article_id,
                sentences=item.sentences,
                target=target,
            )
        )

        if len(examples) == count:
            break

    return examples


def few_shot_target_to_text(target: Dict[str, Any]) -> str:
    """Serialize few-shot target JSON compactly."""
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def target_json_to_hierarchical_list(
    target: Dict[str, Any],
    sentences: List[str],
) -> str:
    """Convert target JSON mind map into hierarchical numbered-list format."""
    root = target["root"]
    edges = [tuple(edge) for edge in target["edges"]]

    children: Dict[int, List[int]] = {}

    for parent, child in edges:
        children.setdefault(parent, []).append(child)

    for parent in children:
        children[parent].sort()

    lines = [f"[{root}] {sentences[root]}"]

    def dfs(node: int, prefix: str) -> None:
        for index, child in enumerate(children.get(node, []), start=1):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            lines.append(f"{child_prefix}. [{child}] {sentences[child]}")
            dfs(child, child_prefix)

    dfs(root, "")

    return "\n".join(lines)