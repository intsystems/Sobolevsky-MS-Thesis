import ast
import json
import re
import gc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from data_processing import format_numbered_document
from few_shot import FewShotExample, target_json_to_hierarchical_list


@dataclass
class MindMap:
    """Sentence-based rooted mind-map representation."""
    root: int
    edges: List[Tuple[int, int]]
    selected: List[int]


def cleanup_memory() -> None:
    """Release Python and CUDA memory between generations."""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


class LocalLLMMindMapGenerator:
    """Local LLM pipeline for generating sentence-based mind maps."""
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        max_nodes: int | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        few_shot_examples: Optional[List[FewShotExample]] = None,
        quantization: str = "none",
    ):
        """Initialize tokenizer, model, generation settings and few-shot examples."""
        if quantization not in {"none", "4bit", "8bit"}:
            raise ValueError(
                f"quantization must be one of 'none', '4bit', '8bit', got {quantization!r}"
            )

        self.model_name = model_name
        self.max_nodes = max_nodes
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.few_shot_examples = few_shot_examples or []
        self.quantization = quantization

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **self.build_model_loading_kwargs(),
        )

        self.model.eval()

    def build_model_loading_kwargs(self) -> Dict[str, Any]:
        """Build model loading kwargs according to the selected quantization mode."""
        kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
        }

        if self.quantization == "none":
            kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            return kwargs

        if self.quantization == "4bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                bnb_4bit_use_double_quant=True,
            )
            return kwargs

        if self.quantization == "8bit":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            kwargs["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            return kwargs

    def get_effective_max_new_tokens(self, n_sentences: int) -> int:
        """Choose a safe generation budget based on document size and shot count."""
        adaptive_limit = max(1024, 128 * n_sentences)

        return min(self.max_new_tokens, adaptive_limit)
    
    def build_system_prompt(self) -> str:
        """Build concise shared instructions for hierarchical-list mind-map generation."""
        max_nodes_text = ""
        if self.max_nodes is not None:
            max_nodes_text = f"\n- Use at most {self.max_nodes} selected nodes."

        return f"""
You generate sentence-based mind maps for news articles.

Task:
Given a numbered article, choose important sentence IDs and organize them into a hierarchical numbered list.

A good mind map:
- has one root sentence that represents the main topic;
- has several main branches for different topics or sections;
- places details under the branch they explain;
- uses parent -> child relations where the child gives more specific information about the parent.

Output format:
- Output only a hierarchical numbered list.
- The first line must be the root node.
- Every line must contain exactly one source sentence ID in square brackets.
- Copy the sentence text after the ID.
- Do not output JSON.
- Do not output explanations or markdown bullets.

Required format:
[0] Root sentence text
1. [3] First major branch sentence text
1.1. [4] Detail sentence text
1.2. [5] Another detail sentence text
2. [10] Second major branch sentence text
2.1. [11] Detail sentence text

Meaning:
- The first line is the root.
- Lines like "1.", "2.", "3." are children of the root.
- Lines like "1.1.", "1.2." are children of "1.".
- Lines like "1.2.1." are children of "1.2.".

Rules:
- Use only sentence IDs from the article.
- Each selected sentence ID should appear only once.
- The hierarchy must be connected.
- Do not include sentence IDs that are not in the article.
- Do not invent or rewrite sentences.{max_nodes_text}
""".strip()
    
    def build_document_prompt(
        self,
        sentences: List[str],
        is_few_shot_example: bool = False,
    ) -> str:
        """Build the document-specific user prompt."""
        numbered_doc = format_numbered_document(sentences)

        label = "Example article" if is_few_shot_example else "Article"

        return f"""
{label}:
{numbered_doc}

Generate the hierarchical mind-map list for this article.
""".strip()
    
    def build_feedback_prompt(self, feedback: str) -> str:
        """Build a concise retry prompt from validation feedback."""
        return f"""
The previous hierarchical list was invalid.

Fix these issues:
{feedback}

Return a new hierarchical numbered list only.

Use this format:
[0] Root sentence text
1. [3] Child sentence text
1.1. [4] Detail sentence text

Do not output JSON, explanations, or markdown bullets.
""".strip()

    def build_prompt(self, sentences: List[str]) -> List[Dict[str, str]]:
        """Build the initial chat prompt with shared instructions and few-shot examples."""
        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(),
            }
        ]

        for example in self.few_shot_examples:
            messages.append(
                {
                    "role": "user",
                    "content": self.build_document_prompt(
                        sentences=example.sentences,
                        is_few_shot_example=True,
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": target_json_to_hierarchical_list(
                        target=example.target,
                        sentences=example.sentences,
                    ),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": self.build_document_prompt(
                    sentences=sentences,
                    is_few_shot_example=False,
                ),
            }
        )

        return messages

    @torch.inference_mode()
    def generate_raw(self, messages: List[Dict[str, str]], n_sentences: int) -> str:
        """Generate raw model output from prepared chat messages and free temporary memory."""
        cleanup_memory()

        inputs = None
        output_ids = None
        generated = None

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            prompt_len = inputs["input_ids"].shape[-1]

            generation_kwargs = {
                **inputs,
                "max_new_tokens": self.get_effective_max_new_tokens(n_sentences),
                "do_sample": self.temperature > 0,
                "repetition_penalty": 1.05,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            if self.temperature > 0:
                generation_kwargs["temperature"] = self.temperature
                generation_kwargs["top_p"] = 0.9

            if hasattr(self, "use_cache"):
                generation_kwargs["use_cache"] = self.use_cache

            output_ids = self.model.generate(**generation_kwargs)

            generated = output_ids[0][prompt_len:]
            decoded = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

            return decoded

        finally:
            del inputs
            del output_ids
            del generated

            if "generation_kwargs" in locals():
                del generation_kwargs

            if "prompt" in locals():
                del prompt

            cleanup_memory()

    def generate(self, sentences: List[str]) -> Dict[str, Any]:
        """Generate, parse, validate and render a mind map for one article."""
        messages = self.build_prompt(sentences)
        attempts = []
        raw = ""

        for _ in range(3):
            raw = self.generate_raw(messages, n_sentences=len(sentences))
            attempts.append(raw[:4000])

            try:
                mindmap = parse_hierarchical_list_output(
                    raw=raw,
                    n_sentences=len(sentences),
                )
            except Exception:
                feedback = (
                    "The output could not be parsed as a hierarchical numbered list. "
                    "Use lines like '[0] root sentence' and '1. [3] child sentence'."
                )

                messages.append({"role": "assistant", "content": raw[:1000]})
                messages.append({"role": "user", "content": self.build_feedback_prompt(feedback)})
                continue

            validation_errors = validate_mindmap_object(
                mindmap=mindmap,
                n_sentences=len(sentences),
                max_nodes=self.max_nodes,
            )

            if validation_errors:
                feedback = "\n".join(f"- {error}" for error in validation_errors[:8])
                messages.append({"role": "user", "content": self.build_feedback_prompt(feedback)})
                continue

            return {
                "raw_output": raw,
                "attempts": attempts,
                "mindmap": {
                    "root": mindmap.root,
                    "selected": mindmap.selected,
                    "edges": mindmap.edges,
                },
                "rendered": render_mindmap(mindmap, sentences),
                "pairs": mindmap_to_sentence_pairs(mindmap, sentences),
                "generation_status": "ok",
            }

        try:
            mindmap = parse_hierarchical_list_output(
                raw=raw,
                n_sentences=len(sentences),
            )
        except Exception:
            mindmap = MindMap(root=0, selected=[0], edges=[])

        return {
            "raw_output": raw,
            "attempts": attempts,
            "mindmap": {
                "root": mindmap.root,
                "selected": mindmap.selected,
                "edges": mindmap.edges,
            },
            "rendered": render_mindmap(mindmap, sentences),
            "pairs": mindmap_to_sentence_pairs(mindmap, sentences),
            "generation_status": "failed_validation",
        }
    

def parse_hierarchical_list_output(raw: str, n_sentences: int) -> MindMap:
    """Parse a numbered hierarchical list into a MindMap."""
    lines = [line.strip() for line in raw.splitlines() if line.strip()]

    parsed_nodes = []

    root_pattern = re.compile(
        r"^\s*\[(?P<id>\d+)\]\s*(?P<text>.*)$"
    )

    child_pattern = re.compile(
        r"^\s*(?P<path>\d+(?:\.\d+)*\.?)\s*\[(?P<id>\d+)\]\s*(?P<text>.*)$"
    )

    for line in lines:
        root_match = root_pattern.match(line)

        if root_match:
            sentence_id = int(root_match.group("id"))

            if 0 <= sentence_id < n_sentences:
                parsed_nodes.append(
                    {
                        "path": "",
                        "id": sentence_id,
                        "text": root_match.group("text").strip(),
                    }
                )

            continue

        child_match = child_pattern.match(line)

        if child_match:
            path = child_match.group("path").strip().rstrip(".")
            sentence_id = int(child_match.group("id"))

            if 0 <= sentence_id < n_sentences:
                parsed_nodes.append(
                    {
                        "path": path,
                        "id": sentence_id,
                        "text": child_match.group("text").strip(),
                    }
                )

    if not parsed_nodes:
        raise ValueError("Could not parse any nodes from hierarchical list output.")

    root_nodes = [node for node in parsed_nodes if node["path"] == ""]

    if root_nodes:
        root = root_nodes[0]["id"]
    else:
        parsed_nodes[0]["path"] = ""
        root = parsed_nodes[0]["id"]

    path_to_id: Dict[str, int] = {}
    selected = []
    seen_ids = set()

    for node in parsed_nodes:
        sentence_id = node["id"]

        if sentence_id in seen_ids:
            continue

        seen_ids.add(sentence_id)
        selected.append(sentence_id)
        path_to_id[node["path"]] = sentence_id

    edges = []

    for node in parsed_nodes:
        path = node["path"]
        child_id = node["id"]

        if path == "":
            continue

        if child_id not in seen_ids:
            continue

        parent_path = get_hierarchical_parent_path(path)
        parent_id = path_to_id.get(parent_path)

        if parent_id is None:
            parent_id = root

        if parent_id == child_id:
            continue

        edges.append((parent_id, child_id))

    mindmap = MindMap(
        root=root,
        selected=selected,
        edges=edges,
    )

    return repair_mindmap_tree(mindmap, n_sentences)


def get_hierarchical_parent_path(path: str) -> str:
    """Return the parent path for a numbered hierarchy path."""
    path = path.rstrip(".")

    if "." not in path:
        return ""

    return path.rsplit(".", 1)[0]


def repair_mindmap_tree(mindmap: MindMap, n_sentences: int) -> MindMap:
    """Repair duplicate parents, cycles and orphan nodes in a parsed MindMap."""
    root = mindmap.root

    selected = []

    for node in mindmap.selected:
        if 0 <= node < n_sentences and node not in selected:
            selected.append(node)

    if root not in selected:
        selected.insert(0, root)

    selected_set = set(selected)

    cleaned_edges = []
    parent_of = {}
    adjacency = {idx: [] for idx in range(n_sentences)}

    def creates_cycle(parent: int, child: int) -> bool:
        stack = [child]
        seen = set()

        while stack:
            current = stack.pop()

            if current == parent:
                return True

            if current in seen:
                continue

            seen.add(current)
            stack.extend(adjacency.get(current, []))

        return False

    for parent, child in mindmap.edges:
        if parent not in selected_set or child not in selected_set:
            continue

        if parent == child:
            continue

        if child == root:
            continue

        if child in parent_of:
            continue

        if creates_cycle(parent, child):
            continue

        parent_of[child] = parent
        adjacency[parent].append(child)
        cleaned_edges.append((parent, child))

    for node in selected:
        if node == root:
            continue

        if node not in parent_of:
            cleaned_edges.append((root, node))
            parent_of[node] = root

    selected = sorted(set([root] + [node for edge in cleaned_edges for node in edge]))

    return MindMap(
        root=root,
        selected=selected,
        edges=cleaned_edges,
    )


def validate_mindmap_object(
    mindmap: MindMap,
    n_sentences: int,
    max_nodes: int | None,
) -> List[str]:
    """Validate a parsed MindMap using formal tree constraints."""
    errors = []

    selected_set = set(mindmap.selected)

    if not 0 <= mindmap.root < n_sentences:
        errors.append("Root is outside valid sentence ID range.")

    if mindmap.root not in selected_set:
        errors.append("Root must be included in selected nodes.")

    if len(selected_set) < 2:
        errors.append("Too few selected nodes.")

    if max_nodes is not None and len(selected_set) > max_nodes:
        errors.append(f"Too many selected nodes: {len(selected_set)} > {max_nodes}.")

    if len(mindmap.edges) != max(0, len(selected_set) - 1):
        errors.append(
            f"Wrong number of edges: got {len(mindmap.edges)}, expected {len(selected_set) - 1}."
        )

    child_counts = {}

    for parent, child in mindmap.edges:
        if parent not in selected_set:
            errors.append(f"Edge parent {parent} is not selected.")

        if child not in selected_set:
            errors.append(f"Edge child {child} is not selected.")

        if parent == child:
            errors.append(f"Self-loop edge found: {parent} -> {child}.")

        child_counts[child] = child_counts.get(child, 0) + 1

    if child_counts.get(mindmap.root, 0) > 0:
        errors.append("Root must not appear as a child.")

    for node in selected_set:
        if node == mindmap.root:
            continue

        parent_count = child_counts.get(node, 0)

        if parent_count == 0:
            errors.append(f"Selected non-root node {node} has no parent.")
        elif parent_count > 1:
            errors.append(f"Selected non-root node {node} has multiple parents.")

    reachable = get_reachable_nodes(mindmap.root, mindmap.edges)

    if selected_set and reachable != selected_set:
        missing = sorted(selected_set - reachable)
        errors.append(f"Tree is not connected from root. Unreachable nodes: {missing}.")

    if has_cycle(mindmap.root, mindmap.edges):
        errors.append("Tree contains a cycle.")

    return errors


def validate_raw_mindmap_data(
    data: dict,
    n_sentences: int,
    max_nodes: int | None,
) -> list[str]:
    """Validate parsed raw mind-map JSON using only formal tree constraints."""
    errors = []

    if not isinstance(data, dict):
        return ["Output is not a JSON object."]

    root = data.get("root")
    selected = data.get("selected")
    edges = data.get("edges")

    if not isinstance(root, int):
        errors.append('"root" must be an integer.')

    if not isinstance(selected, list):
        errors.append('"selected" must be a list.')
        selected = []

    if not isinstance(edges, list):
        errors.append('"edges" must be a list.')
        edges = []

    selected_ints = []

    for node in selected:
        if not isinstance(node, int):
            errors.append(f'Selected node {node!r} is not an integer.')
            continue

        if not 0 <= node < n_sentences:
            errors.append(f"Selected node {node} is outside valid range 0..{n_sentences - 1}.")
            continue

        selected_ints.append(node)

    selected_set = set(selected_ints)

    if len(selected_ints) != len(selected_set):
        errors.append('"selected" contains duplicate nodes.')

    if isinstance(root, int):
        if not 0 <= root < n_sentences:
            errors.append(f'"root" is outside valid range 0..{n_sentences - 1}.')
        elif root not in selected_set:
            errors.append('"root" must be included in "selected".')

    if len(selected_set) < 2:
        errors.append("Too few selected nodes.")

    if max_nodes is not None and len(selected_set) > max_nodes:
        errors.append(f"Too many selected nodes: {len(selected_set)} > {max_nodes}.")

    edge_tuples = []

    for edge in edges:
        if not isinstance(edge, list) or len(edge) != 2:
            errors.append(f"Invalid edge format: {edge!r}.")
            continue

        parent, child = edge

        if not isinstance(parent, int) or not isinstance(child, int):
            errors.append(f"Edge contains non-integer IDs: {edge!r}.")
            continue

        if not 0 <= parent < n_sentences:
            errors.append(f"Edge parent {parent} is outside valid range.")

        if not 0 <= child < n_sentences:
            errors.append(f"Edge child {child} is outside valid range.")

        if parent == child:
            errors.append(f"Self-loop edge found: {edge!r}.")

        if parent not in selected_set:
            errors.append(f"Edge parent {parent} is not in selected.")

        if child not in selected_set:
            errors.append(f"Edge child {child} is not in selected.")

        edge_tuples.append((parent, child))

    if len(edge_tuples) != len(set(edge_tuples)):
        errors.append('"edges" contains duplicate edges.')

    expected_edges = max(0, len(selected_set) - 1)

    if len(edges) != expected_edges:
        errors.append(
            f"Wrong number of edges: got {len(edges)}, expected {expected_edges}."
        )

    if not isinstance(root, int) or root not in selected_set:
        return errors

    child_counts = {}

    for parent, child in edge_tuples:
        child_counts[child] = child_counts.get(child, 0) + 1

    if child_counts.get(root, 0) > 0:
        errors.append("Root must not appear as an edge child.")

    for node in selected_set:
        if node == root:
            continue

        parent_count = child_counts.get(node, 0)

        if parent_count == 0:
            errors.append(f"Selected non-root node {node} has no parent.")
        elif parent_count > 1:
            errors.append(f"Selected non-root node {node} has multiple parents.")

    reachable = get_reachable_nodes(root, edge_tuples)

    if selected_set and reachable != selected_set:
        missing = sorted(selected_set - reachable)
        errors.append(f"Tree is not connected from root. Unreachable nodes: {missing}.")

    if has_cycle(root, edge_tuples):
        errors.append("Tree contains a cycle.")

    return errors


def get_reachable_nodes(root: int, edges: list[tuple[int, int]]) -> set[int]:
    """Return all nodes reachable from the root."""
    children = {}

    for parent, child in edges:
        children.setdefault(parent, []).append(child)

    reachable = set()
    stack = [root]

    while stack:
        node = stack.pop()

        if node in reachable:
            continue

        reachable.add(node)
        stack.extend(children.get(node, []))

    return reachable


def has_cycle(root: int, edges: list[tuple[int, int]]) -> bool:
    """Detect whether a directed graph contains a cycle reachable from root."""
    children = {}

    for parent, child in edges:
        children.setdefault(parent, []).append(child)

    visiting = set()
    visited = set()

    def dfs(node: int) -> bool:
        if node in visiting:
            return True

        if node in visited:
            return False

        visiting.add(node)

        for child in children.get(node, []):
            if dfs(child):
                return True

        visiting.remove(node)
        visited.add(node)
        return False

    return dfs(root)


def extract_json_like(text: str) -> Any:
    """Extract the first JSON-like object or list from raw model output."""
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    if fenced:
        text = fenced.group(1).strip()

    first_obj = text.find("{")
    if first_obj != -1:
        depth = 0
        for idx in range(first_obj, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[first_obj : idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        return ast.literal_eval(candidate)

    first_list = text.find("[")
    if first_list != -1:
        depth = 0
        for idx in range(first_list, len(text)):
            if text[idx] == "[":
                depth += 1
            elif text[idx] == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[first_list : idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        return ast.literal_eval(candidate)

    raise ValueError(f"Could not extract JSON from model output:\n{text}")


def normalize_model_output(
    raw: str,
    n_sentences: int,
) -> MindMap:
    """Parse and repair raw model output into a valid rooted mind map."""
    data = extract_json_like(raw)

    if isinstance(data, list):
        edges = []
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                edges.append((int(item[0]), int(item[1])))

        selected = sorted(set(node for edge in edges for node in edge))
        root = selected[0] if selected else 0
        data = {"root": root, "selected": selected, "edges": edges}

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected parsed output type: {type(data)}")

    root = _safe_sentence_id(data.get("root", 0), n_sentences, default=0)

    selected = []
    for node in data.get("selected", []):
        parsed = _safe_sentence_id(node, n_sentences, default=None)
        if parsed is not None and parsed not in selected:
            selected.append(parsed)

    edges = []
    for edge in data.get("edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) != 2:
            continue

        parent = _safe_sentence_id(edge[0], n_sentences, default=None)
        child = _safe_sentence_id(edge[1], n_sentences, default=None)

        if parent is None or child is None or parent == child:
            continue

        edges.append((parent, child))

    selected_set = set(selected)
    selected_set.add(root)

    for parent, child in edges:
        selected_set.add(parent)
        selected_set.add(child)

    if len(selected_set) == 1:
        return MindMap(root=root, edges=[], selected=[root])

    selected = sorted(selected_set)

    cleaned_edges = _repair_edges(
        root=root,
        selected=selected,
        edges=edges,
        n_sentences=n_sentences,
    )

    selected = sorted(set([root] + [node for edge in cleaned_edges for node in edge]))
    return MindMap(root=root, edges=cleaned_edges, selected=selected)


def _safe_sentence_id(value: Any, n_sentences: int, default: int | None) -> int | None:
    """Convert a value to a valid sentence ID or return a default."""
    try:
        value = int(value)
    except Exception:
        return default

    if 0 <= value < n_sentences:
        return value

    return default


def _repair_edges(
    root: int,
    selected: List[int],
    edges: List[Tuple[int, int]],
    n_sentences: int,
) -> List[Tuple[int, int]]:
    """Remove invalid tree edges and attach orphan nodes to the root."""
    cleaned_edges = []
    parent_of = {}
    adjacency = {idx: [] for idx in range(n_sentences)}

    def creates_cycle(parent: int, child: int) -> bool:
        stack = [child]
        seen = set()

        while stack:
            current = stack.pop()

            if current == parent:
                return True

            if current in seen:
                continue

            seen.add(current)
            stack.extend(adjacency.get(current, []))

        return False

    for parent, child in edges:
        if child == root:
            continue

        if child in parent_of:
            continue

        if creates_cycle(parent, child):
            continue

        parent_of[child] = parent
        adjacency[parent].append(child)
        cleaned_edges.append((parent, child))

    for node in selected:
        if node == root:
            continue

        if node not in parent_of:
            cleaned_edges.append((root, node))
            parent_of[node] = root

    return cleaned_edges


def render_mindmap(mindmap: MindMap, sentences: List[str]) -> str:
    """Render a mind map as a numbered hierarchical list."""
    children = {}

    for parent, child in mindmap.edges:
        children.setdefault(parent, []).append(child)

    for parent in children:
        children[parent].sort()

    lines = [f"[{mindmap.root}] {sentences[mindmap.root]}"]

    def dfs(node: int, prefix: str) -> None:
        for child_index, child in enumerate(children.get(node, []), start=1):
            child_prefix = f"{prefix}.{child_index}" if prefix else str(child_index)
            lines.append(f"{child_prefix}. [{child}] {sentences[child]}")
            dfs(child, child_prefix)

    dfs(mindmap.root, "")
    return "\n".join(lines)


def mindmap_to_sentence_pairs(mindmap: MindMap, sentences: List[str]) -> List[List[str]]:
    """Convert a mind map to parent-child sentence pairs for CMGN-style evaluation."""
    pairs = [["", sentences[mindmap.root]]]

    for parent, child in mindmap.edges:
        pairs.append([sentences[parent], sentences[child]])

    if len(pairs) == 1:
        pairs.append(["", sentences[mindmap.root]])

    return pairs