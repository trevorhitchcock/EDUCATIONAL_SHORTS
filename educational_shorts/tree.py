from pathlib import Path
from educational_shorts.client import ask_llm
from educational_shorts.schemas import KnowledgeNode
import json

def find_node(
    node: KnowledgeNode,
    target_name: str,
) -> KnowledgeNode | None:
    if node.name == target_name:
        return node

    for child in node.children:
        result = find_node(child, target_name)

        if result is not None:
            return result

    return None

def find_first_unexpanded_node(
    node: KnowledgeNode,
    max_depth: int,
    current_depth: int = 0,
) -> KnowledgeNode | None:
    if not node.children and current_depth < max_depth:
        return node

    for child in node.children:
        result = find_first_unexpanded_node(
            node=child,
            max_depth=max_depth,
            current_depth=current_depth + 1,
        )

        if result is not None:
            return result

    return None

def build_expansion_prompt(category_name: str) -> str:
    return f"""
Create a knowledge node for the category "{category_name}".

Generate exactly 8 immediate subcategories.

Generate only the immediate children of the requested category.

Do not include the category itself, its ancestors, its siblings, or broader related fields.

Each child should represent exactly one level deeper in the knowledge hierarchy.

The root node should be named "{category_name}".

Each child should have an empty children list.
"""

def expand_node(
    tree: KnowledgeNode,
    category_name: str,
    system_prompt: str,
) -> None:
    user_prompt = build_expansion_prompt(category_name)

    expanded_node = ask_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=KnowledgeNode,
    )

    node = find_node(tree, category_name)

    if node is None:
        raise ValueError(f'Could not find "{category_name}" in the tree.')

    node.children = expanded_node.children

def save_tree(
    tree: KnowledgeNode,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        tree.model_dump_json(indent=2),
        encoding="utf-8",
    )

def load_tree(input_path: Path) -> KnowledgeNode:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    return KnowledgeNode.model_validate(data)

def expand_next_node(
    tree: KnowledgeNode,
    system_prompt: str,
    max_depth: int,
) -> KnowledgeNode | None:
    next_node = find_first_unexpanded_node(
        node=tree,
        max_depth=max_depth,
    )

    if next_node is None:
        return None

    expand_node(
        tree=tree,
        category_name=next_node.name,
        system_prompt=system_prompt,
    )

    return next_node

def expand_tree(
    tree: KnowledgeNode,
    system_prompt: str,
    max_nodes: int,
    max_depth: int,
) -> list[str]:
    expanded_names = []

    while len(expanded_names) < max_nodes:
        expanded_node = expand_next_node(
            tree=tree,
            system_prompt=system_prompt,
            max_depth=max_depth,
        )

        if expanded_node is None:
            break

        expanded_names.append(expanded_node.name)

    return expanded_names
def find_node_depth(
    node: KnowledgeNode,
    target_name: str,
    current_depth: int = 0,
) -> int | None:
    if node.name == target_name:
        return current_depth

    for child in node.children:
        result = find_node_depth(
            child,
            target_name,
            current_depth + 1,
        )

        if result is not None:
            return result

    return None

def count_nodes(node: KnowledgeNode) -> int:
    return 1 + sum(count_nodes(child) for child in node.children)