from __future__ import annotations

from typing import Optional, List

from binary_search_tree import TreeNode


def in_order_recursive(node: Optional[TreeNode]) -> None:
    """Рекурсивный симметричный (in-order) обход: левый–корень–правый.

    Для BST выводит значения в порядке возрастания.:contentReference[oaicite:5]
    {index=5}
    Сложность: O(n).
    """
    if node is None:
        return

    in_order_recursive(node.left)
    print(node.value, end=" ")
    in_order_recursive(node.right)
    # Общая сложность: O(n).


def pre_order_recursive(node: Optional[TreeNode]) -> None:
    """Рекурсивный прямой (pre-order) обход: корень–левый–правый.

    Полезен для копирования структуры дерева.:contentReference[oaicite:6]
    {index=6}
    Сложность: O(n).
    """
    if node is None:
        return

    print(node.value, end=" ")
    pre_order_recursive(node.left)
    pre_order_recursive(node.right)
    # Общая сложность: O(n).


def post_order_recursive(node: Optional[TreeNode]) -> None:
    """Рекурсивный обратный (post-order) обход: левый–правый–корень.

    Полезен для удаления дерева.:contentReference[oaicite:7]{index=7}
    Сложность: O(n).
    """
    if node is None:
        return

    post_order_recursive(node.left)
    post_order_recursive(node.right)
    print(node.value, end=" ")
    # Общая сложность: O(n).


def in_order_iterative(root: Optional[TreeNode]) -> None:
    """Итеративный in-order обход с использованием стека.

    Сложность: O(n), память: O(h), где h — высота дерева.
    """
    stack: List[TreeNode] = []
    current = root

    while stack or current is not None:
        while current is not None:
            stack.append(current)
            current = current.left

        current = stack.pop()
        print(current.value, end=" ")
        current = current.right

    # Общая сложность: O(n).
