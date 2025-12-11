from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TreeNode:
    """Узел бинарного дерева поиска."""

    value: int
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None


class BinarySearchTree:
    """Указательная реализация бинарного дерева поиска (BST).

    В среднем высота сбалансированного дерева h ~ O(log n), поэтому
    операции insert/search/delete работают за O(log n), в худшем случае
    (вырожденное дерево-цепочка) — за O(n).:contentReference[oaicite:2]
    {index=2}
    """

    def __init__(self) -> None:
        """Инициализирует пустое дерево.

        Сложность: O(1).
        """
        self.root: Optional[TreeNode] = None
        # Общая сложность метода: O(1).

    def insert(self, value: int) -> None:
        """Вставляет новое значение в BST.

        Средняя сложность: O(log n).
        Худшая сложность: O(n).
        """
        if self.root is None:
            self.root = TreeNode(value)
            return

        self._insert_recursive(self.root, value)
        # Общая сложность метода: O(h) ~ O(log n) в среднем, O(n) в худшем.

    def _insert_recursive(self, node: TreeNode, value: int) -> TreeNode:
        """Рекурсивный помощник для вставки.

        Сложность одного рекурсивного шага: O(1).
        Общая сложность: O(h), где h — высота дерева.
        """
        if node is None:
            return TreeNode(value)

        if value < node.value:
            if node.left is None:
                node.left = TreeNode(value)
            else:
                node.left = self._insert_recursive(node.left, value)
        elif value > node.value:
            if node.right is None:
                node.right = TreeNode(value)
            else:
                node.right = self._insert_recursive(node.right, value)
        # Если value == node.value, ничего не делаем, дубликаты не вставляем.

        return node
        # Общая сложность метода: O(h) ~ O(log n) в среднем, O(n) в худшем.

    def search(self, value: int) -> bool:
        """Ищет значение в BST.

        Возвращает True, если value найдено, иначе False.

        Средняя сложность: O(log n).
        Худшая сложность: O(n).
        """
        current = self.root

        while current is not None:
            if value == current.value:
                return True

            if value < current.value:
                current = current.left
            else:
                current = current.right

        return False
        # Общая сложность метода: O(h) ~ O(log n) в среднем, O(n) в худшем.

    def delete(self, value: int) -> None:
        """Удаляет значение из дерева, если оно есть.

        Средняя сложность: O(log n).
        Худшая сложность: O(n).
        """
        self.root = self._delete_recursive(self.root, value)
        # Общая сложность метода: O(h) ~ O(log n) в среднем, O(n) в худшем.

    def _delete_recursive(
        self,
        node: Optional[TreeNode],
        value: int,
    ) -> Optional[TreeNode]:
        """Рекурсивный помощник для удаления.

        Обрабатываются три случая:
        - удаление листа;
        - удаление узла с одним потомком;
        - удаление узла с двумя потомками (замена на минимальный в правом
          поддереве).:contentReference[oaicite:3]{index=3}

        Средняя сложность: O(log n).
        Худшая сложность: O(n).
        """
        if node is None:
            return None

        if value < node.value:
            node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            node.right = self._delete_recursive(node.right, value)
        else:
            # Нашли узел для удаления. O(1).
            if node.left is None and node.right is None:
                return None

            if node.left is None:
                return node.right

            if node.right is None:
                return node.left

            # Узел с двумя потомками: ищем минимальный в правом поддереве.
            successor = self.find_min(node.right)
            node.value = successor.value
            node.right = self._delete_recursive(
                node.right,
                successor.value,
            )

        return node
        # Общая сложность метода: O(h) ~ O(log n) в среднем, O(n) в худшем.

    def find_min(self, node: Optional[TreeNode]) -> TreeNode:
        """Возвращает узел с минимальным значением в поддереве.

        Сложность: O(h_subtree), где h_subtree — высота поддерева.
        """
        if node is None:
            raise ValueError("Cannot find minimum in an empty subtree.")

        current = node
        while current.left is not None:  # O(h_subtree).
            current = current.left

        return current
        # Общая сложность метода: O(h_subtree).

    def find_max(self, node: Optional[TreeNode]) -> TreeNode:
        """Возвращает узел с максимальным значением в поддереве.

        Сложность: O(h_subtree).
        """
        if node is None:
            raise ValueError("Cannot find maximum in an empty subtree.")

        current = node
        while current.right is not None:
            current = current.right

        return current

    def is_valid_bst(self) -> bool:
        """Проверяет, удовлетворяет ли дерево свойству BST.

        Используется рекурсивная проверка с диапазоном допустимых значений.

        Сложность: O(n), где n — число узлов.:contentReference[oaicite:4]
        {index=4}
        """
        result = self._is_valid_bst_node(self.root, None, None)
        return result
        # Общая сложность метода: O(n).

    def _is_valid_bst_node(
        self,
        node: Optional[TreeNode],
        min_value: Optional[int],
        max_value: Optional[int],
    ) -> bool:
        """Рекурсивная проверка узла и его поддеревьев.

        Сложность: O(n), каждый узел посещается один раз.
        """
        if node is None:
            return True

        if (min_value is not None and node.value <= min_value) or (
            max_value is not None and node.value >= max_value
        ):
            return False

        left_ok = self._is_valid_bst_node(
            node.left,
            min_value,
            node.value,
        )
        right_ok = self._is_valid_bst_node(
            node.right,
            node.value,
            max_value,
        )

        return left_ok and right_ok
        # Общая сложность метода: O(n).

    def height(self) -> int:
        """Возвращает высоту дерева.

        Для пустого дерева высота = -1.
        Сложность: O(n).
        """
        result = self._height_node(self.root)  # O(n).
        return result
        # Общая сложность метода: O(n).

    def _height_node(self, node: Optional[TreeNode]) -> int:
        """Рекурсивно вычисляет высоту поддерева.

        Сложность: O(n_subtree).
        """
        if node is None:
            return -1

        left_height = self._height_node(node.left)
        right_height = self._height_node(node.right)
        result = 1 + max(left_height, right_height)

        return result
        # Общая сложность метода: O(n_subtree).

    def to_ascii_tree(self) -> str:
        """Возвращает красивую ASCII-визуализацию BST (дерево сбоку)."""

        if self.root is None:
            return "<empty tree>"

        def build(node, prefix: str, is_left: bool):
            """Рекурсивно строит список строк для поддерева.

            Печатаем так:
            - сначала правое поддерево (оно сверху),
            - потом текущий узел,
            - потом левое поддерево (оно снизу).
            """
            if node is None:
                return []

            lines = []

            # Правое поддерево — идёт вверх
            if node.right is not None:
                lines += build(
                    node.right,
                    prefix + ("│   " if is_left else "    "),
                    False,
                )

            # Текущий узел
            lines.append(
                prefix
                + ("└── " if is_left else "┌── ")
                + str(node.value),
            )

            # Левое поддерево — идёт вниз
            if node.left is not None:
                lines += build(
                    node.left,
                    prefix + ("    " if is_left else "│   "),
                    True,
                )

            return lines

        # Корень считаем «правым» (is_left=False), чтобы он рисовался с "┌──"
        return "\n".join(build(self.root, "", False))
