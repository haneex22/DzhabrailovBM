import unittest
from io import StringIO
from contextlib import redirect_stdout

from binary_search_tree import BinarySearchTree
from tree_traversal import (
    in_order_recursive,
    pre_order_recursive,
    post_order_recursive,
    in_order_iterative,
)


BASE_VALUES = [8, 3, 10, 1, 6, 14, 4, 7, 13]


def build_tree(values):
    """Вспомогательная функция: строит BST из списка значений."""
    tree = BinarySearchTree()
    for value in values:
        tree.insert(value)
    return tree


class TestBinarySearchTreeBasic(unittest.TestCase):
    """Тесты базовых операций BST: вставка, поиск, удаление и т.п."""

    def setUp(self) -> None:
        """Создаём одно и то же дерево перед каждым тестом."""
        self.tree = build_tree(BASE_VALUES)

    def test_search_existing_values(self):
        """Все вставленные значения должны находиться в дереве."""
        for value in BASE_VALUES:
            with self.subTest(value=value):
                self.assertTrue(self.tree.search(value))

    def test_search_missing_values(self):
        """Значения, которых не было, найти нельзя."""
        for value in (-10, 0, 2, 5, 9, 11, 15, 100):
            with self.subTest(value=value):
                self.assertFalse(self.tree.search(value))

    def test_find_min_max(self):
        """Проверка нахождения минимума и максимума."""
        min_node = self.tree.find_min(self.tree.root)
        max_node = self.tree.find_max(self.tree.root)

        self.assertEqual(min_node.value, 1)
        self.assertEqual(max_node.value, 14)

    def test_height(self):
        """Высота тестового дерева должна быть равна 3.

        Путь максимальной длины: 8 -> 3 -> 6 -> 4 (3 ребра).
        Для пустого дерева высота = -1.
        """
        self.assertEqual(self.tree.height(), 3)

        empty_tree = BinarySearchTree()
        self.assertEqual(empty_tree.height(), -1)

    def test_is_valid_bst_true(self):
        """Корректно построенное дерево должно быть валидным BST."""
        self.assertTrue(self.tree.is_valid_bst())

    def test_is_valid_bst_false_after_manual_corruption(self):
        """После нарушения свойства BST метод должен вернуть False."""
        # Нарушаем свойство: делаем левый дочерний узел больше корня.
        self.tree.root.left.value = 100
        self.assertFalse(self.tree.is_valid_bst())

    def test_delete_leaf(self):
        """Удаление листа (узел без потомков) работает корректно."""
        self.assertTrue(self.tree.search(1))
        self.tree.delete(1)
        self.assertFalse(self.tree.search(1))
        self.assertTrue(self.tree.is_valid_bst())

    def test_delete_node_with_one_child(self):
        """Удаление узла с одним потомком (например, 14)."""
        # В исходном дереве 14 имеет только левый потомок 13.
        self.assertTrue(self.tree.search(14))
        self.assertTrue(self.tree.search(13))

        self.tree.delete(14)

        self.assertFalse(self.tree.search(14))
        self.assertTrue(self.tree.search(13))
        self.assertTrue(self.tree.is_valid_bst())

    def test_delete_node_with_two_children(self):
        """Удаление узла с двумя потомками (например, 3)."""
        self.assertTrue(self.tree.search(3))

        self.tree.delete(3)

        self.assertFalse(self.tree.search(3))
        self.assertTrue(self.tree.is_valid_bst())


class TestTreeTraversals(unittest.TestCase):
    """Тесты обходов дерева."""

    def setUp(self) -> None:
        self.tree = build_tree(BASE_VALUES)

    def _capture_output(self, func, *args, **kwargs) -> str:
        """Запускает функцию и возвращает всё, что она напечатала."""
        buffer = StringIO()
        with redirect_stdout(buffer):
            func(*args, **kwargs)
        return buffer.getvalue().strip()

    def test_in_order_recursive_sorted(self):
        """Рекурсивный in-order обход должен дать отсортированные значения."""
        output = self._capture_output(in_order_recursive, self.tree.root)
        self.assertEqual(output, "1 3 4 6 7 8 10 13 14")

    def test_in_order_iterative_equals_recursive(self):
        """Итеративный и рекурсивный in-order обходы должны совпадать."""
        recursive_output = self._capture_output(
            in_order_recursive,
            self.tree.root,
        )
        iterative_output = self._capture_output(
            in_order_iterative,
            self.tree.root,
        )
        self.assertEqual(recursive_output, iterative_output)

    def test_pre_order_recursive(self):
        """Проверяем корректность pre-order обхода."""
        output = self._capture_output(pre_order_recursive, self.tree.root)
        # Для BASE_VALUES = [8, 3, 10, 1, 6, 14, 4, 7, 13]:
        # Структура дерева:
        #          8
        #       /     \
        #      3       10
        #    /   \        \
        #   1     6        14
        #        / \      /
        #       4   7    13
        self.assertEqual(output, "8 3 1 6 4 7 10 14 13")

    def test_post_order_recursive(self):
        """Проверяем корректность post-order обхода."""
        output = self._capture_output(post_order_recursive, self.tree.root)
        # Post-order: левое поддерево, правое поддерево, корень.
        # Для дерева из комментария выше: 1 4 7 6 3 13 14 10 8
        self.assertEqual(output, "1 4 7 6 3 13 14 10 8")


class TestAsciiTree(unittest.TestCase):
    """Тесты ASCII-визуализации дерева (если реализована to_ascii_tree)."""

    def test_to_ascii_tree_exists_and_contains_values(self):
        """Если метод to_ascii_tree есть, он должен возвращать с узлами."""
        tree = build_tree(BASE_VALUES)

        # Не во всех версиях класса он может быть реализован — проверим.
        if hasattr(tree, "to_ascii_tree"):
            ascii_view = tree.to_ascii_tree()
            self.assertIsInstance(ascii_view, str)
            # Проверяем, что в ASCII-представлении есть все значения.
            for value in BASE_VALUES:
                with self.subTest(value=value):
                    self.assertIn(str(value), ascii_view)
        else:
            # Если метода нет, тест считаем пройденным, чтобы не ломать сборку.
            self.skipTest("Метод to_ascii_tree не реализован.")


if __name__ == "__main__":
    unittest.main()
