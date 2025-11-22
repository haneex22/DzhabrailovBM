# linked_list.py
"""
Реализация связного списка (LinkedList) с основными операциями.
"""

from typing import Any, Optional


class Node:
    """Узел связного списка."""

    def __init__(self, data: Any) -> None:
        """
        Инициализация узла.

        Args:
            data: Данные, хранящиеся в узле

        Сложность: O(1)
        """
        self.data: Any = data  # O(1)
        self.next: Optional['Node'] = None  # O(1)
# Общая сложность: O(1)


class LinkedList:
    """Односвязный список с основными операциями."""

    def __init__(self) -> None:
        """
        Инициализация пустого связного списка.

        Сложность: O(1)
        """
        self.head: Optional[Node] = None  # O(1)
        self.tail: Optional[Node] = None
        # O(1) - указатель на конец для оптимизации
        self.length: int = 0  # O(1)
# Общая сложность: O(1)

    def insert_at_start(self, data: Any) -> None:
        """
        Вставка элемента в начало списка.

        Args:
            data: Данные для вставки

        Сложность: O(1)
        """
        new_node = Node(data)  # O(1)
        if self.head is None:  # O(1)
            self.head = new_node  # O(1)
            self.tail = new_node  # O(1)
        else:
            new_node.next = self.head  # O(1)
            self.head = new_node  # O(1)
        self.length += 1  # O(1)
# Общая сложность: O(1)

    def insert_at_end(self, data: Any) -> None:
        """
        Вставка элемента в конец списка.

        Args:
            data: Данные для вставки

        Сложность: O(1) с tail, O(n) без tail
        """
        new_node = Node(data)  # O(1)
        if self.head is None:  # O(1)
            self.head = new_node  # O(1)
            self.tail = new_node  # O(1)
        else:
            # Используем assert для проверки, что tail не None
            assert self.tail is not None, "Tail should not be None"
            self.tail.next = new_node  # O(1)
            self.tail = new_node  # O(1)
        self.length += 1  # O(1)
# Общая сложность: O(1)

    def delete_from_start(self) -> Optional[Any]:
        """
        Удаление элемента из начала списка.

        Returns:
            Данные удаленного элемента или None если список пуст

        Сложность: O(1)
        """
        if self.head is None:  # O(1)
            return None  # O(1)

        data = self.head.data  # O(1)
        self.head = self.head.next  # O(1)
        self.length -= 1  # O(1)

        if self.head is None:  # O(1)
            self.tail = None  # O(1)

        return data  # O(1)
# Общая сложность: O(1)

    def traversal(self) -> list[Any]:
        """
        Обход списка и возврат всех элементов.

        Returns:
            Список всех элементов

        Сложность: O(n)
        """
        elements: list[Any] = []  # O(1)
        current = self.head  # O(1)
        while current is not None:  # O(n)
            elements.append(current.data)  # O(1)
            current = current.next  # O(1)
        return elements  # O(1)
# Общая сложность: O(n)

    def is_empty(self) -> bool:
        """
        Проверка, пуст ли список.

        Returns:
            True если список пуст, иначе False

        Сложность: O(1)
        """
        return self.head is None  # O(1)
# Общая сложность: O(1)
