from __future__ import annotations

from typing import Generic, Optional, Tuple, TypeVar

from heap import MinHeap


T = TypeVar("T")


class PriorityQueue(Generic[T]):
    """Приоритетная очередь на основе min-кучи.

    Внутри используется min-heap из троек:
    (priority, counter, item).

    - priority: чем меньше число, тем выше приоритет;
    - counter: счётчик для сохранения стабильности (FIFO
      для одинаковых приоритетов);
    - item: сам объект.

    Основные операции:
    - enqueue(item, priority): O(log n);
    - dequeue(): O(log n);
    - peek(): O(1).
    """

    def __init__(self) -> None:
        """Создаёт пустую приоритетную очередь. O(1)."""
        self._heap: MinHeap[Tuple[int, int, T]] = MinHeap()  # O(1).
        self._counter: int = 0  # O(1).

    def is_empty(self) -> bool:
        """Проверяет, пуста ли очередь. O(1)."""
        return self._heap.is_empty()  # O(1).

    def enqueue(self, item: T, priority: int) -> None:
        """Добавляет элемент с указанным приоритетом.

        Меньшее значение priority означает более высокий приоритет.

        Сложность: O(log n).
        """
        entry = (priority, self._counter, item)  # O(1).
        self._counter += 1  # O(1).
        self._heap.insert(entry)  # O(log n).

    def dequeue(self) -> T:
        """Извлекает элемент с наивысшим приоритетом.

        Сложность: O(log n).
        """
        if self.is_empty():  # O(1).
            raise IndexError("Priority queue is empty.")

        priority, counter, item = self._heap.extract()  # O(log n).
        return item  # O(1).

    def peek(self) -> Optional[T]:
        """Возвращает элемент с наивысшим приоритетом без удаления.

        Сложность: O(1).
        """
        if self.is_empty():  # O(1).
            return None  # O(1).

        priority, counter, item = self._heap.peek()  # O(1).
        return item  # O(1).
