import unittest
from typing import List

from heap import MinHeap, MaxHeap
from heapsort import heapsort, heapsort_in_place
from priority_queue import PriorityQueue


class TestHeapBasic(unittest.TestCase):
    """Тесты базовых операций кучи."""

    def test_insert_and_extract_min_heap(self) -> None:
        """Min-heap должна всегда выдавать минимальный элемент первой."""
        heap = MinHeap[int]()
        values = [5, 3, 8, 1, 7]

        for value in values:
            heap.insert(value)

        extracted: List[int] = []
        while not heap.is_empty():
            extracted.append(heap.extract())

        self.assertEqual(sorted(values), extracted)

    def test_insert_and_extract_max_heap(self) -> None:
        """Max-heap должна всегда выдавать максимальный элемент первой."""
        heap = MaxHeap[int]()
        values = [5, 3, 8, 1, 7]

        for value in values:
            heap.insert(value)

        extracted: List[int] = []
        while not heap.is_empty():
            extracted.append(heap.extract())

        self.assertEqual(sorted(values, reverse=True), extracted)

    def test_build_heap_from_array(self) -> None:
        """build_heap должен строить корректную min-кучу из массива."""
        values = [5, 3, 8, 1, 7]
        heap = MinHeap(values)

        # Проверяем свойство min-кучи: каждый родитель <= детей.
        data = heap._data  # type: ignore[attr-defined]
        size = len(data)

        for i in range(size):
            left = 2 * i + 1
            right = 2 * i + 2

            if left < size:
                self.assertLessEqual(data[i], data[left])
            if right < size:
                self.assertLessEqual(data[i], data[right])

    def test_peek_does_not_remove(self) -> None:
        """peek возвращает корень, но не удаляет его."""
        heap = MinHeap[int]()
        for value in [5, 3, 8]:
            heap.insert(value)

        top = heap.peek()
        self.assertEqual(top, 3)
        self.assertEqual(len(heap), 3)


class TestHeapsort(unittest.TestCase):
    """Тесты алгоритмов Heapsort."""

    def test_heapsort_returns_sorted_list(self) -> None:
        """Функция heapsort должна возвращать отсортированный список."""
        values = [5, 1, 4, 2, 8, 3]
        result = heapsort(values)
        self.assertEqual(result, sorted(values))
        # Исходный список не изменяется.
        self.assertEqual(values, [5, 1, 4, 2, 8, 3])

    def test_heapsort_in_place_sorts_array(self) -> None:
        """heapsort_in_place должен сортировать список на месте."""
        values = [5, 1, 4, 2, 8, 3]
        heapsort_in_place(values)
        self.assertEqual(values, sorted([5, 1, 4, 2, 8, 3]))


class TestPriorityQueue(unittest.TestCase):
    """Тесты приоритетной очереди на куче."""

    def test_enqueue_dequeue_by_priority(self) -> None:
        """Элементы должны выходить в порядке приоритета."""
        pq: PriorityQueue[str] = PriorityQueue()
        pq.enqueue("low", priority=10)
        pq.enqueue("medium", priority=5)
        pq.enqueue("high", priority=1)

        self.assertEqual(pq.dequeue(), "high")
        self.assertEqual(pq.dequeue(), "medium")
        self.assertEqual(pq.dequeue(), "low")
        self.assertTrue(pq.is_empty())

    def test_stable_for_equal_priorities(self) -> None:
        """Для одинаковых приоритетов сохраняется FIFO-порядок."""
        pq: PriorityQueue[str] = PriorityQueue()
        pq.enqueue("first", priority=5)
        pq.enqueue("second", priority=5)
        pq.enqueue("third", priority=5)

        self.assertEqual(pq.dequeue(), "first")
        self.assertEqual(pq.dequeue(), "second")
        self.assertEqual(pq.dequeue(), "third")


if __name__ == "__main__":
    unittest.main()
