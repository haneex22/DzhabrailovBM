from __future__ import annotations

from typing import Iterable, List, TypeVar

from heap import MinHeap, SupportsComparison


T = TypeVar("T", bound=SupportsComparison)


def heapsort(iterable: Iterable[T]) -> List[T]:
    """Сортировка списка с использованием min-кучи.

    Алгоритм:
    1. Строим min-heap из всех элементов: O(n).
    2. Последовательно извлекаем минимум и добавляем в результат: O(n log n).

    Итоговая сложность: O(n log n).

    Возвращает новый отсортированный список, исходные данные не изменяются.
    """
    heap = MinHeap(iterable)
    result: List[T] = []

    while not heap.is_empty():
        result.append(heap.extract())

    return result
    # Общая сложность: O(n log n).


def _sift_down_range(
    array: List[T],
    start: int,
    end: int,
) -> None:
    """Вспомогательный sift-down для in-place Heapsort.

    Поддерживает max-heap в подмассиве array[start:end].

    Сложность: O(log n_sub), где n_sub = end - start.
    """
    root = start

    while True:
        left = 2 * root + 1
        right = 2 * root + 2
        largest = root

        if left < end and array[left] > array[largest]:
            largest = left

        if right < end and array[right] > array[largest]:
            largest = right

        if largest == root:
            break

        array[root], array[largest] = array[largest], array[root]
        root = largest

    # Общая сложность: O(h) ~ O(log n_sub).


def heapsort_in_place(array: List[T]) -> None:
    """In-place версия Heapsort.

    Сортирует массив по возрастанию "на месте",
    не используя дополнительную память под структуру кучи.

    Этапы:
    1. Преобразуем массив в max-heap (O(n)).
    2. Повторяем:
       - меняем местами корень (максимум) и последний элемент
         неотсортированной части;
       - уменьшаем размер кучи;
       - восстанавливаем свойство кучи sift-down (O(log n)).

    Итоговая сложность: O(n log n).
    Память: O(1) дополнительная.
    """
    n = len(array)

    # Строим max-heap (bottom-up). O(n).
    for index in range(n // 2 - 1, -1, -1):
        _sift_down_range(array, index, n)

    # Выносим максимум в конец и восстанавливаем кучу. O(n log n).
    for end in range(n - 1, 0, -1):
        array[0], array[end] = array[end], array[0]
        _sift_down_range(array, 0, end)

    # Общая сложность: O(n log n).
