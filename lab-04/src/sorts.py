"""
Модуль с реализацией алгоритмов сортировки.
"""

from typing import List


def bubble_sort(arr: List[int]) -> List[int]:
    """
    Сортировка пузырьком.

    Временная сложность:
    - Лучший случай: O(n)
    - Средний случай: O(n²)
    - Худший случай: O(n²)

    Пространственная сложность: O(1)
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr


def selection_sort(arr: List[int]) -> List[int]:
    """
    Сортировка выбором.

    Временная сложность:
    - Лучший случай: O(n²)
    - Средний случай: O(n²)
    - Худший случай: O(n²)

    Пространственная сложность: O(1)
    """
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr


def insertion_sort(arr: List[int]) -> List[int]:
    """
    Сортировка вставками.

    Временная сложность:
    - Лучший случай: O(n)
    - Средний случай: O(n²)
    - Худший случай: O(n²)

    Пространственная сложность: O(1)
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


def merge_sort(arr: List[int]) -> List[int]:
    """
    Сортировка слиянием.

    Временная сложность:
    - Лучший случай: O(n log n)
    - Средний случай: O(n log n)
    - Худший случай: O(n log n)

    Пространственная сложность: O(n)
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return _merge(left, right)


def _merge(left: List[int], right: List[int]) -> List[int]:
    """Вспомогательная функция для слияния двух отсортированных массивов."""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


def quick_sort(arr: List[int]) -> List[int]:
    """
    Быстрая сортировка.

    Временная сложность:
    - Лучший случай: O(n log n)
    - Средний случай: O(n log n)
    - Худший случай: O(n²)

    Пространственная сложность: O(log n)
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)


def is_sorted(arr: List[int]) -> bool:
    """Проверка, отсортирован ли массив."""
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
