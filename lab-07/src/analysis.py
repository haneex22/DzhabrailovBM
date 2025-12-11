from __future__ import annotations

import random
import timeit
from typing import Callable, List

import matplotlib.pyplot as plt

from heap import MinHeap
from heapsort import heapsort_in_place


pc_info = """
Характеристики ПК для тестирования:
- Процессор: Intel Core i5-11400 @ 2.60GHz
- Оперативная память: 16 GB
- ОС: Windows 10 x64
- Python: 3.13.3
"""


def build_heap_by_inserts(values: List[int]) -> MinHeap[int]:
    """Строит min-кучу последовательными вставками.

    Сложность: O(n log n) в худшем случае.
    """
    heap = MinHeap[int]()

    for value in values:
        heap.insert(value)

    return heap
    # Общая сложность: O(n log n).


def build_heap_by_heapify(values: List[int]) -> MinHeap[int]:
    """Строит min-кучу с помощью алгоритма build_heap (heapify).

    Сложность: O(n).
    """
    heap = MinHeap[int](values)
    return heap


def measure_time(func: Callable[[], None], repeats: int = 5) -> float:
    """Измеряет среднее время выполнения функции в миллисекундах.

    Сложность: O(repeats * cost(func)).
    """
    total_time = timeit.timeit(func, number=repeats)
    avg_ms = (total_time / repeats) * 1000
    return avg_ms


def experiment_build_heap() -> None:
    """Сравнивает время построения кучи двумя методами.

    1. Последовательные вставки (O(n log n)).
    2. Алгоритм heapify / build_heap (O(n)).

    Строит таблицу и график.
    """
    sizes = [1_000, 5_000, 10_000, 50_000, 100_000]
    times_insert: List[float] = []
    times_heapify: List[float] = []

    print("Сравнение методов построения кучи (min-heap):")
    print("{:>10} {:>20} {:>20}".format(
        "N",
        "Вставки, мс",
        "Heapify, мс",
    ))

    for size in sizes:
        values = [random.randint(0, size) for _ in range(size)]

        def build_by_inserts() -> None:
            build_heap_by_inserts(values)

        def build_by_heapify() -> None:
            build_heap_by_heapify(values)

        time_insert_ms = measure_time(build_by_inserts, repeats=3)
        time_heapify_ms = measure_time(build_by_heapify, repeats=3)

        times_insert.append(time_insert_ms)
        times_heapify.append(time_heapify_ms)

        print("{:>10} {:>20.4f} {:>20.4f}".format(
            size,
            time_insert_ms,
            time_heapify_ms,
        ))

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_insert, "o-", label="Последовательные вставки")
    plt.plot(sizes, times_heapify, "s-", label="Heapify (build_heap)")
    plt.xlabel("Количество элементов N")
    plt.ylabel("Время построения, мс")
    plt.title("Построение кучи: вставки против heapify")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig("heap_build_times.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nВывод:")
    print(
        "- метод heapify показывает почти линейный рост времени (O(N)), "
        "что согласуется с теорией;"
    )
    print(
        "- последовательные вставки растут заметно быстрее ближе к O(N log N)"
        "особенно на больших N."
    )


def quicksort(array: List[int]) -> List[int]:
    """Реализация быстрой сортировки (QuickSort) с рандомным выбором опорного.

    Средняя сложность: O(n log n).
    Худшая сложность: O(n²) (редко, благодаря случайному опорному).
    """
    if len(array) <= 1:
        return array[:]

    pivot = random.choice(array)
    less = [x for x in array if x < pivot]
    equal = [x for x in array if x == pivot]
    greater = [x for x in array if x > pivot]

    return quicksort(less) + equal + quicksort(greater)
    # T(n) ~ 2 T(n/2) + O(n) => O(n log n) в среднем.


def mergesort(array: List[int]) -> List[int]:
    """Классическая сортировка слиянием (MergeSort).

    Сложность: O(n log n) в лучшем, среднем и худшем случаях.
    """
    n = len(array)
    if n <= 1:
        return array[:]

    mid = n // 2
    left = mergesort(array[:mid])
    right = mergesort(array[mid:])

    # Слияние двух отсортированных массивов. O(n).
    merged: List[int] = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])

    return merged


def experiment_sorting() -> None:
    """Сравнивает Heapsort, QuickSort и MergeSort на случайных данных.

    Замеряется время сортировки для нескольких размеров N.
    Строится таблица и график.
    """
    sizes = [1_000, 5_000, 10_000, 20_000]
    times_heap: List[float] = []
    times_quick: List[float] = []
    times_merge: List[float] = []

    print("\nСравнение алгоритмов сортировки:")
    print("{:>10} {:>15} {:>15} {:>15}".format(
        "N",
        "Heapsort, мс",
        "QuickSort, мс",
        "MergeSort, мс",
    ))

    for size in sizes:
        base = [random.randint(0, size) for _ in range(size)]

        def run_heapsort() -> None:
            data = list(base)
            heapsort_in_place(data)

        def run_quicksort() -> None:
            data = list(base)
            quicksort(data)

        def run_mergesort() -> None:
            data = list(base)
            mergesort(data)

        time_heap_ms = measure_time(run_heapsort, repeats=3)
        time_quick_ms = measure_time(run_quicksort, repeats=3)
        time_merge_ms = measure_time(run_mergesort, repeats=3)

        times_heap.append(time_heap_ms)
        times_quick.append(time_quick_ms)
        times_merge.append(time_merge_ms)

        print("{:>10} {:>15.4f} {:>15.4f} {:>15.4f}".format(
            size,
            time_heap_ms,
            time_quick_ms,
            time_merge_ms,
        ))

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, times_heap, "o-", label="Heapsort (in-place)")
    plt.plot(sizes, times_quick, "s-", label="QuickSort")
    plt.plot(sizes, times_merge, "^-", label="MergeSort")
    plt.xlabel("Количество элементов N")
    plt.ylabel("Время сортировки, мс")
    plt.title("Сравнение алгоритмов сортировки")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.savefig("sorting_algorithms_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.show()

    print("\nАнализ:")
    print(
        "- QuickSort обычно самый быстрый на случайных данных (благодаря малым"
        "константам и хорошей локальности);"
    )
    print(
        "- MergeSort показывает стабильное O(N log N), но требует"
        "дополнительную память;"
    )
    print(
        "- Heapsort имеет гарантию O(N log N) и O(1) дополнительной памяти, "
        "но из-за констант может быть немного медленнее."
    )


def demo_heap_visualization() -> None:
    """Небольшой пример визуализации кучи.

    Сложность: O(n).
    """
    values = [5, 3, 8, 1, 7, 10, 2]
    heap = MinHeap(values)
    print("Куча (min-heap) по уровням:")
    print(heap.to_tree_string())


if __name__ == "__main__":
    print(pc_info)
    demo_heap_visualization()
    experiment_build_heap()
    experiment_sorting()
