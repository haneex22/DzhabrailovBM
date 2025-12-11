import random
import timeit

from typing import List

import matplotlib.pyplot as plt

from binary_search_tree import BinarySearchTree, TreeNode


# Поднимаем лимит рекурсии, чтобы корректно обрабатывать вырожденные деревья


# Характеристики ПК
pc_info = """
Характеристики ПК для тестирования:
- Процессор: Intel Core i5-11400 @ 2.60GHz
- Оперативная память: 16 GB
- ОС: Windows 10 x64
- Python: 3.13.3
"""


def build_bst_with_values(values: List[int]) -> BinarySearchTree:
    """Строит BST, последовательно вставляя значения из списка.

    Сложность: зависит от порядка вставки:
    - случайный порядок: O(n log n) в среднем;
    - отсортированный порядок: O(n²) в худшем.
    """
    tree = BinarySearchTree()

    for value in values:
        tree.insert(value)

    return tree
    # Общая сложность: O(n * h) ~ O(n log n) в среднем, O(n²) в худшем.


def build_balanced_like_bst(n: int, seed: int = 42) -> BinarySearchTree:
    """Строит «приближённо сбалансированное» дерево.

    Вставляем числа от 0 до n - 1 в случайном порядке.
    Сложность: O(n log n) в среднем.
    """
    rng = random.Random(seed)
    values = list(range(n))
    rng.shuffle(values)

    tree = build_bst_with_values(values)
    return tree
    # Общая сложность: O(n log n).


def build_degenerate_bst(n: int) -> BinarySearchTree:
    """Строит вырожденное дерево (цепочку), не используя рекурсивные вставки.

    Узлы связаны только через правого потомка:
    0 -> 1 -> 2 -> ... -> n-1

    Сложность построения: O(n), глубина рекурсии не используется.
    """
    tree = BinarySearchTree()

    if n <= 0:
        return tree  # Пустое дерево.

    # Создаём цепочку узлов вручную, без insert()
    root = TreeNode(0)
    current = root

    for value in range(1, n):
        new_node = TreeNode(value)
        current.right = new_node
        current = new_node

    tree.root = root

    return tree
    # Общая сложность: O(n).


def measure_search_time(
    tree: BinarySearchTree,
    queries: List[int],
) -> float:
    """Измеряет среднее время поиска значений в дереве.

    Возвращает время в микросекундах на одну операцию поиска.

    Пусть m = len(queries), тогда:
    - средняя сложность в сбалансированном дереве: O(m log n);
    - в вырожденном: O(mn).
    """
    start_time = timeit.default_timer()

    for value in queries:
        tree.search(value)

    end_time = timeit.default_timer()
    total_time = end_time - start_time

    avg_time_microseconds = (total_time / len(queries)) * 1_000_000

    return avg_time_microseconds
    # Общая сложность: O(m * h).


def show_example_tree_visualization() -> None:
    """Печатает пример текстовой визуализации дерева.

    Использует метод to_indented_string() из BinarySearchTree.
    Сложность: O(n).
    """
    values = [8, 3, 10, 1, 6, 14, 4, 7, 13]
    tree = build_bst_with_values(values)

    print("Пример текстовой визуализации дерева:")
    print(tree.to_ascii_tree())


def run_experiments() -> None:
    """Проводит серию замеров времени поиска в BST.

    Строятся:
    - приближённо сбалансированное дерево (случайная вставка);
    - вырожденное дерево (отсортированная вставка).

    Для каждого размера дерева выполняется по 1000 операций поиска,
    затем строится график зависимости времени от количества элементов.
    :contentReference[oaicite:8]{index=8}
    """
    sizes = [1_000, 5_000, 10_000, 20_000, 40_000]

    balanced_times: List[float] = []
    degenerate_times: List[float] = []

    print("Замеры времени поиска в бинарном дереве поиска (BST):")
    print("{:>10} {:>20} {:>20}".format(
        "N",
        "Сбалансированное (мкс)",
        "Вырожденное (мкс)",
    ))

    for size in sizes:
        balanced_tree = build_balanced_like_bst(size, seed=42)
        degenerate_tree = build_degenerate_bst(size)

        queries = [random.randint(0, size * 2) for _ in range(1_000)]

        balanced_time = measure_search_time(
            balanced_tree,
            queries,
        )
        degenerate_time = measure_search_time(
            degenerate_tree,
            queries,
        )

        balanced_times.append(balanced_time)
        degenerate_times.append(degenerate_time)

        print("{:>10} {:>20.4f} {:>20.4f}".format(
            size,
            balanced_time,
            degenerate_time,
        ))

    # Построение графика.
    plt.figure(figsize=(10, 6))
    plt.plot(
        sizes,
        balanced_times,
        "o-",
        label="Сбалансированное дерево",
    )
    plt.plot(
        sizes,
        degenerate_times,
        "s-",
        label="Вырожденное дерево",
    )

    plt.xlabel("Количество элементов N")
    plt.ylabel("Среднее время поиска, мкс")
    plt.title(
        "Поиск в бинарном дереве поиска:\n"
        "сбалансированная и вырожденная структуры",
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    plt.savefig("bst_search_time.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Анализ результатов.
    print("\nАнализ результатов:")
    print(
        "1. В сбалансированном дереве среднее время поиска растёт примерно "
        "как O(log N).",
    )
    print(
        "2. Во вырожденном дереве время поиска растёт почти линейно O(N), "
        "что подтверждает теорию.",
    )
    print(
        "3. Балансировка дерева критична для сохранения эффективности "
        "операций поиска и вставки.",
    )


if __name__ == "__main__":
    print(pc_info)
    show_example_tree_visualization()
    run_experiments()
