from __future__ import annotations

from typing import List

from greedy_algorithms import (
    Edge,
    Interval,
    Item,
    coin_change_greedy,
    fractional_knapsack,
    huffman_decode,
    huffman_encode,
    kruskal_mst,
    select_intervals,
)


PC_INFO = """\
Характеристики ПК для тестирования:
- Процессор: Intel Core i5-11400 @ 2.60GHz
- Оперативная память: 16 GB
- ОС: Windows 10 x64
- Python: 3.13.3
"""


def demo_interval_scheduling() -> None:
    """Демонстрация жадного выбора непересекающихся интервалов.

    Сложность демонстрации: O(n log n), где n — число интервалов.
    """
    print("=" * 80)
    print("Задача о выборе заявок (Interval Scheduling)")
    print("=" * 80)

    intervals: List[Interval] = [
        Interval(1, 4),
        Interval(3, 5),
        Interval(0, 6),
        Interval(5, 7),
        Interval(3, 9),
        Interval(5, 9),
        Interval(6, 10),
        Interval(8, 11),
        Interval(8, 12),
        Interval(2, 14),
        Interval(12, 16),
    ]

    print("Все интервалы:")
    for it in intervals:
        print(f"  [{it.start}, {it.end}]")

    selected = select_intervals(intervals)

    print("\nВыбранные интервалы (жадный алгоритм по раннему окончанию):")
    for it in selected:
        print(f"  [{it.start}, {it.end}]")

    print(f"\nВсего выбрано интервалов: {len(selected)}")
    print()


def demo_fractional_knapsack() -> None:
    """Демонстрация дробного (непрерывного) рюкзака.

    Сложность демонстрации: O(n log n), где n — число предметов.
    """
    print("=" * 80)
    print("Задача о дробном рюкзаке (fractional knapsack)")
    print("=" * 80)

    items = [
        Item(weight=15, value=70),
        Item(weight=25, value=150),
        Item(weight=35, value=180),
    ]
    capacity = 70.0

    print(f"Вместимость рюкзака: {capacity}")
    print("Предметы (вес, стоимость, удельная стоимость):")
    for it in items:
        ratio = it.value / it.weight
        print(f"  w = {it.weight:5.1f},v = {it.value:5.1f},v/w = {ratio:5.2f}")

    total_value, selection = fractional_knapsack(capacity, items)

    print("\nЖадный выбор (по максимальной удельной стоимости):")
    for it, fraction in selection:
        print(
            f"  берем {fraction*100:6.2f}% предмета "
            f"(w = {it.weight:.1f}, v = {it.value:.1f})"
        )

    print(f"\nИтоговая ценность рюкзака: {total_value:.2f}")
    print()


def demo_coin_change() -> None:
    """Демонстрация жадного алгоритма размена монет.

    Сложность демонстрации: O(k log k), где k — число номиналов.
    """
    print("=" * 80)
    print("Задача о размене монет (coin change, greedy)")
    print("=" * 80)

    amount = 63
    denominations = [1, 5, 10, 25]

    print(f"Сумма для размена: {amount}")
    print(f"Номиналы монет: {denominations}")

    result = coin_change_greedy(amount, denominations)

    print("\nЖадный размен (максимально крупные монеты):")
    for coin in sorted(result.keys(), reverse=True):
        count = result[coin]
        print(f"  монета {coin}: {count} шт.")

    total = sum(coin * count for coin, count in result.items())
    num_coins = sum(result.values())
    print(f"\nПроверка: набранная сумма = {total}, число монет = {num_coins}")
    print()


def demo_huffman() -> None:
    """Демонстрация кодирования и декодирования строки кодом Хаффмана.

    Сложность:
        Пусть n — длина текста, k — размер алфавита.
        Временная ~ O(n + k log k).
    """
    print("=" * 80)
    print("Код Хаффмана (Huffman coding)")
    print("=" * 80)

    text = "abracadabra huffman test"
    print(f"Исходный текст: {text!r}")

    encoded_bits, codes, root = huffman_encode(text)

    print("\nКоды символов (префиксный код):")
    for symbol, code in sorted(codes.items()):
        printable = symbol if symbol != " " else "<space>"
        print(f"  {printable!r}: {code}")

    print(f"\nЗакодированная битовая строка: {encoded_bits}")
    print(f"Длина исходного текста (символы): {len(text)}")
    print(f"Длина закодированного текста (биты): {len(encoded_bits)}")

    decoded = huffman_decode(encoded_bits, root)
    print(f"\nДекодированный текст: {decoded!r}")
    print(f"Совпадение с исходным: {decoded == text}")
    print()


def demo_kruskal_mst() -> None:
    """Демонстрация алгоритма Краскала для MST.

    Сложность:
        Пусть E — число рёбер, V — число вершин.
        Временная: O(E log E).
    """
    print("=" * 80)
    print("Минимальное остовное дерево (алгоритм Краскала)")
    print("=" * 80)

    num_vertices = 5
    edges: List[Edge] = [
        Edge(0, 1, 2.0),
        Edge(0, 3, 6.0),
        Edge(1, 2, 3.0),
        Edge(1, 3, 8.0),
        Edge(1, 4, 5.0),
        Edge(2, 4, 7.0),
        Edge(3, 4, 9.0),
    ]

    print(f"Число вершин: {num_vertices}")
    print("Рёбра графа (u, v, w):")
    for edge in edges:
        print(f"  {edge.u} -- {edge.v} (w = {edge.weight})")

    total_weight, mst_edges = kruskal_mst(num_vertices, edges)

    print("\nРёбра MST (минимальное остовное дерево):")
    for edge in mst_edges:
        print(f"  {edge.u} -- {edge.v} (w = {edge.weight})")

    print(f"\nСуммарный вес MST: {total_weight}")
    print()


def main() -> None:
    """Основная точка входа для демонстрации всех жадных задач темы 08."""
    print(PC_INFO)
    demo_interval_scheduling()
    demo_fractional_knapsack()
    demo_coin_change()
    demo_huffman()
    demo_kruskal_mst()


if __name__ == "__main__":
    main()
