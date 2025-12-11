from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Interval:
    """Интервал задачи для задачи о выборе заявок.

    Атрибуты:
        start: время начала интервала.
        end:   время окончания интервала (должно быть >= start).
    """

    start: float
    end: float


@dataclass(frozen=True)
class Item:
    """Предмет для задачи о рюкзаке.

    Атрибуты:
        weight: вес предмета (> 0).
        value:  стоимость предмета (>= 0).
    """

    weight: float
    value: float


@dataclass
class HuffmanNode:
    """Вершина дерева Хаффмана.

    Листовая вершина имеет символ `symbol` и None в детях.
    Внутренняя вершина имеет `symbol is None` и двух потомков.
    """

    freq: int
    symbol: str | None = None
    left: HuffmanNode | None = None
    right: HuffmanNode | None = None

    def __lt__(self, other: "HuffmanNode") -> bool:
        """Сравнение по частоте для использования в heapq. O(1)."""
        return self.freq < other.freq


@dataclass(frozen=True)
class Edge:
    """Ребро неориентированного взвешенного графа для MST."""

    u: int
    v: int
    weight: float


def select_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    """Жадный выбор максимального набора непересекающихся интервалов.

    Стратегия:
    1. Отсортировать интервалы по времени окончания по возрастанию.
    2. Идти по списку и выбирать каждый интервал, который начинается
       не раньше, чем закончился предыдущий выбранный.

    Это классический жадный алгоритм для задачи Interval Scheduling.

    Сложность:
        O(n log n) на сортировку + O(n) на проход = O(n log n),
        где n — число интервалов.
    """
    sorted_intervals = sorted(intervals, key=lambda it: it.end)
    result: List[Interval] = []
    current_end = float("-inf")

    for interval in sorted_intervals:
        if interval.start >= current_end:
            result.append(interval)
            current_end = interval.end

    return result


def fractional_knapsack(
    capacity: float,
    items: Iterable[Item],
) -> Tuple[float, List[Tuple[Item, float]]]:
    """Жадный алгоритм для непрерывной (дробной) задачи о рюкзаке.

    Стратегия:
    1. Для каждого предмета считаем удельную стоимость value/weight.
    2. Сортируем предметы по удельной стоимости по убыванию.
    3. Заполняем рюкзак, каждый раз беря максимальную возможную часть
       текущего лучшего предмета, пока есть свободная емкость.

    Алгоритм оптимален только для дробного рюкзака, когда разрешено
    брать части предметов.

    Аргументы:
        capacity: максимальный вес рюкзака (> 0).
        items:    итерируемая коллекция предметов.

    Возвращает:
        Кортеж (total_value, selection), где selection — список пар
        (item, fraction) с 0 < fraction <= 1.

    Сложность:
        O(n log n) на сортировку + O(n) на один проход = O(n log n),
        где n — число предметов.
    """
    if capacity <= 0:
        return 0.0, []

    # Строим список с удельной стоимостью. O(n).
    items_with_ratio: List[Tuple[float, Item]] = [
        (it.value / it.weight, it) for it in items if it.weight > 0
    ]

    # Сортировка по убыванию удельной стоимости. O(n log n).
    items_with_ratio.sort(key=lambda pair: pair[0], reverse=True)

    remaining = capacity
    total_value = 0.0
    selection: List[Tuple[Item, float]] = []

    for ratio, item in items_with_ratio:
        if remaining <= 0:
            break

        take_weight = min(item.weight, remaining)
        fraction = take_weight / item.weight
        total_value += item.value * fraction
        remaining -= take_weight
        selection.append((item, fraction))

    return total_value, selection


def build_huffman_tree(frequencies: Dict[str, int]) -> HuffmanNode:
    """Строит дерево Хаффмана по словарю частот символов.

    Используется жадная стратегия:
    на каждом шаге объединяем два наименее частых дерева.

    Сложность:
        Пусть k — число различных символов.
        Инициализация кучи: O(k).
        Каждое извлечение/добавление — O(log k), выполняется O(k) раз.
        Итого: O(k log k).
    """
    heap: List[HuffmanNode] = []

    for symbol, freq in frequencies.items():
        heappush(heap, HuffmanNode(freq=freq, symbol=symbol))

    if not heap:
        raise ValueError("Пустой словарь частот.")

    # Если символ один, всё равно построим дерево с корнем-одиночкой.
    while len(heap) > 1:
        first = heappop(heap)
        second = heappop(heap)

        merged = HuffmanNode(
            freq=first.freq + second.freq,
            left=first,
            right=second,
        )  # O(1).
        heappush(heap, merged)

    return heap[0]


def _build_codes_recursive(
    node: HuffmanNode,
    prefix: str,
    codes: Dict[str, str],
) -> None:
    """Вспомогательная рекурсивная функция для построения кодов. O(k)."""
    if node.symbol is not None and node.left is None and node.right is None:
        # Листовая вершина. O(1).
        codes[node.symbol] = prefix or "0"  # Не допускаем пустого кода.
        return

    if node.left is not None:
        _build_codes_recursive(node.left, prefix + "0", codes)
    if node.right is not None:
        _build_codes_recursive(node.right, prefix + "1", codes)


def build_huffman_codes(frequencies: Dict[str, int]) -> Dict[str, str]:
    """Строит оптимальные префиксные коды Хаффмана для символов.

    Сложность:
        Строительство дерева: O(k log k),
        Обход дерева и генерация кодов: O(k),
        Итого: O(k log k).
    """
    root = build_huffman_tree(frequencies)
    codes: Dict[str, str] = {}
    _build_codes_recursive(root, "", codes)
    return codes


def huffman_encode(text: str) -> Tuple[str, Dict[str, str], HuffmanNode]:
    """Кодирует строку с помощью кода Хаффмана.

    Возвращает кортеж:
        (encoded_bits, codes, tree_root).

    Сложность:
        Пусть n — длина текста, k — размер алфавита.
        Подсчёт частот: O(n).
        Построение кодов и дерева: O(k log k).
        Кодирование: O(n).
        Итого: O(n + k log k).
    """
    if not text:
        raise ValueError("Нельзя кодировать пустую строку.")

    frequencies: Dict[str, int] = {}
    for ch in text:
        frequencies[ch] = frequencies.get(ch, 0) + 1

    codes = build_huffman_codes(frequencies)
    root = build_huffman_tree(frequencies)

    encoded_bits = "".join(codes[ch] for ch in text)

    return encoded_bits, codes, root


def huffman_decode(encoded_bits: str, root: HuffmanNode) -> str:
    """Декодирует строку битов, используя дерево Хаффмана.

    Сложность:
        Пусть n — длина закодированной последовательности.
        Один проход по битам, каждый шаг — движение по дереву.
        Итого: O(n).
    """
    if not encoded_bits:
        return ""

    result_chars: List[str] = []
    node = root

    for bit in encoded_bits:
        if bit == "0":
            if node.left is None:
                raise ValueError("Некорректный код Хаффмана.")
            node = node.left
        elif bit == "1":
            if node.right is None:
                raise ValueError("Некорректный код Хаффмана.")
            node = node.right
        else:
            raise ValueError(f"Недопустимый символ в коде: {bit!r}")

        if node.symbol is not None:
            result_chars.append(node.symbol)
            node = root

    return "".join(result_chars)


def coin_change_greedy(amount: int, denominations: Iterable[int]) -> Dict[int,
                                                                          int]:
    """Жадная выдача суммы минимальным количеством монет.

    Стратегия:
        1. Сортируем номиналы монет по убыванию.
        2. Для каждого номинала берём максимально возможное количество монет
           этого номинала, затем переходим к следующему.

    Для стандартных "канонических" систем монет (например, 1, 5, 10, 25)
    алгоритм оптимален, но в общем случае может быть неоптимальным.

    Сложность:
        Пусть k — число разных номиналов.
        Сортировка: O(k log k), проход по номиналам: O(k).
        Итого: O(k log k).
    """
    if amount < 0:
        raise ValueError("Сумма не может быть отрицательной.")

    sorted_denoms = sorted(denominations, reverse=True)
    result: Dict[int, int] = {}
    remaining = amount

    for coin in sorted_denoms:
        if coin <= 0:
            continue
        count = remaining // coin
        if count > 0:
            result[coin] = count
            remaining -= coin * count

    if remaining != 0:
        # В данной системе монет нельзя набрать точную сумму.
        raise ValueError("Нельзя набрать сумму данной системой монет.")

    return result


class DisjointSet:
    """Система непересекающихся множеств (union-find).

    Используется в алгоритме Краскала для построения MST.

    Амортизированная сложность операций:
        find:  O(alpha(n)),
        union: O(alpha(n)),
    где alpha — обратная функция Аккермана (растёт очень медленно,
    практически можно считать её константой).
    """

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("Размер должен быть положительным.")
        self.parent: List[int] = list(range(size))
        self.rank: List[int] = [0] * size

    def find(self, x: int) -> int:
        """Находит представителя множества с путевой компрессией.O(alpha(n))"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        """Объединяет два множества по рангу. O(alpha(n))."""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def kruskal_mst(num_vertices: int, edges: Iterable[Edge]) -> Tuple[float,
                                                                   List[Edge]]:
    """Жадный алгоритм Краскала для поиска минимального остовного дерева.

    Стратегия:
        1. Отсортировать все рёбра по весу по возрастанию.
        2. Идти по рёбрам и добавлять ребро в остов, если оно не образует цикл,
           что проверяется с помощью структуры DisjointSet.

    Аргументы:
        num_vertices: число вершин в графе (вершины нумеруются от 0 до n-1).
        edges:       итерируемая коллекция рёбер.

    Возвращает:
        (total_weight, mst_edges) — суммарный вес и список рёбер остова.

    Сложность:
        Пусть E — количество рёбер.
        Сортировка рёбер: O(E log E).
        Каждая операция union/find — амортизированно O(alpha(V)).
        Итого: O(E log E).
    """
    edge_list = list(edges)
    if num_vertices <= 0:
        raise ValueError("Граф должен содержать хотя бы одну вершину.")

    # Сортировка рёбер по весу. O(E log E).
    edge_list.sort(key=lambda e: e.weight)

    dsu = DisjointSet(num_vertices)
    mst_edges: List[Edge] = []
    total_weight = 0.0

    for edge in edge_list:
        root_u = dsu.find(edge.u)
        root_v = dsu.find(edge.v)
        if root_u != root_v:
            mst_edges.append(edge)
            total_weight += edge.weight
            dsu.union(root_u, root_v)
            if len(mst_edges) == num_vertices - 1:
                break

    if len(mst_edges) != num_vertices - 1:
        raise ValueError("Граф должен быть связным для построения MST.")

    return total_weight, mst_edges
