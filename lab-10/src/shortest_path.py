from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

from graph_representation import Edge


class Graph(Protocol):
    def num_vertices(self) -> int: ...
    def neighbors(self, u: int) -> List[Edge]: ...


@dataclass(frozen=True)
class DijkstraResult:
    """Результат Дейкстры.

    dist[v]   = минимальная стоимость пути от start до v или None.
    parent[v] = предок для восстановления пути.
    """

    dist: List[Optional[float]]
    parent: List[Optional[int]]


def dijkstra(graph: Graph, start: int) -> DijkstraResult:
    """Алгоритм Дейкстры (без отрицательных весов).

    Сложность:
        Для списков смежности: O((V + E) log V) через heapq.
        Для матрицы смежности (neighbors O(V)): O(V^2 log V)
        по факту вызовов neighbors.
    """
    n = graph.num_vertices()
    if start < 0 or start >= n:
        raise IndexError('start вне диапазона.')

    dist: List[Optional[float]] = [None] * n
    parent: List[Optional[int]] = [None] * n

    dist[start] = 0.0
    pq: List[Tuple[float, int]] = [(0.0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if dist[u] is not None and d > dist[u]:
            continue

        for e in graph.neighbors(u):
            if e.weight < 0:
                raise ValueError('Дейкстра не работает отрицательными весами')
            v = e.to
            nd = d + e.weight
            if dist[v] is None or nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return DijkstraResult(dist=dist, parent=parent)


def topo_sort_kahn(graph: Graph) -> List[int]:
    """Топологическая сортировка (алгоритм Кана) для DAG.

    Возвращает порядок вершин. Если есть цикл — возбуждает ValueError.

    Сложность:
        O(V + E) (или O(V^2) для матрицы из-за neighbors).
    """
    n = graph.num_vertices()
    indeg = [0] * n

    for u in range(n):
        for e in graph.neighbors(u):
            indeg[e.to] += 1

    queue: List[int] = [i for i in range(n) if indeg[i] == 0]
    head = 0
    order: List[int] = []

    while head < len(queue):
        u = queue[head]
        head += 1
        order.append(u)

        for e in graph.neighbors(u):
            v = e.to
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    if len(order) != n:
        raise ValueError('Граф содержит цикл, топосорт невозможен.')

    return order
