from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Edge:
    """Ребро графа.

    Атрибуты:
        to: конечная вершина.
        weight: вес ребра (для невзвешенного графа можно считать 1.0).
    """

    to: int
    weight: float


class GraphAdjacencyMatrix:
    """Граф на матрице смежности.

    Поддерживает ориентированный/неориентированный и взвешенный граф.
    Отсутствие ребра — None в матрице.

    Память:
        O(V^2), где V — число вершин.
    """

    def __init__(self, num_vertices: int, directed: bool = False) -> None:
        if num_vertices <= 0:
            raise ValueError('num_vertices должно быть положительным.')
        self.directed = directed
        self._n = num_vertices
        self._matrix: List[List[Optional[float]]] = [
            [None] * num_vertices for _ in range(num_vertices)
        ]  # O(V^2).

    def num_vertices(self) -> int:
        """Возвращает число вершин. O(1)."""
        return self._n

    def add_vertex(self) -> int:
        """Добавляет вершину и возвращает её индекс.

        Сложность:
            O(V) на расширение каждой строки + O(V) на новую строку => O(V).
        """
        for row in self._matrix:
            row.append(None)
        self._n += 1
        self._matrix.append([None] * self._n)
        return self._n - 1

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """Добавляет ребро u->v (и v->u если граф неориентированный). O(1)."""
        self._validate_vertex(u)
        self._validate_vertex(v)
        self._matrix[u][v] = float(weight)
        if not self.directed:
            self._matrix[v][u] = float(weight)

    def remove_edge(self, u: int, v: int) -> None:
        """Удаляет ребро u->v (и v->u если граф неориентированный). O(1)."""
        self._validate_vertex(u)
        self._validate_vertex(v)
        self._matrix[u][v] = None
        if not self.directed:
            self._matrix[v][u] = None

    def has_edge(self, u: int, v: int) -> bool:
        """Проверка наличия ребра. O(1)."""
        self._validate_vertex(u)
        self._validate_vertex(v)
        return self._matrix[u][v] is not None

    def neighbors(self, u: int) -> List[Edge]:
        """Возвращает список соседей вершины u.

        Сложность:
            O(V), т.к. нужно пройти всю строку матрицы.
        """
        self._validate_vertex(u)
        result: List[Edge] = []
        for v, w in enumerate(self._matrix[u]):
            if w is not None:
                result.append(Edge(to=v, weight=w))
        return result

    def memory_cells(self) -> int:
        """Оценка памяти как число ячеек матрицы. O(1)."""
        return self._n * self._n

    def _validate_vertex(self, v: int) -> None:
        """Проверка допустимости индекса вершины. O(1)."""
        if v < 0 or v >= self._n:
            raise IndexError('Вершина вне диапазона.')


class GraphAdjacencyList:
    """Граф на списках смежности.

    adj[u] = список рёбер из u.

    Память:
        O(V + E), где E — число рёбер.

    Преимущества:
        - Быстрый обход соседей за O(deg(u)).
        - Эффективно по памяти на разреженных графах.

    Недостатки:
        - Проверка наличия ребра u->v может быть O(deg(u)).
    """

    def __init__(self, num_vertices: int, directed: bool = False) -> None:
        if num_vertices <= 0:
            raise ValueError('num_vertices должно быть положительным.')
        self.directed = directed
        self._adj: List[List[Edge]] = [[] for _ in range(num_vertices)]

    def num_vertices(self) -> int:
        """Возвращает число вершин. O(1)."""
        return len(self._adj)

    def add_vertex(self) -> int:
        """Добавляет вершину, возвращает её индекс. O(1) амортизированно."""
        self._adj.append([])
        return self.num_vertices() - 1

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """Добавляет ребро u->v (и v->u если неориентированный). O(deg(u))."""
        self._validate_vertex(u)
        self._validate_vertex(v)
        self._add_or_replace(u, v, float(weight))
        if not self.directed:
            self._add_or_replace(v, u, float(weight))

    def remove_edge(self, u: int, v: int) -> None:
        """Удаляет ребро u->v (и v->u если неориентированный). O(deg(u))."""
        self._validate_vertex(u)
        self._validate_vertex(v)
        self._adj[u] = [e for e in self._adj[u] if e.to != v]
        if not self.directed:
            self._adj[v] = [e for e in self._adj[v] if e.to != u]

    def has_edge(self, u: int, v: int) -> bool:
        """Проверка наличия ребра. O(deg(u))."""
        self._validate_vertex(u)
        self._validate_vertex(v)
        return any(e.to == v for e in self._adj[u])

    def neighbors(self, u: int) -> List[Edge]:
        """Соседи вершины u. O(deg(u)) на копирование списка."""
        self._validate_vertex(u)
        return list(self._adj[u])

    def edges_count(self) -> int:
        """Возвращает число ориентированных рёбер (внутреннее). O(V + E)."""
        return sum(len(lst) for lst in self._adj)

    def memory_units(self) -> int:
        """Грубая оценка памяти: V + E(ориент). O(V + E)."""
        v = self.num_vertices()
        e = self.edges_count()
        return v + e

    def _add_or_replace(self, u: int, v: int, weight: float) -> None:
        """Добавляет ребро или обновляет вес, если ребро уже было O(deg(u))."""
        for i, e in enumerate(self._adj[u]):
            if e.to == v:
                self._adj[u][i] = Edge(to=v, weight=weight)
                return
        self._adj[u].append(Edge(to=v, weight=weight))

    def _validate_vertex(self, v: int) -> None:
        """Проверка допустимости индекса вершины. O(1)."""
        if v < 0 or v >= self.num_vertices():
            raise IndexError('Вершина вне диапазона.')
