from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Protocol

from graph_representation import Edge


class Graph(Protocol):
    """Протокол графа (общий интерфейс для двух представлений)."""

    def num_vertices(self) -> int: ...
    def neighbors(self, u: int) -> List[Edge]: ...


@dataclass(frozen=True)
class BFSResult:
    """Результат BFS.

    distances[v] = расстояние (число рёбер) от start до v или None.
    parent[v]    = предыдущая вершина на кратчайшем пути или None.
    """

    distances: List[Optional[int]]
    parent: List[Optional[int]]


def bfs(graph: Graph, start: int) -> BFSResult:
    """BFS (поиск в ширину) от вершины start.

    Возвращает расстояния и родителей для восстановления пути.

    Сложность:
        O(V + E), где V — вершины, E — рёбра (для списка смежности).
        Для матрицы смежности neighbors(u) стоит O(V),
        и общий обход становится O(V^2).
    """
    n = graph.num_vertices()
    if start < 0 or start >= n:
        raise IndexError('start вне диапазона.')

    distances: List[Optional[int]] = [None] * n
    parent: List[Optional[int]] = [None] * n

    q: Deque[int] = deque()
    distances[start] = 0
    q.append(start)

    while q:
        u = q.popleft()
        for e in graph.neighbors(u):  # суммарно O(E) (или O(V^2) для матрицы).
            v = e.to
            if distances[v] is None:
                distances[v] = distances[u] + 1
                parent[v] = u
                q.append(v)

    return BFSResult(distances=distances, parent=parent)


def reconstruct_path(parent: List[Optional[int]], start: int,
                     goal: int) -> List[int]:
    """Восстановление пути start->goal по массиву parent.

    Сложность: O(L), где L — длина пути.
    """
    if start == goal:
        return [start]

    path: List[int] = []
    cur: Optional[int] = goal
    while cur is not None:
        path.append(cur)
        if cur == start:
            break
        cur = parent[cur]

    if not path or path[-1] != start:
        return []  # пути нет

    path.reverse()
    return path


def dfs_recursive(graph: Graph, start: int) -> List[int]:
    """DFS рекурсивный. Возвращает порядок посещения.

    Сложность:
        O(V + E) (или O(V^2) для матрицы из-за neighbors(u)).
    """
    n = graph.num_vertices()
    if start < 0 or start >= n:
        raise IndexError('start вне диапазона.')

    visited = [False] * n
    order: List[int] = []

    def visit(u: int) -> None:
        visited[u] = True
        order.append(u)
        for e in graph.neighbors(u):
            if not visited[e.to]:
                visit(e.to)

    visit(start)
    return order


def dfs_iterative(graph: Graph, start: int) -> List[int]:
    """DFS итеративный (стек). Возвращает порядок посещения.

    Сложность:
        O(V + E) (или O(V^2) для матрицы из-за neighbors(u)).
    """
    n = graph.num_vertices()
    if start < 0 or start >= n:
        raise IndexError('start вне диапазона.')

    visited = [False] * n
    order: List[int] = []
    stack: List[int] = [start]

    while stack:
        u = stack.pop()
        if visited[u]:
            continue
        visited[u] = True
        order.append(u)

        # Чтобы порядок был похож на рекурсивный,
        # кладём соседей в обратном порядке.
        neigh = graph.neighbors(u)
        for e in reversed(neigh):
            if not visited[e.to]:
                stack.append(e.to)

    return order


def connected_components(graph: Graph) -> List[List[int]]:
    """Поиск компонент связности (для НЕориентированного графа).

    Реализовано через BFS по каждой ещё не посещённой вершине.

    Сложность:
        O(V + E) (или O(V^2) для матрицы).
    """
    n = graph.num_vertices()
    visited = [False] * n
    components: List[List[int]] = []

    for s in range(n):
        if visited[s]:
            continue

        q: Deque[int] = deque([s])
        visited[s] = True
        comp: List[int] = []

        while q:
            u = q.popleft()
            comp.append(u)
            for e in graph.neighbors(u):
                v = e.to
                if not visited[v]:
                    visited[v] = True
                    q.append(v)

        components.append(comp)

    return components
