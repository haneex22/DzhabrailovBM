from __future__ import annotations

import random
import timeit
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt

from graph_representation import GraphAdjacencyList, GraphAdjacencyMatrix
from graph_traversal import bfs, dfs_iterative
from shortest_path import dijkstra


PC_INFO = """\
Характеристики ПК для тестирования:
- Процессор: Intel Core i5-11400 @ 2.60GHz
- Оперативная память: 16 GB
- ОС: Windows 10 x64
- Python: 3.13.3
"""


def measure_ms(func: Callable[[], None], repeats: int = 3) -> float:
    """Среднее время выполнения func в миллисекундах. O(repeats * T(func))."""
    t = timeit.Timer(func).timeit(number=repeats)  # O(repeats * T).
    return (t / repeats) * 1000.0


def build_random_graph_edges(v: int, e: int,
                             directed: bool) -> List[Tuple[int, int]]:
    """Генерирует список рёбер случайного графа. O(E)."""
    edges: List[Tuple[int, int]] = []
    for _ in range(e):
        u = random.randrange(v)
        w = random.randrange(v)
        if u == w:
            continue
        edges.append((u, w))
        if not directed:
            edges.append((w, u))
    return edges


def experiment_representation_ops() -> None:
    """Сравнение операций/обходов для матрицы и списка смежности.

    Строим графики времени BFS/DFS на разных V.
    """
    print("=" * 80)
    print("Эксперимент: сравнение представлений графа (BFS/DFS)")
    print("=" * 80)

    sizes = [50, 100, 200, 400, 800]
    bfs_list_ms: List[float] = []
    bfs_mat_ms: List[float] = []
    dfs_list_ms: List[float] = []
    dfs_mat_ms: List[float] = []

    for v in sizes:
        e = v * 3  # плотность ~ 3 ребра на вершину
        edges = build_random_graph_edges(v, e, directed=False)

        g_list = GraphAdjacencyList(v, directed=False)
        g_mat = GraphAdjacencyMatrix(v, directed=False)

        for u, w in edges:
            g_list.add_edge(u, w, 1.0)
            g_mat.add_edge(u, w, 1.0)

        bfs_list_ms.append(measure_ms(lambda: bfs(g_list, 0)))
        bfs_mat_ms.append(measure_ms(lambda: bfs(g_mat, 0)))

        dfs_list_ms.append(measure_ms(lambda: dfs_iterative(g_list, 0)))
        dfs_mat_ms.append(measure_ms(lambda: dfs_iterative(g_mat, 0)))

        print(f"V={v:4d}: bfs_list={bfs_list_ms[-1]:8.3f} ms, bfs_mat={bfs_mat_ms[-1]:8.3f} ms")

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, bfs_list_ms, marker="o", label="BFS list")
    plt.plot(sizes, bfs_mat_ms, marker="s", label="BFS matrix")
    plt.xlabel("V")
    plt.ylabel("Время, ms")
    plt.title("BFS: список смежности vs матрица смежности")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bfs_representation_scaling.png", dpi=200)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, dfs_list_ms, marker="o", label="DFS list")
    plt.plot(sizes, dfs_mat_ms, marker="s", label="DFS matrix")
    plt.xlabel("V")
    plt.ylabel("Время, ms")
    plt.title("DFS (итеративный): список смежности vs матрица смежности")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("dfs_representation_scaling.png", dpi=200)
    plt.show()


def experiment_dijkstra_scaling() -> None:
    """Масштабируемость Дейкстры на разреженных графах (список смежности)."""
    print("=" * 80)
    print("Эксперимент: Дейкстра (масштабируемость)")
    print("=" * 80)

    sizes = [200, 400, 800, 1200, 1600]
    times: List[float] = []

    for v in sizes:
        e = v * 4
        g = GraphAdjacencyList(v, directed=True)
        for _ in range(e):
            u = random.randrange(v)
            w = random.randrange(v)
            if u == w:
                continue
            weight = float(random.randint(1, 10))
            g.add_edge(u, w, weight)

        t = measure_ms(lambda: dijkstra(g, 0))
        times.append(t)
        print(f"V={v:4d}, E≈{e:6d} => {t:8.3f} ms")

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, times, marker="o")
    plt.xlabel("V")
    plt.ylabel("Время, ms")
    plt.title("Дейкстра: рост времени при увеличении V (разреженный граф)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("dijkstra_scaling.png", dpi=200)
    plt.show()


def main() -> None:
    print(PC_INFO)
    experiment_representation_ops()
    experiment_dijkstra_scaling()


if __name__ == "__main__":
    main()
