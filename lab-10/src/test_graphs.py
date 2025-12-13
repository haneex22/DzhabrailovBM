from __future__ import annotations

import unittest

from graph_representation import GraphAdjacencyList, GraphAdjacencyMatrix
from graph_traversal import bfs, connected_components, dfs_iterative
from graph_traversal import dfs_recursive, reconstruct_path
from shortest_path import dijkstra, topo_sort_kahn
from tasks import Maze, find_dependencies_order, is_network_connected
from tasks import shortest_path_in_maze


class TestRepresentations(unittest.TestCase):
    def test_matrix_add_has_remove_edge(self) -> None:
        g = GraphAdjacencyMatrix(3, directed=False)
        g.add_edge(0, 1, 2.0)
        self.assertTrue(g.has_edge(0, 1))
        self.assertTrue(g.has_edge(1, 0))
        g.remove_edge(0, 1)
        self.assertFalse(g.has_edge(0, 1))
        self.assertFalse(g.has_edge(1, 0))

    def test_list_add_has_remove_edge(self) -> None:
        g = GraphAdjacencyList(3, directed=True)
        g.add_edge(0, 2, 5.0)
        self.assertTrue(g.has_edge(0, 2))
        self.assertFalse(g.has_edge(2, 0))
        g.remove_edge(0, 2)
        self.assertFalse(g.has_edge(0, 2))


class TestTraversal(unittest.TestCase):
    def setUp(self) -> None:
        self.g = GraphAdjacencyList(6, directed=False)
        for u, v in [(0, 1), (1, 2), (0, 3), (4, 5)]:
            self.g.add_edge(u, v, 1.0)

    def test_bfs_distances(self) -> None:
        res = bfs(self.g, 0)
        self.assertEqual(res.distances[0], 0)
        self.assertEqual(res.distances[2], 2)
        self.assertIsNone(res.distances[4])

    def test_reconstruct_path(self) -> None:
        res = bfs(self.g, 0)
        path = reconstruct_path(res.parent, 0, 2)
        self.assertEqual(path[0], 0)
        self.assertEqual(path[-1], 2)

    def test_dfs(self) -> None:
        r = dfs_recursive(self.g, 0)
        i = dfs_iterative(self.g, 0)
        self.assertIn(0, r)
        self.assertIn(0, i)

    def test_connected_components(self) -> None:
        comps = connected_components(self.g)
        sizes = sorted(len(c) for c in comps)
        self.assertEqual(sizes, [2, 4])


class TestAlgorithms(unittest.TestCase):
    def test_dijkstra(self) -> None:
        g = GraphAdjacencyList(5, directed=True)
        g.add_edge(0, 1, 2.0)
        g.add_edge(0, 2, 4.0)
        g.add_edge(1, 2, 1.0)
        g.add_edge(1, 3, 7.0)
        g.add_edge(2, 4, 3.0)

        res = dijkstra(g, 0)
        self.assertAlmostEqual(res.dist[2], 3.0, places=5)  # 0->1->2
        self.assertAlmostEqual(res.dist[4], 6.0, places=5)  # 0->1->2->4

    def test_toposort(self) -> None:
        dag = GraphAdjacencyList(6, directed=True)
        for u, v in [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]:
            dag.add_edge(u, v, 1.0)
        order = topo_sort_kahn(dag)
        self.assertEqual(len(order), 6)

    def test_toposort_cycle_raises(self) -> None:
        g = GraphAdjacencyList(2, directed=True)
        g.add_edge(0, 1, 1.0)
        g.add_edge(1, 0, 1.0)
        with self.assertRaises(ValueError):
            topo_sort_kahn(g)


class TestTasks(unittest.TestCase):
    def test_maze_shortest_path(self) -> None:
        maze = Maze(
            grid=[
                [0, 0, 1, 0],
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
            ],
        )
        path = shortest_path_in_maze(maze, (0, 0), (3, 3))
        self.assertTrue(path)
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (3, 3))

    def test_network_connected(self) -> None:
        self.assertTrue(is_network_connected(4, [(0, 1), (1, 2), (2, 3)]))
        self.assertFalse(is_network_connected(4, [(0, 1)]))

    def test_dependencies_order(self) -> None:
        order = find_dependencies_order(4, [(0, 1), (1, 2), (0, 3)])
        self.assertEqual(len(order), 4)


if __name__ == "__main__":
    unittest.main()
