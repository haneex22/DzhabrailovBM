from __future__ import annotations

import unittest

from kmp_search import kmp_search_all
from prefix_function import prefix_function
from string_matching import naive_search_all, rabin_karp_search_all
from string_matching import z_search_all
from tasks import is_cyclic_shift, smallest_period_prefix, smallest_period_z
from z_function import z_function


class TestPrefixFunction(unittest.TestCase):
    def test_prefix_basic(self) -> None:
        self.assertEqual(prefix_function("abacaba"), [0, 0, 1, 0, 1, 2, 3])
        self.assertEqual(prefix_function("aaaa"), [0, 1, 2, 3])
        self.assertEqual(prefix_function(""), [])

    def test_prefix_single(self) -> None:
        self.assertEqual(prefix_function("x"), [0])


class TestZFunction(unittest.TestCase):
    def test_z_basic(self) -> None:
        self.assertEqual(z_function("aaaa"), [0, 3, 2, 1])
        self.assertEqual(z_function("abacaba"), [0, 0, 1, 0, 3, 0, 1])
        self.assertEqual(z_function(""), [])

    def test_z_single(self) -> None:
        self.assertEqual(z_function("x"), [0])


class TestSearchAlgorithms(unittest.TestCase):
    def _check_all_equal(self, text: str, pattern: str) -> None:
        expected = naive_search_all(text, pattern)
        self.assertEqual(expected, z_search_all(text, pattern))
        self.assertEqual(expected, rabin_karp_search_all(text, pattern))
        self.assertEqual(expected, kmp_search_all(text, pattern))

    def test_search_common(self) -> None:
        self._check_all_equal("abracadabra", "abra")
        self._check_all_equal("aaaaa", "aa")
        self._check_all_equal("abcdef", "gh")
        self._check_all_equal("", "")
        self._check_all_equal("abc", "")

    def test_search_pattern_longer(self) -> None:
        self._check_all_equal("abc", "abcd")


class TestTasks(unittest.TestCase):
    def test_period(self) -> None:
        s = "abcabcabcabc"
        self.assertEqual(3, smallest_period_prefix(s))
        self.assertEqual(3, smallest_period_z(s))

        s2 = "aaaaaa"
        self.assertEqual(1, smallest_period_prefix(s2))
        self.assertEqual(1, smallest_period_z(s2))

        s3 = "abac"
        self.assertEqual(len(s3), smallest_period_prefix(s3))
        self.assertEqual(len(s3), smallest_period_z(s3))

    def test_cyclic_shift(self) -> None:
        self.assertTrue(is_cyclic_shift("abcd", "cdab"))
        self.assertFalse(is_cyclic_shift("abcd", "acbd"))
        self.assertTrue(is_cyclic_shift("", ""))


if __name__ == "__main__":
    unittest.main()
