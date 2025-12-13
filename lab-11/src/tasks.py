from __future__ import annotations

from typing import List

from kmp_search import kmp_search_all
from prefix_function import prefix_function
from string_matching import SearchMethod, search_all
from z_function import z_function


def task_find_all_occurrences(text: str, pattern: str,
                              method: SearchMethod) -> List[int]:
    """Практическая задача 1: найти все вхождения pattern в text алгоритмом."""
    if method == "z" or method == "rabin_karp" or method == "naive":
        return search_all(text, pattern, method)
    return kmp_search_all(text, pattern)


def smallest_period_prefix(s: str) -> int:
    """Практическая задача 2: найти период строки (через префикс-функцию).

    Если строка s состоит из повторения некоторой подстроки p,
    то длина периода = n - π[n-1], и n % period == 0.

    Временная сложность: O(n).
    """
    n = len(s)
    if n == 0:
        return 0

    pi = prefix_function(s)
    period = n - pi[n - 1]

    if period != 0 and n % period == 0:
        return period

    return n


def smallest_period_z(s: str) -> int:
    """Практическая задача 2 (альтернатива): период строки через Z-функцию.

    Ищем минимальное k такое, что:
        k | n и z[k] >= n - k.
    Временная сложность: O(n) + O(n) проверки => O(n).
    """
    n = len(s)
    if n == 0:
        return 0

    z = z_function(s)
    for k in range(1, n):
        if n % k == 0 and z[k] >= n - k:
            return k
    return n


def is_cyclic_shift(a: str, b: str, method: SearchMethod = "z") -> bool:
    """Практическая задача 3: проверка циклического сдвига.

    a и b — циклические сдвиги тогда и только тогда, когда:
        len(a) == len(b) и a встречается в (b + b).

    Временная сложность:
        зависит от выбранного метода поиска.
    """
    if len(a) != len(b):
        return False
    if a == "":
        return True

    doubled = b + b
    occurrences = search_all(doubled, a, method)  # зависит от метода.
    return len(occurrences) > 0
