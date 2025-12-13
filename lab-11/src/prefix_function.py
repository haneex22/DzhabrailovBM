from __future__ import annotations

from typing import List


def prefix_function(s: str) -> List[int]:
    """Вычисляет префикс-функцию (π-функцию) для строки s.

    π[i] — длина наибольшего собственного префикса строки s,
    являющегося суффиксом подстроки s[0..i].

    Временная сложность: O(n).
    Пространственная сложность: O(n).

    Args:
        s: Входная строка.

    Returns:
        Массив π длины len(s).
    """
    n = len(s)
    pi: List[int] = [0] * n

    j = 0
    for i in range(1, n):
        while j > 0 and s[i] != s[j]:
            j = pi[j - 1]

        if s[i] == s[j]:
            j += 1

        pi[i] = j

    return pi


def format_prefix_function(s: str, pi: List[int]) -> str:
    """Текстовая визуализация: строка + π-функция в виде таблицы.

    Временная сложность: O(n).
    """
    if len(s) != len(pi):
        raise ValueError("Длина строки и π-массива должна совпадать.")

    header = "Индексы: " + " ".join(f"{i:2d}" for i in range(len(s)))
    chars = "Символы: " + " ".join(f"{ch:2s}" for ch in s)
    values = "π(i):   " + " ".join(f"{v:2d}" for v in pi)
    return "\n".join([header, chars, values])
