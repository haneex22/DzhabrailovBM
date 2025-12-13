from __future__ import annotations

from typing import List, Literal


def z_function(s: str) -> List[int]:
    """Вычисляет Z-функцию строки s.

    z[i] — длина наибольшего общего префикса строки s и суффикса s[i..].

    Временная сложность: O(n).
    Пространственная сложность: O(n).
    """
    n = len(s)
    z: List[int] = [0] * n
    l: Literal[0]
    r = 0

    for i in range(1, n):
        if i <= r:
            z[i] = min(r - i + 1, z[i - l])

        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1

        if i + z[i] - 1 > r:
            l = i
            r = i + z[i] - 1

    return z


def format_z_function(s: str, z: List[int]) -> str:
    """Текстовая визуализация: строка + Z-функция в виде таблицы.

    Временная сложность: O(n).
    """
    if len(s) != len(z):
        raise ValueError("Длина строки и Z-массива должна совпадать.")

    header = "Индексы: " + " ".join(f"{i:2d}" for i in range(len(s)))
    chars = "Символы: " + " ".join(f"{ch:2s}" for ch in s)
    values = "z(i):   " + " ".join(f"{v:2d}" for v in z)
    return "\n".join([header, chars, values])
