from __future__ import annotations

from typing import List

from prefix_function import prefix_function


def kmp_search_all(text: str, pattern: str) -> List[int]:
    """Находит все вхождения pattern в text алгоритмом Кнута-Морриса-Пратта.

    Временная сложность: O(n + m).
    Пространственная сложность: O(m).

    Args:
        text: Текст, где ищем.
        pattern: Паттерн (подстрока).

    Returns:
        Список индексов начала каждого вхождения.
    """
    if pattern == "":
        return list(range(len(text) + 1))

    m = len(pattern)
    pi = prefix_function(pattern)

    result: List[int] = []
    j = 0  # длина совпавшего префикса pattern

    for i in range(len(text)):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]

        if text[i] == pattern[j]:
            j += 1

        if j == m:
            result.append(i - m + 1)
            j = pi[j - 1]

    return result
