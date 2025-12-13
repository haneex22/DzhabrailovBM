from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from z_function import z_function


SearchMethod = Literal["naive", "z", "rabin_karp"]


def naive_search_all(text: str, pattern: str) -> List[int]:
    """Наивный поиск всех вхождений pattern в text.

    Временная сложность: O(n * m) в худшем случае.
    Пространственная сложность: O(1) доп.памяти.
    """
    if pattern == "":
        return list(range(len(text) + 1))
    n = len(text)
    m = len(pattern)
    result: List[int] = []

    for i in range(0, n - m + 1):
        if text[i: i + m] == pattern:
            result.append(i)

    return result


def z_search_all(text: str, pattern: str) -> List[int]:
    """Поиск всех вхождений pattern в text через Z-функцию.

    Идея: строим строку pattern + '#' + text и считаем Z.
    Если z[i] == len(pattern) — найдено вхождение.

    Временная сложность: O(n + m).
    Пространственная сложность: O(n + m).
    """
    if pattern == "":
        return list(range(len(text) + 1))

    combined = pattern + "#" + text
    z = z_function(combined)

    m = len(pattern)
    result: List[int] = []
    offset = m + 1  # позиция начала текста в combined

    for i in range(offset, len(combined)):
        if z[i] == m:
            result.append(i - offset)

    return result


@dataclass(frozen=True)
class RabinKarpParams:
    """Параметры Рабина-Карпа.

    base: основание полиномиального хеша.
    mod:  модуль.
    """
    base: int = 911382323
    mod: int = 1_000_000_007


def rabin_karp_search_all(
    text: str,
    pattern: str,
    params: RabinKarpParams | None = None,
) -> List[int]:
    """Поиск всех вхождений Рабином-Карпом (rolling hash).

    В среднем быстро, но в худшем случае O(n*m) при большом числе коллизий.
    Здесь мы делаем "проверку подстроки" при совпадении хеша,
    чтобы гарантировать корректность.

    Временная сложность:
        Средняя близка к O(n + m), худшая O(n * m).
    Пространственная сложность:
        O(1) доп.памяти (кроме результата).
    """
    if pattern == "":
        return list(range(len(text) + 1))
    if params is None:
        params = RabinKarpParams()

    n = len(text)
    m = len(pattern)
    if m > n:
        return []

    base = params.base
    mod = params.mod

    # Предвычислим base^(m-1) mod mod
    high = 1
    for _ in range(m - 1):
        high = (high * base) % mod

    def char_code(ch: str) -> int:
        return ord(ch) + 1

    # Хеш pattern и первого окна text
    pat_hash = 0
    win_hash = 0
    for i in range(m):
        pat_hash = (pat_hash * base + char_code(pattern[i])) % mod
        win_hash = (win_hash * base + char_code(text[i])) % mod

    result: List[int] = []

    def check(pos: int) -> bool:
        return text[pos: pos + m] == pattern

    for i in range(0, n - m + 1):
        if win_hash == pat_hash and check(i):
            result.append(i)

        if i < n - m:
            left = char_code(text[i])
            right = char_code(text[i + m])

            # Удаляем левый символ и добавляем правый
            win_hash = (win_hash - left * high) % mod
            win_hash = (win_hash * base + right) % mod

    return result


def search_all(text: str, pattern: str, method: SearchMethod) -> List[int]:
    """Единая точка входа для поиска подстроки разными методами."""
    if method == "naive":
        return naive_search_all(text, pattern)
    if method == "z":
        return z_search_all(text, pattern)
    if method == "rabin_karp":
        return rabin_karp_search_all(text, pattern)
    raise ValueError(f"Неизвестный метод: {method!r}")
