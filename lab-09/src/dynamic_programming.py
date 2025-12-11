from __future__ import annotations


from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class KnapsackInstance:
    """Описание экземпляра задачи о рюкзаке 0-1.

    attributes:
        values: стоимости предметов.
        weights: веса предметов.
        capacity: вместимость рюкзака.

    Один и тот же индекс i соответствует одному предмету.

    Сложность хранения: O(n),
    где n = len(values) = len(weights).
    """
    values: List[int]
    weights: List[int]
    capacity: int


def fib_recursive_naive(n: int) -> int:
    """Наивный рекурсивный алгоритм вычисления чисел Фибоначчи.

    F(0) = 0, F(1) = 1,
    F(n) = F(n-1) + F(n-2) для n >= 2.

    Временная сложность:
        Экспоненциальная, примерно O(phi^n),
        где phi ≈ 1.618 — золотое сечение.
    Пространственная сложность:
        O(n) по стеку вызовов.
    """
    if n < 0:
        raise ValueError("n должно быть неотрицательным")

    if n == 0:
        return 0

    if n == 1:
        return 1

    return fib_recursive_naive(n - 1) + fib_recursive_naive(n - 2)


def fib_memoized(n: int, cache: Optional[Dict[int, int]] = None) -> int:
    """Рекурсивное вычисление Фибоначчи с мемоизацией (top-down ДП).

    cache:
        словарь для сохранения уже вычисленных значений.

    Временная сложность:
        O(n), каждая F(k) вычисляется ровно один раз.
    Пространственная сложность:
        O(n) для словаря и стека рекурсии.
    """
    if n < 0:
        raise ValueError("n должно быть неотрицательным")

    if cache is None:
        cache = {}

    if n in cache:
        return cache[n]

    if n == 0:
        cache[0] = 0
    elif n == 1:
        cache[1] = 1
    else:
        cache[n] = fib_memoized(n - 1, cache) + fib_memoized(n - 2, cache)

    return cache[n]


def fib_bottom_up(n: int) -> int:
    """Итеративное табличное вычисление Фибоначчи (bottom-up ДП).

    Временная сложность:
        O(n).
    Пространственная сложность:
        O(1) дополнительной памяти (храним только два последних значения).
    """
    if n < 0:
        raise ValueError("n должно быть неотрицательным")

    if n == 0:
        return 0
    if n == 1:
        return 1

    prev2 = 0  # F(0)
    prev1 = 1  # F(1)

    for _ in range(2, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current

    return prev1


def knapsack_01_dp(instance: KnapsackInstance) -> int:
    """Рюкзак 0-1: классический bottom-up ДП без восстановления решения.

    dp[i][w] — максимальная стоимость, которую можно получить,
    рассматривая первые i предметов и имея вместимость w.

    Временная сложность:
        Пусть n — число предметов, W — capacity.
        O(n * W).
    Пространственная сложность:
        O(n * W) для таблицы dp.
    """
    values = instance.values
    weights = instance.weights
    capacity = instance.capacity

    if len(values) != len(weights):
        raise ValueError("values и weights должны иметь одинаковую длину")

    n = len(values)

    dp: List[List[int]] = [
        [0] * (capacity + 1) for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        value = values[i - 1]
        weight = weights[i - 1]
        for w in range(capacity + 1):
            without_item = dp[i - 1][w]
            if weight <= w:
                with_item = value + dp[i - 1][w - weight]
                dp[i][w] = max(without_item, with_item)
            else:
                dp[i][w] = without_item

    return dp[n][capacity]


def knapsack_01_dp_with_items(
    instance: KnapsackInstance,
) -> Tuple[int, List[int]]:
    """Рюкзак 0-1: bottom-up ДП с восстановлением набора предметов.

    Возвращает:
        (max_value, chosen_indices),
        где chosen_indices — индексы выбранных предметов (0-based).

    Временная сложность:
        O(n * W) — построение таблицы + O(n) восстановление.
    Пространственная сложность:
        O(n * W).
    """
    values = instance.values
    weights = instance.weights
    capacity = instance.capacity

    if len(values) != len(weights):
        raise ValueError("values и weights должны иметь одинаковую длину")

    n = len(values)

    dp: List[List[int]] = [
        [0] * (capacity + 1) for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        value = values[i - 1]
        weight = weights[i - 1]
        for w in range(capacity + 1):
            without_item = dp[i - 1][w]
            if weight <= w:
                with_item = value + dp[i - 1][w - weight]
                dp[i][w] = max(without_item, with_item)
            else:
                dp[i][w] = without_item

    max_value = dp[n][capacity]
    chosen_indices: List[int] = []

    i = n
    w = capacity
    while i > 0 and w >= 0:
        if dp[i][w] == dp[i - 1][w]:
            i -= 1
        else:
            index = i - 1
            chosen_indices.append(index)
            w -= weights[index]
            i -= 1

    chosen_indices.reverse()
    return max_value, chosen_indices


def lcs_length_bottom_up(a: str, b: str) -> int:
    """Длина LCS (Longest Common Subsequence) двух строк.

    dp[i][j] — длина LCS для префиксов a[:i] и b[:j].

    Временная сложность:
        O(len(a) * len(b)).
    Пространственная сложность:
        O(len(a) * len(b)).
    """
    n = len(a)
    m = len(b)

    dp: List[List[int]] = [
        [0] * (m + 1) for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[n][m]


def lcs_reconstruct(a: str, b: str) -> str:
    """Восстановление одной LCS для строк a и b.

    Сначала считаем табличку dp (как в lcs_length_bottom_up),
    затем идём от dp[n][m] к dp[0][0] и собираем символы.

    Временная сложность:
        O(len(a) * len(b)).
    Пространственная сложность:
        O(len(a) * len(b)).
    """
    n = len(a)
    m = len(b)

    dp: List[List[int]] = [
        [0] * (m + 1) for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    i = n
    j = m
    result_chars: List[str] = []

    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            result_chars.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    result_chars.reverse()
    return "".join(result_chars)


def levenshtein_distance(a: str, b: str) -> int:
    """Расстояние Левенштейна (редакционное расстояние) между строками.

    Операции: вставка, удаление, замена символа.

    dp[i][j] — минимальное число операций для превращения a[:i] в b[:j].

    Временная сложность:
        O(len(a) * len(b)).
    Пространственная сложность:
        O(len(a) * len(b)).
    """
    n = len(a)
    m = len(b)

    dp: List[List[int]] = [
        [0] * (m + 1) for _ in range(n + 1)
    ]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    return dp[n][m]


def coin_change_min_coins(
    amount: int,
    coins: Sequence[int],
) -> Optional[int]:
    """Задача размена монет: минимальное количество монет для суммы.

    Вариант: считаем только минимальное число монет, без состава.

    Временная сложность:
        Пусть A = amount, k = количество номиналов.
        O(A * k).
    Пространственная сложность:
        O(A).
    """
    if amount < 0:
        raise ValueError("amount должно быть неотрицательным")

    if amount == 0:
        return 0

    if not coins:
        return None

    max_value = amount + 1
    dp: List[int] = [max_value] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        if coin <= 0:
            raise ValueError("номиналы монет должны быть положительными")
        for s in range(coin, amount + 1):
            dp[s] = min(dp[s], dp[s - coin] + 1)

    if dp[amount] == max_value:
        return None

    return dp[amount]


def lis_length_dp(sequence: Sequence[int]) -> int:
    """Наибольшая возрастающая подпоследовательность (LIS), длина.

    Классическое ДП O(n^2):
        dp[i] — длина LIS, заканчивающейся на позиции i.

    Временная сложность:
        O(n^2), где n = len(sequence).
    Пространственная сложность:
        O(n).
    """
    n = len(sequence)
    if n == 0:
        return 0

    dp: List[int] = [1] * n

    for i in range(n):
        for j in range(i):
            if sequence[j] < sequence[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)


def print_dp_table(table: List[List[int]]) -> None:
    """Простая текстовая визуализация таблицы ДП в виде матрицы.

    Используется для маленьких примеров (учебная визуализация).
    Временная сложность: O(n * m),
    где n, m — размеры таблицы.
    """
    for row in table:
        print(" ".join(f"{cell:3d}" for cell in row))


def build_lcs_table(a: str, b: str) -> List[List[int]]:
    """Строит и возвращает DP-таблицу для LCS (для визуализации).

    Это вспомогательная функция: логику совпадает с lcs_length_bottom_up,
    но возвращает всю таблицу.

    Временная сложность:
        O(len(a) * len(b)).
    Пространственная сложность:
        O(len(a) * len(b)).
    """
    n = len(a)
    m = len(b)

    dp: List[List[int]] = [
        [0] * (m + 1) for _ in range(n + 1)
    ]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp
