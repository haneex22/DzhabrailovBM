"""Модуль с классическими рекурсивными алгоритмами."""


def factorial(n: int) -> int:
    """
    Вычисляет факториал числа n рекурсивным способом.

    Args:
        n: Целое неотрицательное число

    Returns:
        Факториал числа n

    Raises:
        ValueError: Если n отрицательное
    """
    if n < 0:
        raise ValueError('Факториал определен только для положительных чисел')
    if n == 0 or n == 1:
        return 1
    return n * factorial(n - 1)
# Временная сложность: O(n)
# Глубина рекурсии: O(n)


def fibonacci(n: int) -> int:
    """
    Вычисляет n-е число Фибоначчи наивным рекурсивным способом.

    Args:
        n: Порядковый номер числа Фибоначчи

    Returns:
        n-е число Фибоначчи

    Raises:
        ValueError: Если n отрицательное
    """
    if n < 0:
        raise ValueError('Номер числа Фибоначчи должен быть неотрицательным')
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)
# Временная сложность: O(2^n)
# Глубина рекурсии: O(n)


def fast_power(a: float, n: int) -> float:
    """
    Быстрое возведение числа a в степень n через степень двойки.

    Args:
        a: Основание
        n: Показатель степени (целое неотрицательное число)

    Returns:
        a в степени n

    Raises:
        ValueError: Если n отрицательное
    """
    if n < 0:
        raise ValueError('Показатель степени должен быть неотрицательным')
    if n == 0:
        return 1
    if n == 1:
        return a

    half_power = fast_power(a, n // 2)
    if n % 2 == 0:
        return half_power * half_power
    else:
        return a * half_power * half_power
# Временная сложность: O(log n)
# Глубина рекурсии: O(log n)


if __name__ == '__main__':
    # Тестирование функций
    print('Факториал 5:', factorial(5))
    print('10-е число Фибоначчи:', fibonacci(10))
    print('2 в степени 10:', fast_power(2, 10))
