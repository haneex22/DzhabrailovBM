"""Модуль с реализацией различных хеш-функций для строковых ключей."""


def simple_hash(key: str, table_size: int) -> int:
    """
    Простая хеш-функция: сумма кодов символов.

    Особенности: быстрая, но плохое распределение для похожих строк.
    Качество распределения: низкое.

    Args:
        key: Строка для хеширования
        table_size: Размер хеш-таблицы

    Returns:
        Хеш-значение в диапазоне [0, table_size-1]
    """
    hash_value = 0
    for char in key:
        hash_value += ord(char)
    return hash_value % table_size


def polynomial_hash(key: str, table_size: int, base: int = 31) -> int:
    """
    Полиномиальная хеш-функция.

    Особенности: хорошее распределение, устойчива к анаграммам.
    Качество распределения: высокое.

    Args:
        key: Строка для хеширования
        table_size: Размер хеш-таблицы
        base: Основание полинома

    Returns:
        Хеш-значение в диапазоне [0, table_size-1]
    """
    hash_value = 0
    for char in key:
        hash_value = (hash_value * base + ord(char)) % table_size
    return hash_value


def djb2_hash(key: str, table_size: int) -> int:
    """
    Хеш-функция DJB2.

    Особенности: отличное распределение, популярный выбор.
    Качество распределения: очень высокое.

    Args:
        key: Строка для хеширования
        table_size: Размер хеш-таблицы

    Returns:
        Хеш-значение в диапазоне [0, table_size-1]
    """
    hash_value = 5381
    for char in key:
        hash_value = ((hash_value << 5) + hash_value) + ord(char)
    return hash_value % table_size
