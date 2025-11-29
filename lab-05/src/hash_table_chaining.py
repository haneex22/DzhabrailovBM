"""Модуль с реализацией хеш-таблицы с методом цепочек."""

from typing import Any, Optional, List
from hash_functions import polynomial_hash


class HashItem:
    """Элемент хеш-таблицы для метода цепочек."""

    def __init__(self, key: str, value: Any) -> None:
        """
        Инициализация элемента.

        Args:
            key: Ключ элемента
            value: Значение элемента
        """
        self.key = key
        self.value = value
        self.next: Optional['HashItem'] = None


class HashTableChaining:
    """
    Хеш-таблица с методом цепочек.

    Сложность операций:
    - В среднем случае: O(1 + α), где α - коэффициент заполнения
    - В худшем случае: O(n)
    """

    def __init__(self, size: int = 101, hash_func=None,
                 load_factor_threshold: float = 0.7) -> None:
        """
        Инициализация хеш-таблицы.

        Args:
            size: Начальный размер таблицы (простое число)
            hash_func: Функция для вычисления хеша
            load_factor_threshold: Порог коэффициента заполнения рехеширования
        """
        self.size = size
        self.count = 0
        self.table: List[Optional[HashItem]] = [None] * size
        self.hash_func = hash_func or (lambda k, s: polynomial_hash(k, s))
        self.load_factor_threshold = load_factor_threshold

    def _hash(self, key: str) -> int:
        """
        Вычисление хеша для ключа.

        Args:
            key: Ключ для хеширования

        Returns:
            Индекс в таблице
        """
        return self.hash_func(key, self.size)

    def _resize(self, new_size: int) -> None:
        """
        Изменение размера таблицы и перехеширование всех элементов.

        Args:
            new_size: Новый размер таблицы
        """
        old_table = self.table
        old_size = self.size

        self.size = new_size
        self.table = [None] * new_size
        self.count = 0

        for i in range(old_size):
            current = old_table[i]
            while current is not None:
                self.insert(current.key, current.value)
                current = current.next

    def _check_resize(self) -> None:
        """Проверка необходимости изменения размера таблицы."""
        load_factor = self.count / self.size
        if load_factor > self.load_factor_threshold:
            new_size = self.size * 2
            # Проверяем, является ли новое число простым
            is_prime = False
            while not is_prime:
                is_prime = True
                for i in range(2, int(new_size**0.5) + 1):
                    if new_size % i == 0:
                        is_prime = False
                        new_size += 1
                        break
            self._resize(new_size)

    def insert(self, key: str, value: Any) -> None:
        """
        Вставка элемента в таблицу.

        Сложность: O(1 + α) в среднем, O(n) в худшем случае.

        Args:
            key: Ключ элемента
            value: Значение элемента
        """
        self._check_resize()

        index = self._hash(key)
        new_item = HashItem(key, value)

        if self.table[index] is None:
            self.table[index] = new_item
        else:
            current = self.table[index]
            # Проверяем, не существует ли уже ключ
            while current is not None:
                if current.key == key:
                    current.value = value  # Обновляем значение
                    return
                if current.next is None:
                    break
                current = current.next
            # Добавляем новый элемент в конец цепочки
            if current is not None:
                current.next = new_item

        self.count += 1

    def search(self, key: str) -> Optional[Any]:
        """
        Поиск элемента по ключу.

        Сложность: O(1 + α) в среднем, O(n) в худшем случае.

        Args:
            key: Ключ для поиска

        Returns:
            Значение элемента или None, если не найден
        """
        index = self._hash(key)
        current = self.table[index]

        while current is not None:
            if current.key == key:
                return current.value
            current = current.next

        return None

    def delete(self, key: str) -> bool:
        """
        Удаление элемента по ключу.

        Сложность: O(1 + α) в среднем, O(n) в худшем случае.

        Args:
            key: Ключ для удаления

        Returns:
            True если элемент удален, False если не найден
        """
        index = self._hash(key)
        current = self.table[index]
        prev = None

        while current is not None:
            if current.key == key:
                if prev is None:
                    self.table[index] = current.next
                else:
                    prev.next = current.next
                self.count -= 1
                return True

            prev = current
            current = current.next

        return False

    def get_load_factor(self) -> float:
        """
        Получение коэффициента заполнения.

        Returns:
            Коэффициент заполнения таблицы
        """
        return self.count / self.size
