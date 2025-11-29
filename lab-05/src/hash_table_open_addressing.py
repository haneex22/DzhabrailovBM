"""Модуль с реализацией хеш-таблицы с открытой адресацией."""

from typing import Any, Optional, List
from hash_functions import polynomial_hash, djb2_hash


class OpenAddressingItem:
    """Элемент хеш-таблицы для открытой адресации."""

    def __init__(self, key: Optional[str] = None, value: Any = None) -> None:
        """
        Инициализация элемента.

        Args:
            key: Ключ элемента
            value: Значение элемента
        """
        self.key = key
        self.value = value
        self.is_deleted = False


class HashTableOpenAddressing:
    """
    Хеш-таблица с открытой адресацией.

    Поддерживает линейное пробирование и двойное хеширование.
    """

    def __init__(self, size: int = 101, method: str = 'linear',
                 hash_func1=None, hash_func2=None,
                 load_factor_threshold: float = 0.7) -> None:
        """
        Инициализация хеш-таблицы.

        Args:
            size: Начальный размер таблицы (простое число)
            method: Метод разрешения коллизий ('linear' или 'double')
            hash_func1: Первая хеш-функция
            hash_func2: Вторая хеш-функция (для двойного хеширования)
            load_factor_threshold: Порог коэффициента заполнения рехеширования
        """
        self.size = size
        self.count = 0
        self.deleted_count = 0
        self.table: List[OpenAddressingItem] = [OpenAddressingItem() for _
                                                in range(size)]
        self.method = method
        self.hash_func1 = hash_func1 or (lambda k, s: polynomial_hash(k, s))
        self.hash_func2 = hash_func2 or (lambda k, s: djb2_hash(k, s) if s > 1
                                         else 1)
        self.load_factor_threshold = load_factor_threshold

    def _hash1(self, key: str) -> int:
        """
        Вычисление первого хеша для ключа.

        Args:
            key: Ключ для хеширования

        Returns:
            Индекс в таблице
        """
        return self.hash_func1(key, self.size)

    def _hash2(self, key: str) -> int:
        """
        Вычисление второго хеша для ключа (для двойного хеширования).

        Args:
            key: Ключ для хеширования

        Returns:
            Шаг для пробирования
        """
        hash2 = self.hash_func2(key, self.size - 1)
        return hash2 if hash2 != 0 else 1

    def _probe_sequence(self, key: str, i: int) -> int:
        """
        Вычисление последовательности пробирования.

        Args:
            key: Ключ элемента
            i: Номер попытки

        Returns:
            Индекс в таблице
        """
        if self.method == 'linear':
            return (self._hash1(key) + i) % self.size
        elif self.method == 'double':
            return (self._hash1(key) + i * self._hash2(key)) % self.size
        else:
            raise ValueError("Неизвестный метод пробирования")

    def _find_index(self, key: str, for_insert: bool = False) -> int:
        """
        Поиск индекса для ключа.

        Args:
            key: Ключ для поиска
            for_insert: Флаг для вставки (ищет пустую ячейку)

        Returns:
            Индекс элемента или -1 если не найден
        """
        i = 0
        first_deleted = -1

        while i < self.size:
            index = self._probe_sequence(key, i)
            item = self.table[index]

            if item.key is None and not item.is_deleted:
                # Нашли пустую ячейку
                if first_deleted != -1 and for_insert:
                    return first_deleted
                return index

            if item.key == key and not item.is_deleted:
                # Нашли ключ
                return index

            if item.is_deleted and first_deleted == -1 and for_insert:
                # Запоминаем первую удаленную ячейку
                first_deleted = index

            i += 1

        return -1

    def _resize(self, new_size: int) -> None:
        """
        Изменение размера таблицы и перехеширование всех элементов.

        Args:
            new_size: Новый размер таблицы
        """
        old_table = self.table
        old_size = self.size

        self.size = new_size
        self.table = [OpenAddressingItem() for _ in range(new_size)]
        self.count = 0
        self.deleted_count = 0

        for i in range(old_size):
            item = old_table[i]
            if item.key is not None and not item.is_deleted:
                self.insert(item.key, item.value)

    def _check_resize(self) -> None:
        """Проверка необходимости изменения размера таблицы."""
        load_factor = (self.count + self.deleted_count) / self.size
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

        Сложность: O(1/(1-α)) в среднем, O(n) в худшем случае.

        Args:
            key: Ключ элемента
            value: Значение элемента
        """
        self._check_resize()

        index = self._find_index(key, for_insert=True)
        if index == -1:
            raise Exception("Хеш-таблица переполнена")

        item = self.table[index]
        if item.key is None or item.is_deleted:
            # Новая вставка
            if item.is_deleted:
                self.deleted_count -= 1
            item.key = key
            item.value = value
            item.is_deleted = False
            self.count += 1
        else:
            # Обновление существующего элемента
            item.value = value

    def search(self, key: str) -> Optional[Any]:
        """
        Поиск элемента по ключу.

        Сложность: O(1/(1-α)) в среднем, O(n) в худшем случае.

        Args:
            key: Ключ для поиска

        Returns:
            Значение элемента или None, если не найден
        """
        index = self._find_index(key)
        if index != -1 and not self.table[index].is_deleted:
            return self.table[index].value
        return None

    def delete(self, key: str) -> bool:
        """
        Удаление элемента по ключу.

        Сложность: O(1/(1-α)) в среднем, O(n) в худшем случае.

        Args:
            key: Ключ для удаления

        Returns:
            True если элемент удален, False если не найден
        """
        index = self._find_index(key)
        if index != -1 and not self.table[index].is_deleted:
            self.table[index].is_deleted = True
            self.count -= 1
            self.deleted_count += 1
            return True
        return False

    def get_load_factor(self) -> float:
        """
        Получение коэффициента заполнения.

        Returns:
            Коэффициент заполнения таблицы
        """
        return self.count / self.size
