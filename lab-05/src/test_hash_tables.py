"""Модуль с unit-тестами для хеш-таблиц."""

import unittest
from hash_functions import simple_hash, polynomial_hash, djb2_hash
from hash_table_chaining import HashTableChaining
from hash_table_open_addressing import HashTableOpenAddressing


class TestHashFunctions(unittest.TestCase):
    """Тесты хеш-функций."""

    def test_simple_hash(self):
        """Тест простой хеш-функции."""
        self.assertEqual(simple_hash('test', 100), simple_hash('test', 100))
        self.assertNotEqual(simple_hash('test', 100), simple_hash('test2',
                                                                  100))

    def test_polynomial_hash(self):
        """Тест полиномиальной хеш-функции."""
        self.assertEqual(polynomial_hash('test', 100), polynomial_hash('test',
                                                                       100))
        # Анаграммы должны иметь разные хеши
        self.assertNotEqual(polynomial_hash('abc', 100), polynomial_hash('cba',
                                                                         100))

    def test_djb2_hash(self):
        """Тест хеш-функции DJB2."""
        self.assertEqual(djb2_hash('test', 100), djb2_hash('test', 100))
        self.assertNotEqual(djb2_hash('test', 100), djb2_hash('test2', 100))


class TestHashTableChaining(unittest.TestCase):
    """Тесты хеш-таблицы с методом цепочек."""

    def setUp(self):
        """Подготовка тестового окружения."""
        self.table = HashTableChaining(size=10)

    def test_insert_search(self):
        """Тест вставки и поиска."""
        self.table.insert('key1', 'value1')
        self.table.insert('key2', 'value2')

        self.assertEqual(self.table.search('key1'), 'value1')
        self.assertEqual(self.table.search('key2'), 'value2')
        self.assertIsNone(self.table.search('key3'))

    def test_update(self):
        """Тест обновления значения."""
        self.table.insert('key1', 'value1')
        self.table.insert('key1', 'value2')

        self.assertEqual(self.table.search('key1'), 'value2')

    def test_delete(self):
        """Тест удаления."""
        self.table.insert('key1', 'value1')
        self.assertTrue(self.table.delete('key1'))
        self.assertIsNone(self.table.search('key1'))
        self.assertFalse(self.table.delete('key1'))

    def test_collision(self):
        """Тест обработки коллизий."""
        # Создаем коллизию
        self.table.insert('a', 'value1')
        self.table.insert('k', 'value2')  # Должна быть коллизия с 'a' для хеша

        self.assertEqual(self.table.search('a'), 'value1')
        self.assertEqual(self.table.search('k'), 'value2')


class TestHashTableOpenAddressing(unittest.TestCase):
    """Тесты хеш-таблицы с открытой адресацией."""

    def test_linear_probing(self):
        """Тест линейного пробирования."""
        table = HashTableOpenAddressing(size=10, method='linear')

        table.insert('key1', 'value1')
        table.insert('key2', 'value2')

        self.assertEqual(table.search('key1'), 'value1')
        self.assertEqual(table.search('key2'), 'value2')
        self.assertIsNone(table.search('key3'))

    def test_double_hashing(self):
        """Тест двойного хеширования."""
        table = HashTableOpenAddressing(size=10, method='double')

        table.insert('key1', 'value1')
        table.insert('key2', 'value2')

        self.assertEqual(table.search('key1'), 'value1')
        self.assertEqual(table.search('key2'), 'value2')

    def test_delete(self):
        """Тест удаления для открытой адресации."""
        table = HashTableOpenAddressing(size=10)

        table.insert('key1', 'value1')
        self.assertTrue(table.delete('key1'))
        self.assertIsNone(table.search('key1'))

        # Проверяем, что можем вставить новый элемент с тем же ключом
        table.insert('key1', 'value2')
        self.assertEqual(table.search('key1'), 'value2')

    def test_reinsert_after_delete(self):
        """Тест повторной вставки после удаления."""
        table = HashTableOpenAddressing(size=10)

        table.insert('key1', 'value1')
        self.assertTrue(table.delete('key1'))
        self.assertIsNone(table.search('key1'))

        # Повторная вставка того же ключа
        table.insert('key1', 'value2')
        self.assertEqual(table.search('key1'), 'value2')


if __name__ == '__main__':
    unittest.main()
