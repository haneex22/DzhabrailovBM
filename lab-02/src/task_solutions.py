# task_solutions.py
"""
Решение практических задач с использованием различных структур данных.
"""

from collections import deque
from linked_list import LinkedList


def check_brackets_balance(expression: str) -> bool:
    """
    Проверка сбалансированности скобок с использованием стека.

    Args:
        expression: Строка со скобками

    Returns:
        True если скобки сбалансированы, иначе False

    Сложность: O(n), где n - длина строки
    """
    stack = []  # O(1) - используем list как стек
    brackets = {')': '(', ']': '[', '}': '{'}  # O(1)

    for char in expression:  # O(n)
        if char in '([{':  # O(1)
            stack.append(char)  # O(1) - добавление в конец стека
        elif char in ')]}':  # O(1)
            if not stack:  # O(1)
                return False  # O(1)
            if stack.pop() != brackets[char]:  # O(1) - удаление с конца
                return False  # O(1)

    return len(stack) == 0  # O(1)
# Общая сложность: O(n)


def is_palindrome_deque(text: str) -> bool:
    """
    Проверка, является ли строка палиндромом с использованием дека.

    Args:
        text: Строка для проверки

    Returns:
        True если строка палиндром, иначе False

    Сложность: O(n), где n - длина строки
    """
    char_deque = deque(text.lower())  # O(n)

    while len(char_deque) > 1:  # O(n/2) = O(n)
        first = char_deque.popleft()  # O(1)
        last = char_deque.pop()  # O(1)
        if first != last:  # O(1)
            return False  # O(1)

    return True  # O(1)
# Общая сложность: O(n)


def simulate_print_queue() -> None:
    """
    Симуляция обработки задач в очереди печати.

    Сложность: O(n), где n - количество задач
    """
    print_queue: deque[str] = deque()  # O(1) - аннотированная очередь строк

    # Добавление задач в очередь
    tasks = ['Документ1.pdf', 'Отчет.docx', 'Презентация.pptx',
             'Фото.jpg']  # O(1)

    for task in tasks:  # O(n)
        print_queue.append(task)  # O(1)
        print(f'Добавлена задача: {task}')  # O(1)

    print(f'\nВсего задач в очереди: {len(print_queue)}')  # O(1)

    # Обработка задач
    while print_queue:  # O(n)
        current_task = print_queue.popleft()  # O(1)
        print(f'Печатается: {current_task}')  # O(1)
        print(f'Осталось задач: {len(print_queue)}')  # O(1)

    print('Все задачи выполнены!')  # O(1)
# Общая сложность: O(n)


def linked_list_demo() -> None:
    """
    Демонстрация работы связного списка.

    Сложность: O(n) для операций обхода
    """
    linked_list = LinkedList()  # O(1)

    print("Демонстрация связного списка:")

    # Вставка в начало
    for i in range(5):  # O(5)
        linked_list.insert_at_start(i)  # O(1)
    print(f"После вставки в начало: {linked_list}")  # O(n)

    # Вставка в конец
    for i in range(5, 10):  # O(5)
        linked_list.insert_at_end(i)  # O(1)
    print(f"После вставки в конец: {linked_list}")  # O(n)

    # Удаление из начала
    removed = linked_list.delete_from_start()  # O(1)
    print(f"Удален из начала: {removed}")  # O(1)
    print(f"После удаления: {linked_list}")  # O(n)
# Общая сложность: O(n)


def main():
    """Основная функция для демонстрации решений."""

    print("=== Проверка сбалансированности скобок ===")
    test_expressions = [
        "((()))",
        "([{}])",
        "({[)]}",
        "((())",
        "(()))"
    ]

    for expr in test_expressions:
        result = check_brackets_balance(expr)
        print(f"{expr}: {'Сбалансированы' if result else 'Не сбалансированы'}")

    print("\n=== Проверка палиндромов ===")
    test_strings = ["радар", "level", "А роза упала на лапу Азора", "hello"]

    for text in test_strings:
        result = is_palindrome_deque(text)
        print(f"'{text}': {'Палиндром' if result else 'Не палиндром'}")

    print("\n=== Симуляция очереди печати ===")
    simulate_print_queue()

    print("\n=== Демонстрация связного списка ===")
    linked_list_demo()


if __name__ == '__main__':
    main()
