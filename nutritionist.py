# Консольный ИИ-нутрициолог на Python 3.
# Использует DeepSeek через библиотеку openai.
# Ключ берётся из .env (переменная DEEPSEEK_API_KEY), не hardcoded.

import os
import sys

from dotenv import load_dotenv
from openai import (
    OpenAI,
    OpenAIError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    NotFoundError,
    BadRequestError,
    APIStatusError,
)

# Название модели DeepSeek. По умолчанию deepseek-chat.
DEFAULT_MODEL = "deepseek-chat"


# Системная инструкция: дружелюбный нутрициолог, русский язык, кратко.
SYSTEM_PROMPT = (
    "Ты дружелюбный нутрициолог. Отвечаешь на русском языке, кратко и понятно. "
    "Даёшь общую информацию о питании, составе продуктов и здоровых привычках. "
    "Не ставишь диагнозы и не назначаешь лечение. "
    "При вопросах о болезнях, беременности или лекарствах советуешь обратиться к врачу."
)


def load_config() -> tuple[str, str]:
    """Читает ключ и название модели из .env."""
    load_dotenv()  # подгружает переменные из .env в окружение
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    model = os.environ.get("DEEPSEEK_MODEL", "").strip() or DEFAULT_MODEL
    return key, model


def create_client(api_key: str) -> OpenAI:
    """Создаёт клиент OpenAI, настроенный на DeepSeek."""
    return OpenAI(
        base_url="https://api.deepseek.com",
        api_key=api_key,
    )


def print_help() -> None:
    """Печатает справку по доступным командам."""
    print("Команды:")
    print("  help  — показать эту справку")
    print("  clear — очистить историю диалога")
    print("  quit  — выйти из программы")


def handle_api_error(exc: Exception) -> str:
    """Преобразует ошибку API в понятное сообщение на русском."""
    if isinstance(exc, AuthenticationError):
        return "Неверный ключ API (401). Проверь DEEPSEEK_API_KEY в .env."
    if isinstance(exc, RateLimitError):
        status = getattr(exc, "status_code", None)
        if status == 402:
            return "Недостаточно баланса (402). Пополни счёт DeepSeek."
        return "Превышен лимит запросов. Попробуй позже."
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == 401:
            return "Неверный ключ API (401). Проверь DEEPSEEK_API_KEY в .env."
        if status == 402:
            return "Недостаточно баланса (402). Пополни счёт DeepSeek."
        if status == 429:
            return "Превышен лимит запросов. Попробуй позже."
    if isinstance(exc, APIConnectionError):
        return "Нет интернета или не удалось подключиться к DeepSeek."
    if isinstance(exc, NotFoundError):
        return "Неверное имя модели. Проверь DEEPSEEK_MODEL в .env."
    if isinstance(exc, BadRequestError):
        msg = str(exc).lower()
        if "model" in msg:
            return "Неверное имя модели. Проверь DEEPSEEK_MODEL в .env."
        return f"Ошибка запроса: {exc}"
    return f"Ошибка API: {exc}"


def run_conversation(client: OpenAI, model: str) -> None:
    """Основной цикл диалога: пользователь пишет вопрос, агент отвечает."""
    # История диалога: список сообщений (system, user, assistant).
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Привет! Я ИИ-нутрициолог. Задавай вопросы о питании.")
    print("Команды: help, clear, quit")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Ctrl+D или Ctrl+C — корректное завершение.
            print()
            break

        if not user_input:
            continue

        command = user_input.lower()
        if command in ("quit", "exit"):
            print("Пока!")
            break
        if command == "help":
            print_help()
            continue
        if command == "clear":
            # Очищаем историю, оставляя только системную инструкцию.
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("История диалога очищена.")
            continue

        # Добавляем вопрос пользователя в историю.
        messages.append({"role": "user", "content": user_input})

        try:
            # Отправляем запрос в DeepSeek.
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            answer = response.choices[0].message.content
        except (OpenAIError, Exception) as exc:
            print(handle_api_error(exc))
            # Убираем неудачное сообщение из истории, чтобы диалог остался согласованным.
            messages.pop()
            continue

        # Печатаем ответ и запоминаем его в истории.
        print(answer)
        messages.append({"role": "assistant", "content": answer})


def main() -> None:
    """Точка входа: проверяем ключ и запускаем диалог."""
    api_key, model = load_config()
    if not api_key:
        print(
            "Ошибка: не найден ключ DEEPSEEK_API_KEY. "
            "Создай файл .env с переменной DEEPSEEK_API_KEY=твой_ключ. "
            "Пример см. в .env.example",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        client = create_client(api_key)
        run_conversation(client, model)
    except OpenAIError as exc:
        print(handle_api_error(exc), file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
