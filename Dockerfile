FROM python:3.12-slim

# Создаем non-root пользователя (требование безопасности)
RUN useradd --create-home --shell /bin/bash appuser
WORKDIR /home/appuser

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY src/ src/
COPY tests/ tests/
COPY data/ data/

# Меняем владельца файлов на non-root пользователя
RUN chown -R appuser:appuser /home/appuser
USER appuser

# Запускаем тесты при старте контейнера
CMD ["python", "-m", "pytest", "tests/", "-v"]
