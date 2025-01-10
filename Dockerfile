# Base image
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY pyproject.toml poetry.lock ./
COPY app/ ./app

# Install dependencies via Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# Expose the Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0"]