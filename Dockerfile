FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY . /app/src/
WORKDIR /app/src

# Default port for the FastAPI app
EXPOSE 8000

# Use gunicorn for production deployment
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "app:app"]