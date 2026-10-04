FROM python:3.12-slim
WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY backend /app/backend
COPY corruptions.py /app/
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
