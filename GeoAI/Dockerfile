FROM python:3.11-slim

WORKDIR /app

COPY requirements-space.txt .
RUN pip install --no-cache-dir -r requirements-space.txt

COPY app.py .

EXPOSE 7860

CMD ["python", "app.py"]
