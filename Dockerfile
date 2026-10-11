FROM python:3.12-slim AS build
WORKDIR /app
COPY requirements.txt requirements-dev.txt serve.py ./
RUN pip install --no-cache-dir -r requirements-dev.txt && python serve.py export

FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve.py .
COPY --from=build /app/model.onnx .
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
