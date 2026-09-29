FROM registry.access.redhat.com/ubi9/python-311:latest

USER root

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/
COPY templates/ ./templates/
COPY benchmark_results.json ./
# Captured model outputs and `vllm bench serve` sweeps; both folders ship with a .gitkeep so COPY never fails
COPY quality/ ./quality/
COPY bench/ ./bench/

# OpenShift runs containers with an arbitrary UID in group 0
RUN useradd -m -u 1001 demouser && \
    chown -R 1001:0 /app && chmod -R g=u /app

USER 1001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
