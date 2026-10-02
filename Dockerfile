FROM registry.access.redhat.com/ubi9/python-311@sha256:a0bdb55576fc5b8d6704279307817828ef027e1065533ceba133fe9516003a6c

USER root

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/
COPY templates/ ./templates/
COPY benchmark_results.json benchmark_results.qwen.json ./
COPY quality/ ./quality/
COPY bench/ ./bench/

# The UBI image already has the non-root user 1001. OpenShift runs containers with an
# arbitrary UID in group 0, so group 0 gets the same permissions as the owner.
RUN chown -R 1001:0 /app && chmod -R g=u /app

USER 1001

EXPOSE 8000


CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
