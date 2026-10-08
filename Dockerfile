FROM python:3.13-slim

WORKDIR /app

COPY docker_packages /tmp/docker_packages

COPY requirements-docker.txt .

RUN pip install --no-cache-dir \
    --no-index \
    --find-links=/tmp/docker_packages \
    -r requirements-docker.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", "--server.port=8501"]