FROM python:3.12.14-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Optional extra root CAs (e.g. a TLS-inspecting corporate proxy): put *.crt files in certs/.
COPY certs/ /tmp/certs/
RUN find /tmp/certs -name '*.crt' -exec cp {} /usr/local/share/ca-certificates/ \; \
    && update-ca-certificates \
    && rm -rf /tmp/certs
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt

RUN useradd --create-home --uid 10001 app && mkdir -p /data && chown app:app /data
WORKDIR /srv
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app ./app

USER app
ENV DATA_DIR=/data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM runtime AS test
USER root
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY tests ./tests
USER app
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
