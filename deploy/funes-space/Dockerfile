FROM python:3.12-slim-bookworm
ARG FUNES_VERSION=v1.3.0
ARG FUNES_SHA256=7ac544a6d14c0018949ada32c39b374fd0b5f6b963a75eb4688b4d23afc944f9
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && curl -fsSL "https://github.com/huggingface/funes/releases/download/${FUNES_VERSION}/funes-x86_64-linux" -o /usr/local/bin/funes \
    && printf '%s  %s\n' "${FUNES_SHA256}" /usr/local/bin/funes | sha256sum -c - \
    && chmod 0755 /usr/local/bin/funes \
    && curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh \
        | sh -s -- -b /usr/local/bin v3.95.5 \
    && rm -rf /var/lib/apt/lists/*
COPY service /app/service
RUN pip install --no-cache-dir -r /app/service/requirements.txt
COPY server.py /app/server.py
ENV FUNES_HOME=/data/.funes PYTHONUNBUFFERED=1 PORT=7860
RUN mkdir -p /data/.funes/sources
WORKDIR /app
EXPOSE 7860
CMD ["python", "/app/server.py"]
