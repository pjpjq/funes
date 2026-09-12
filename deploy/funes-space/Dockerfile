FROM rust:1.94-bookworm AS build
WORKDIR /src
COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl protobuf-compiler \
    && curl -sSfL https://raw.githubusercontent.com/trufflesecurity/trufflehog/main/scripts/install.sh \
        | sh -s -- -b /usr/local/bin v3.95.5 \
    && cargo build --release --no-default-features --features blas

FROM python:3.12-slim-bookworm
COPY --from=build /src/target/release/funes /usr/local/bin/funes
COPY --from=build /usr/local/bin/trufflehog /usr/local/bin/trufflehog
COPY space/server.py /app/server.py
ENV FUNES_HOME=/data/.funes PYTHONUNBUFFERED=1 PORT=7860
RUN mkdir -p /data/.funes/sources
WORKDIR /app
EXPOSE 7860
CMD ["python", "/app/server.py"]
