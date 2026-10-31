FROM rust:1.94-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --no-default-features --features blas

FROM python:3.12-slim-bookworm
COPY --from=build /src/target/release/funes /usr/local/bin/funes
COPY space/server.py /app/server.py
COPY space/canonical_ab.py /app/space/canonical_ab.py
COPY service /app/service
RUN pip install --no-cache-dir -r /app/service/requirements.txt
ENV FUNES_HOME=/data/.funes PYTHONUNBUFFERED=1 PORT=7860
RUN mkdir -p /data/.funes/sources
WORKDIR /app
EXPOSE 7860
CMD ["python", "/app/server.py"]
