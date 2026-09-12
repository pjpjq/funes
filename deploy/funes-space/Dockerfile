FROM python:3.12-slim
WORKDIR /app
COPY service /app/service
RUN pip install --no-cache-dir -r /app/service/requirements.txt
ENV PYTHONUNBUFFERED=1 FUNES_DATA_DIR=/data PORT=7860
EXPOSE 7860
CMD ["python", "-m", "service.server"]
