FROM python:3.11-slim
LABEL org.opencontainers.image.title="GroundedIQ RAG Chatbot"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p documents chroma_db
EXPOSE 8501
CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501"]
