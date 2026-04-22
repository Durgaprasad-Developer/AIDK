FROM python:3.10-slim

WORKDIR /app

# 🛡️ Copy everything first (safer for HF)
COPY . /app

# 📦 Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 🌐 Environment
ENV PYTHONPATH=/app

# 🚀 Port for HuggingFace
EXPOSE 7860

# ⚡ Stable FastAPI launch (HF optimized)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "120"]