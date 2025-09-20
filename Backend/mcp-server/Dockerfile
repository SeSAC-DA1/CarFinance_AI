# CarFin MCP Server - Google Cloud Run 최적화
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치 (Google Cloud Run 최적화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge

# 애플리케이션 코드 복사
COPY src/ ./src/

# Google Cloud Run은 PORT 환경변수 사용
EXPOSE 8080
ENV PORT=8080

# 서버 실행 (Google Cloud Run 포트 사용)
CMD python src/carfin_mcp_server.py --host 0.0.0.0 --port $PORT