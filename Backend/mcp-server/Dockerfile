# CarFin MCP Server - Optimized Dockerfile
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 필수 시스템 의존성만 설치 (최소화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --no-deps -r requirements.txt \
    && pip cache purge

# 애플리케이션 코드만 복사 (테스트, 문서 제외)
COPY src/ ./src/

# Railway의 동적 포트 사용
CMD python src/carfin_mcp_server.py --host 0.0.0.0 --port $PORT