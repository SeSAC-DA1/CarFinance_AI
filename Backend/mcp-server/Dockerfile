# CarFin MCP Server Dockerfile
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# Railway는 동적 포트를 사용하므로 EXPOSE는 제거
# 헬스체크는 Railway에서 자동 처리

# 서버 실행 - Railway의 $PORT 환경변수 사용
CMD python src/carfin_mcp_server.py --host 0.0.0.0 --port $PORT