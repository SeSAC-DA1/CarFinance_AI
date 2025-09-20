# 🚀 CarFin AI - Google Cloud Run 배포 가이드

## 📋 사전 준비사항

### 1. Google Cloud 계정 및 프로젝트 생성
```
1. https://console.cloud.google.com 접속
2. 새 프로젝트 생성: "carfin-ai-production"
3. 프로젝트 ID 기록: carfin-ai-production-XXXXX
4. 결제 계정 연결 ($300 무료 크레딧 활용)
```

### 2. 필수 API 활성화
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

또는 콘솔에서:
- Cloud Build API
- Cloud Run API
- Container Registry API

### 3. 서비스 계정 생성
```bash
# 서비스 계정 생성
gcloud iam service-accounts create github-actions \
    --description="GitHub Actions for CarFin AI" \
    --display-name="GitHub Actions"

# 권한 부여
gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/cloudbuild.builds.editor"

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/run.admin"

gcloud projects add-iam-policy-binding PROJECT_ID \
    --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# 키 파일 생성
gcloud iam service-accounts keys create key.json \
    --iam-account=github-actions@PROJECT_ID.iam.gserviceaccount.com
```

## 🔑 GitHub Secrets 설정

GitHub 리포지토리 → Settings → Secrets and variables → Actions

```
GCP_PROJECT_ID: carfin-ai-production-XXXXX
GCP_SA_KEY: [key.json 파일 내용 전체 복사]
DB_HOST: carfin-db.cbkayiqs4div.ap-northeast-2.rds.amazonaws.com
DB_PORT: 5432
DB_DATABASE: carfin
DB_USER: carfin_user
DB_PASSWORD: [AWS RDS 비밀번호]
```

## 🚀 배포 방법

### 자동 배포 (권장)
```bash
# GitHub에 푸시하면 자동 배포
git add .
git commit -m "🚀 Google Cloud Run 배포 설정"
git push origin main
```

### 수동 배포
```bash
# 로컬에서 직접 배포
cd Backend/mcp-server
gcloud builds submit --config cloudbuild.yaml
```

## 🔍 배포 확인

### 서비스 URL 확인
```bash
gcloud run services list --platform managed --region asia-northeast1
```

### 헬스체크
```bash
curl https://[SERVICE-URL]/health
```

예상 응답:
```json
{
  "status": "healthy",
  "server": "CarFin-MCP",
  "version": "1.0.0-beta",
  "tools_registered": 6
}
```

## 💰 비용 모니터링

### 예상 비용
- **무료 할당량**: 월 200만 요청
- **CPU 시간**: $0.00001/vCPU초
- **메모리**: $0.0000010/GB초
- **네트워킹**: $0.12/GB

### 월 예상 비용 (1000 요청/일 기준)
- CPU (2 vCPU): ~$5
- 메모리 (4GB): ~$8
- **총합**: ~$13/월

## 🔧 문제 해결

### 빌드 실패 시
```bash
# 로그 확인
gcloud builds log [BUILD_ID]

# 수동 빌드 테스트
cd Backend/mcp-server
docker build -t test .
```

### 배포 실패 시
```bash
# Cloud Run 로그 확인
gcloud logs read --service carfin-mcp --limit 50

# 서비스 상태 확인
gcloud run services describe carfin-mcp --region asia-northeast1
```

## 🌟 고급 설정

### 커스텀 도메인 연결
```bash
gcloud run domain-mappings create \
    --service carfin-mcp \
    --domain api.carfin.ai \
    --region asia-northeast1
```

### Auto Scaling 설정
```yaml
# cloudbuild.yaml에서 설정
--min-instances: '1'      # 최소 인스턴스
--max-instances: '10'     # 최대 인스턴스
--concurrency: '80'       # 동시 요청 수
```

## 📊 모니터링

### Google Cloud Console
- Cloud Run → carfin-mcp → 메트릭
- Cloud Logging → 로그 탐색기
- Cloud Monitoring → 대시보드

### Gemini API 사용량
- Google AI Studio → API 사용량
- Cloud Console → API 및 서비스 → 사용량