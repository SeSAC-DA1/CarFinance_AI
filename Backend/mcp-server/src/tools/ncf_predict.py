"""
CarFin MCP Tool: NCF Predict
Neural Collaborative Filtering 기반 실시간 추천 예측
"""

import asyncio
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import pickle
import os
from dataclasses import dataclass

logger = logging.getLogger("CarFin-MCP.NCFPredict")

class NCFPredictError(Exception):
    """NCF 예측 관련 에러"""
    pass

@dataclass
class UserEmbedding:
    user_id: str
    age_group: int
    income_level: int
    location: str
    preferences: Dict[str, float]
    behavior_vector: List[float]

@dataclass
class ItemEmbedding:
    item_id: str
    price: float
    brand_encoded: int
    model_year: int
    fuel_type_encoded: int
    features_vector: List[float]

class NCFModel(nn.Module):
    """Neural Collaborative Filtering 모델 (He et al., 2017)"""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64, hidden_dims: List[int] = [128, 64, 32]):
        super(NCFModel, self).__init__()

        # GMF (Generalized Matrix Factorization) 부분
        self.user_embedding_gmf = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(num_items, embedding_dim)

        # MLP (Multi-Layer Perceptron) 부분
        self.user_embedding_mlp = nn.Embedding(num_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(num_items, embedding_dim)

        # MLP 레이어 구성
        mlp_input_dim = embedding_dim * 2
        mlp_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                mlp_layers.extend([
                    nn.Linear(mlp_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])
            else:
                mlp_layers.extend([
                    nn.Linear(hidden_dims[i-1], hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                ])

        self.mlp_layers = nn.Sequential(*mlp_layers)

        # 최종 예측 레이어
        final_input_dim = embedding_dim + hidden_dims[-1]
        self.prediction = nn.Sequential(
            nn.Linear(final_input_dim, 1),
            nn.Sigmoid()
        )

        # 가중치 초기화
        self._init_weights()

    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        # GMF 부분
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf

        # MLP 부분
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=-1)
        mlp_output = self.mlp_layers(mlp_input)

        # 결합 및 예측
        final_input = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.prediction(final_input)

        return prediction.squeeze()

class NCFPredictTool:
    """NCF 기반 실시간 추천 예측 도구"""

    def __init__(self):
        # 모델 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.user_encoder = {}  # user_id -> index
        self.item_encoder = {}  # item_id -> index
        self.model_metadata = {}

        # 모델 경로
        self.model_dir = "models"
        self.model_path = os.path.join(self.model_dir, "ncf_model.pth")
        self.encoders_path = os.path.join(self.model_dir, "encoders.pkl")

        # 실시간 학습 설정
        self.learning_rate = 0.001
        self.batch_size = 256
        self.online_buffer = []  # 실시간 피드백 저장
        self.buffer_limit = 1000

        # 성능 통계
        self.prediction_stats = {
            "total_predictions": 0,
            "avg_prediction_time": 0.0,
            "cache_hits": 0,
            "model_updates": 0
        }

        logger.info("✅ NCF Predict Tool 초기화 완료")

        # 모델 로드 시도
        asyncio.create_task(self._load_or_init_model())

    async def _load_or_init_model(self):
        """모델 로드 또는 초기화"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.encoders_path):
                await self._load_model()
                logger.info("🔄 기존 NCF 모델 로드 완료")
            else:
                await self._initialize_new_model()
                logger.info("🆕 새로운 NCF 모델 초기화 완료")
        except Exception as e:
            logger.error(f"❌ 모델 초기화 실패: {e}")

    async def _load_model(self):
        """저장된 모델 로드"""
        # 인코더 로드
        with open(self.encoders_path, 'rb') as f:
            encoders_data = pickle.load(f)
            self.user_encoder = encoders_data['user_encoder']
            self.item_encoder = encoders_data['item_encoder']
            self.model_metadata = encoders_data['metadata']

        # 모델 로드
        num_users = len(self.user_encoder)
        num_items = len(self.item_encoder)

        self.model = NCFModel(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.model_metadata.get('embedding_dim', 64),
            hidden_dims=self.model_metadata.get('hidden_dims', [128, 64, 32])
        ).to(self.device)

        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

    async def _initialize_new_model(self):
        """새로운 모델 초기화"""
        # 기본 인코더 설정 (실제로는 데이터베이스에서 가져와야 함)
        self.user_encoder = {"dummy_user": 0}
        self.item_encoder = {"dummy_item": 0}
        self.model_metadata = {
            'embedding_dim': 64,
            'hidden_dims': [128, 64, 32],
            'created_at': datetime.now().isoformat()
        }

        # 기본 모델 생성
        self.model = NCFModel(
            num_users=1000,  # 초기 용량
            num_items=10000, # 초기 용량
            embedding_dim=64,
            hidden_dims=[128, 64, 32]
        ).to(self.device)

        # 디렉토리 생성
        os.makedirs(self.model_dir, exist_ok=True)

    async def execute(self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        NCF 예측 실행

        params:
            action: 'predict' | 'batch_predict' | 'update_feedback' | 'get_stats'
            user_profile: 사용자 프로필
            item_candidates: 추천 후보 아이템들
            feedback_data: 피드백 데이터 (학습용)
        """
        try:
            action = params.get("action", "predict")

            if action == "predict":
                return await self._predict_single(params)
            elif action == "batch_predict":
                return await self._predict_batch(params)
            elif action == "update_feedback":
                return await self._update_feedback(params)
            elif action == "get_stats":
                return await self._get_prediction_stats()
            else:
                raise NCFPredictError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"❌ NCF 예측 실행 실패: {e}")
            raise NCFPredictError(f"Prediction execution failed: {e}")

    async def _predict_single(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """단일 사용자-아이템 예측"""
        start_time = datetime.now()

        user_profile = params.get("user_profile", {})
        item_candidates = params.get("item_candidates", [])

        if not user_profile or not item_candidates:
            raise NCFPredictError("Missing user_profile or item_candidates")

        # 사용자 인코딩
        user_id = user_profile.get("user_id", "anonymous")
        user_idx = self._encode_user(user_id, user_profile)

        predictions = []

        for item in item_candidates:
            item_id = item.get("vehicleid") or item.get("item_id")
            if not item_id:
                continue

            # 아이템 인코딩
            item_idx = self._encode_item(item_id, item)

            # 예측 수행
            with torch.no_grad():
                user_tensor = torch.tensor([user_idx], device=self.device)
                item_tensor = torch.tensor([item_idx], device=self.device)

                score = self.model(user_tensor, item_tensor).item()

                # 컨텍스트 정보 추가로 점수 조정
                adjusted_score = self._adjust_score_with_context(
                    score, user_profile, item, context
                )

                predictions.append({
                    "item_id": item_id,
                    "score": adjusted_score,
                    "raw_ncf_score": score,
                    "item_features": self._extract_item_features(item)
                })

        # 점수순 정렬
        predictions.sort(key=lambda x: x["score"], reverse=True)

        # 통계 업데이트
        execution_time = (datetime.now() - start_time).total_seconds()
        self._update_stats(execution_time, len(predictions))

        logger.info(f"🎯 NCF 예측 완료: {len(predictions)}개 아이템, {execution_time:.3f}초")

        return {
            "success": True,
            "predictions": predictions,
            "execution_time": execution_time,
            "model_info": {
                "device": str(self.device),
                "users_encoded": len(self.user_encoder),
                "items_encoded": len(self.item_encoder)
            }
        }

    async def _predict_batch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """배치 예측 (여러 사용자)"""
        start_time = datetime.now()

        user_profiles = params.get("user_profiles", [])
        item_candidates = params.get("item_candidates", [])

        batch_predictions = {}

        for user_profile in user_profiles:
            user_result = await self._predict_single({
                "user_profile": user_profile,
                "item_candidates": item_candidates
            })

            user_id = user_profile.get("user_id", "anonymous")
            batch_predictions[user_id] = user_result["predictions"]

        execution_time = (datetime.now() - start_time).total_seconds()

        return {
            "success": True,
            "batch_predictions": batch_predictions,
            "total_users": len(user_profiles),
            "execution_time": execution_time
        }

    async def _update_feedback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 피드백으로 모델 업데이트"""
        feedback_data = params.get("feedback_data", [])

        if not feedback_data:
            return {"success": True, "message": "No feedback data provided"}

        # 온라인 버퍼에 추가
        self.online_buffer.extend(feedback_data)

        # 버퍼 크기 관리
        if len(self.online_buffer) > self.buffer_limit:
            self.online_buffer = self.online_buffer[-self.buffer_limit:]

        # 충분한 데이터가 쌓이면 모델 업데이트
        if len(self.online_buffer) >= 100:  # 배치 크기
            await self._perform_online_learning()

        return {
            "success": True,
            "feedback_added": len(feedback_data),
            "buffer_size": len(self.online_buffer),
            "model_updated": len(self.online_buffer) >= 100
        }

    async def _perform_online_learning(self):
        """온라인 학습 수행"""
        try:
            logger.info("🔄 온라인 학습 시작")

            # 피드백 데이터 준비
            training_data = []
            for feedback in self.online_buffer[-100:]:  # 최근 100개
                user_id = feedback.get("user_id")
                item_id = feedback.get("item_id")
                rating = feedback.get("rating", 0.5)  # 클릭: 1, 무시: 0

                if user_id and item_id:
                    user_idx = self.user_encoder.get(user_id, 0)
                    item_idx = self.item_encoder.get(item_id, 0)
                    training_data.append((user_idx, item_idx, rating))

            if len(training_data) < 10:
                return

            # 모델을 학습 모드로 전환
            self.model.train()
            optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
            criterion = nn.MSELoss()

            # 미니배치 학습
            for epoch in range(5):  # 빠른 온라인 학습
                total_loss = 0

                for i in range(0, len(training_data), 32):
                    batch = training_data[i:i+32]

                    user_ids = torch.tensor([item[0] for item in batch], device=self.device)
                    item_ids = torch.tensor([item[1] for item in batch], device=self.device)
                    ratings = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=self.device)

                    optimizer.zero_grad()
                    predictions = self.model(user_ids, item_ids)
                    loss = criterion(predictions, ratings)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                logger.info(f"📈 Epoch {epoch+1}, Loss: {total_loss:.4f}")

            # 모델을 평가 모드로 전환
            self.model.eval()

            # 통계 업데이트
            self.prediction_stats["model_updates"] += 1

            logger.info("✅ 온라인 학습 완료")

        except Exception as e:
            logger.error(f"❌ 온라인 학습 실패: {e}")

    def _encode_user(self, user_id: str, user_profile: Dict[str, Any]) -> int:
        """사용자 인코딩"""
        if user_id not in self.user_encoder:
            # 새로운 사용자 추가
            new_idx = len(self.user_encoder)
            self.user_encoder[user_id] = new_idx

        return self.user_encoder[user_id]

    def _encode_item(self, item_id: str, item_info: Dict[str, Any]) -> int:
        """아이템 인코딩"""
        if item_id not in self.item_encoder:
            # 새로운 아이템 추가
            new_idx = len(self.item_encoder)
            self.item_encoder[item_id] = new_idx

        return self.item_encoder[item_id]

    def _adjust_score_with_context(
        self,
        base_score: float,
        user_profile: Dict[str, Any],
        item: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> float:
        """컨텍스트 정보로 점수 조정"""
        adjusted_score = base_score

        # 예산 적합성 조정
        user_budget = user_profile.get("budget", {})
        item_price = item.get("price", 0)

        if user_budget:
            min_budget = user_budget.get("min", 0)
            max_budget = user_budget.get("max", float('inf'))

            if min_budget <= item_price <= max_budget:
                adjusted_score *= 1.1  # 예산 내 차량 가산점
            elif item_price > max_budget:
                adjusted_score *= 0.7  # 예산 초과 감점

        # 선호 브랜드 조정
        preferred_brands = user_profile.get("preferred_brands", [])
        item_brand = item.get("manufacturer", "")

        if item_brand in preferred_brands:
            adjusted_score *= 1.15

        # 연식 선호도 조정
        current_year = 2025
        item_year = item.get("modelyear", current_year)
        age = current_year - item_year

        if age <= 3:  # 3년 이하 차량
            adjusted_score *= 1.05
        elif age >= 10:  # 10년 이상 차량
            adjusted_score *= 0.9

        return min(adjusted_score, 1.0)  # 최대 1.0으로 제한

    def _extract_item_features(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """아이템 특성 추출"""
        return {
            "price": item.get("price", 0),
            "year": item.get("modelyear", 2020),
            "mileage": item.get("distance", 0),
            "fuel_type": item.get("fueltype", ""),
            "brand": item.get("manufacturer", ""),
            "car_type": item.get("cartype", "")
        }

    def _update_stats(self, execution_time: float, num_predictions: int):
        """통계 업데이트"""
        self.prediction_stats["total_predictions"] += num_predictions

        # 평균 예측 시간 업데이트
        total_ops = self.prediction_stats["total_predictions"]
        current_avg = self.prediction_stats["avg_prediction_time"]
        self.prediction_stats["avg_prediction_time"] = (
            (current_avg * (total_ops - num_predictions) + execution_time) / total_ops
        )

    async def _get_prediction_stats(self) -> Dict[str, Any]:
        """예측 통계 반환"""
        return {
            "success": True,
            "stats": self.prediction_stats,
            "model_info": {
                "users_encoded": len(self.user_encoder),
                "items_encoded": len(self.item_encoder),
                "device": str(self.device),
                "buffer_size": len(self.online_buffer)
            }
        }

    async def save_model(self):
        """모델 저장"""
        try:
            # 모델 저장
            torch.save(self.model.state_dict(), self.model_path)

            # 인코더 저장
            encoders_data = {
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder,
                'metadata': self.model_metadata
            }

            with open(self.encoders_path, 'wb') as f:
                pickle.dump(encoders_data, f)

            logger.info("💾 NCF 모델 저장 완료")

        except Exception as e:
            logger.error(f"❌ 모델 저장 실패: {e}")

# 전역 인스턴스
ncf_predict_tool = NCFPredictTool()

# MCP Tool 인터페이스
async def carfin_ncf_predict(params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    CarFin MCP Tool: NCF 예측

    사용 예시:
    - 단일 예측: {"action": "predict", "user_profile": {...}, "item_candidates": [...]}
    - 배치 예측: {"action": "batch_predict", "user_profiles": [...], "item_candidates": [...]}
    - 피드백 학습: {"action": "update_feedback", "feedback_data": [...]}
    """
    return await ncf_predict_tool.execute(params, context)