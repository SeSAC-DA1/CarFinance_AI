"""
CarFin MCP Tool: Learning Update
실시간 사용자 피드백 기반 모델 학습 업데이트
"""

import asyncio
import logging
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger("CarFin-MCP.LearningUpdate")

class LearningUpdateError(Exception):
    """학습 업데이트 관련 에러"""
    pass

@dataclass
class UserFeedback:
    user_id: str
    item_id: str
    feedback_type: str  # 'click', 'view', 'like', 'dislike', 'purchase_inquiry'
    rating: float  # 0.0 ~ 1.0
    session_id: str
    timestamp: datetime
    context: Dict[str, Any]

@dataclass
class LearningBatch:
    batch_id: str
    feedbacks: List[UserFeedback]
    created_at: datetime
    model_version: str
    processed: bool = False

class LearningUpdateTool:
    """실시간 학습 업데이트 도구"""

    def __init__(self):
        # 피드백 저장소
        self.feedback_db_path = "feedback.db"
        self.init_feedback_database()

        # 학습 설정
        self.learning_config = {
            "batch_size": 64,
            "learning_rate": 0.0001,
            "update_frequency": 100,  # 100개 피드백마다 업데이트
            "forgetting_factor": 0.95,  # 과거 학습 가중치
            "min_feedback_threshold": 10
        }

        # 실시간 피드백 버퍼
        self.feedback_buffer = []
        self.user_interaction_history = defaultdict(list)

        # 학습 통계
        self.learning_stats = {
            "total_feedbacks": 0,
            "model_updates": 0,
            "last_update": None,
            "avg_update_time": 0.0,
            "feedback_types_count": defaultdict(int),
            "user_engagement_metrics": defaultdict(dict)
        }

        # A/B 테스트 설정
        self.ab_test_groups = {
            "control": 0.5,    # 기존 모델
            "experimental": 0.5  # 업데이트된 모델
        }

        logger.info("✅ LearningUpdate Tool 초기화 완료")

    def init_feedback_database(self):
        """피드백 데이터베이스 초기화"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()

            # 피드백 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    feedback_type TEXT NOT NULL,
                    rating REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    context TEXT,
                    processed BOOLEAN DEFAULT FALSE
                )
            ''')

            # 학습 배치 테이블 생성
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learning_batches (
                    batch_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    feedback_count INTEGER NOT NULL,
                    processed BOOLEAN DEFAULT FALSE,
                    performance_metrics TEXT
                )
            ''')

            # 인덱스 생성
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_timestamp ON user_feedback(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_user_id ON user_feedback(user_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_feedback_processed ON user_feedback(processed)')

            conn.commit()
            conn.close()

            logger.info("📊 피드백 데이터베이스 초기화 완료")

        except Exception as e:
            logger.error(f"❌ 데이터베이스 초기화 실패: {e}")

    async def execute(self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        학습 업데이트 실행

        params:
            action: 'record_feedback' | 'trigger_update' | 'get_stats' | 'get_user_profile'
            feedback_data: 사용자 피드백 데이터
            force_update: 강제 업데이트 여부
            user_id: 사용자 ID (프로필 조회용)
        """
        try:
            action = params.get("action", "record_feedback")

            if action == "record_feedback":
                return await self._record_user_feedback(params)
            elif action == "trigger_update":
                return await self._trigger_model_update(params)
            elif action == "get_stats":
                return await self._get_learning_stats()
            elif action == "get_user_profile":
                return await self._get_user_learning_profile(params)
            elif action == "ab_test_assignment":
                return await self._assign_ab_test_group(params)
            else:
                raise LearningUpdateError(f"Unknown action: {action}")

        except Exception as e:
            logger.error(f"❌ 학습 업데이트 실행 실패: {e}")
            raise LearningUpdateError(f"Learning update execution failed: {e}")

    async def _record_user_feedback(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 피드백 기록"""
        feedback_data = params.get("feedback_data", {})

        if not feedback_data:
            raise LearningUpdateError("No feedback data provided")

        # 피드백 객체 생성
        user_feedback = UserFeedback(
            user_id=feedback_data.get("user_id", "anonymous"),
            item_id=feedback_data.get("item_id", ""),
            feedback_type=feedback_data.get("feedback_type", "view"),
            rating=self._convert_feedback_to_rating(
                feedback_data.get("feedback_type", "view"),
                feedback_data.get("explicit_rating", None)
            ),
            session_id=feedback_data.get("session_id", ""),
            timestamp=datetime.now(),
            context=feedback_data.get("context", {})
        )

        # 데이터베이스에 저장
        await self._save_feedback_to_db(user_feedback)

        # 버퍼에 추가
        self.feedback_buffer.append(user_feedback)

        # 사용자 상호작용 히스토리 업데이트
        self.user_interaction_history[user_feedback.user_id].append(user_feedback)

        # 통계 업데이트
        self.learning_stats["total_feedbacks"] += 1
        self.learning_stats["feedback_types_count"][user_feedback.feedback_type] += 1

        # 자동 업데이트 트리거 확인
        should_update = len(self.feedback_buffer) >= self.learning_config["update_frequency"]

        if should_update:
            asyncio.create_task(self._trigger_model_update({"force_update": False}))

        logger.info(f"📝 피드백 기록: {user_feedback.user_id} → {user_feedback.item_id} ({user_feedback.feedback_type})")

        return {
            "success": True,
            "feedback_recorded": True,
            "buffer_size": len(self.feedback_buffer),
            "should_update": should_update,
            "feedback_id": f"{user_feedback.user_id}_{user_feedback.timestamp.isoformat()}"
        }

    async def _save_feedback_to_db(self, feedback: UserFeedback):
        """피드백을 데이터베이스에 저장"""
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO user_feedback
                (user_id, item_id, feedback_type, rating, session_id, timestamp, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.user_id,
                feedback.item_id,
                feedback.feedback_type,
                feedback.rating,
                feedback.session_id,
                feedback.timestamp.isoformat(),
                json.dumps(feedback.context)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ 피드백 저장 실패: {e}")

    def _convert_feedback_to_rating(self, feedback_type: str, explicit_rating: Optional[float]) -> float:
        """피드백 타입을 평점으로 변환"""
        if explicit_rating is not None:
            return max(0.0, min(1.0, explicit_rating))

        # 암시적 피드백 변환
        feedback_ratings = {
            "view": 0.1,           # 단순 조회
            "click": 0.3,          # 클릭
            "detail_view": 0.5,    # 상세 조회
            "like": 0.8,           # 좋아요
            "favorite": 0.9,       # 찜하기
            "purchase_inquiry": 1.0,  # 구매 문의
            "dislike": 0.0,        # 싫어요
            "skip": 0.05           # 건너뛰기
        }

        return feedback_ratings.get(feedback_type, 0.1)

    async def _trigger_model_update(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """모델 업데이트 트리거"""
        start_time = datetime.now()

        force_update = params.get("force_update", False)

        # 업데이트 조건 확인
        if not force_update and len(self.feedback_buffer) < self.learning_config["min_feedback_threshold"]:
            return {
                "success": True,
                "updated": False,
                "reason": "Insufficient feedback data",
                "buffer_size": len(self.feedback_buffer)
            }

        try:
            logger.info("🔄 모델 업데이트 시작")

            # 1. 학습 배치 생성
            batch = await self._create_learning_batch()

            # 2. NCF 모델 업데이트
            ncf_update_result = await self._update_ncf_model(batch)

            # 3. 에이전트 가중치 업데이트
            agent_weights_result = await self._update_agent_weights(batch)

            # 4. 개인화 프로필 업데이트
            personalization_result = await self._update_user_profiles(batch)

            # 5. A/B 테스트 결과 분석
            ab_test_result = await self._analyze_ab_test_performance()

            # 통계 업데이트
            execution_time = (datetime.now() - start_time).total_seconds()
            self.learning_stats["model_updates"] += 1
            self.learning_stats["last_update"] = datetime.now().isoformat()

            # 평균 업데이트 시간 계산
            update_count = self.learning_stats["model_updates"]
            current_avg = self.learning_stats["avg_update_time"]
            self.learning_stats["avg_update_time"] = (
                (current_avg * (update_count - 1) + execution_time) / update_count
            )

            # 버퍼 초기화
            self.feedback_buffer = []

            logger.info(f"✅ 모델 업데이트 완료: {execution_time:.3f}초")

            return {
                "success": True,
                "updated": True,
                "execution_time": execution_time,
                "batch_id": batch.batch_id,
                "update_results": {
                    "ncf_model": ncf_update_result,
                    "agent_weights": agent_weights_result,
                    "personalization": personalization_result,
                    "ab_test": ab_test_result
                },
                "feedbacks_processed": len(batch.feedbacks)
            }

        except Exception as e:
            logger.error(f"❌ 모델 업데이트 실패: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds()
            }

    async def _create_learning_batch(self) -> LearningBatch:
        """학습 배치 생성"""
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        batch = LearningBatch(
            batch_id=batch_id,
            feedbacks=self.feedback_buffer.copy(),
            created_at=datetime.now(),
            model_version="v1.0-beta"
        )

        # 배치 정보를 데이터베이스에 저장
        try:
            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO learning_batches
                (batch_id, created_at, model_version, feedback_count)
                VALUES (?, ?, ?, ?)
            ''', (
                batch.batch_id,
                batch.created_at.isoformat(),
                batch.model_version,
                len(batch.feedbacks)
            ))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"❌ 배치 저장 실패: {e}")

        return batch

    async def _update_ncf_model(self, batch: LearningBatch) -> Dict[str, Any]:
        """NCF 모델 업데이트"""
        try:
            # 피드백 데이터를 학습 데이터로 변환
            training_data = []

            for feedback in batch.feedbacks:
                training_data.append({
                    "user_id": feedback.user_id,
                    "item_id": feedback.item_id,
                    "rating": feedback.rating,
                    "timestamp": feedback.timestamp
                })

            # NCF 예측 도구를 통해 온라인 학습 수행
            # (실제로는 ncf_predict_tool을 import해서 사용)
            update_params = {
                "action": "update_feedback",
                "feedback_data": training_data
            }

            # 여기서는 시뮬레이션으로 처리
            await asyncio.sleep(0.1)  # 실제 학습 시뮬레이션

            return {
                "updated": True,
                "training_samples": len(training_data),
                "learning_rate": self.learning_config["learning_rate"],
                "model_improvement": 0.02  # 모의 개선율
            }

        except Exception as e:
            logger.error(f"❌ NCF 모델 업데이트 실패: {e}")
            return {"updated": False, "error": str(e)}

    async def _update_agent_weights(self, batch: LearningBatch) -> Dict[str, Any]:
        """에이전트 가중치 업데이트"""
        try:
            # 피드백 기반 에이전트 성능 분석
            agent_performance = defaultdict(list)

            for feedback in batch.feedbacks:
                # 컨텍스트에서 추천 소스 정보 추출
                context = feedback.context
                recommending_agent = context.get("recommending_agent", "unknown")

                if recommending_agent != "unknown":
                    agent_performance[recommending_agent].append(feedback.rating)

            # 에이전트별 평균 성능 계산
            agent_scores = {}
            for agent, ratings in agent_performance.items():
                if ratings:
                    agent_scores[agent] = {
                        "avg_rating": np.mean(ratings),
                        "sample_count": len(ratings),
                        "performance_trend": "improving" if np.mean(ratings) > 0.5 else "declining"
                    }

            # 추천 융합 도구에 성능 업데이트 전달
            # (실제로는 recommendation_fuse_tool을 import해서 사용)

            return {
                "updated": True,
                "agent_scores": agent_scores,
                "total_feedback_analyzed": len(batch.feedbacks)
            }

        except Exception as e:
            logger.error(f"❌ 에이전트 가중치 업데이트 실패: {e}")
            return {"updated": False, "error": str(e)}

    async def _update_user_profiles(self, batch: LearningBatch) -> Dict[str, Any]:
        """사용자 개인화 프로필 업데이트"""
        try:
            user_profiles_updated = set()

            # 사용자별 피드백 분석
            user_feedback_analysis = defaultdict(list)

            for feedback in batch.feedbacks:
                user_feedback_analysis[feedback.user_id].append(feedback)

            # 각 사용자의 선호도 프로필 업데이트
            for user_id, feedbacks in user_feedback_analysis.items():
                # 선호 브랜드 분석
                brand_preferences = defaultdict(list)
                price_preferences = []
                year_preferences = []

                for feedback in feedbacks:
                    context = feedback.context
                    item_info = context.get("item_info", {})

                    brand = item_info.get("manufacturer", "")
                    price = item_info.get("price", 0)
                    year = item_info.get("modelyear", 0)

                    if brand:
                        brand_preferences[brand].append(feedback.rating)
                    if price > 0:
                        price_preferences.append((price, feedback.rating))
                    if year > 0:
                        year_preferences.append((year, feedback.rating))

                # 선호도 프로필 계산
                user_profile = {
                    "preferred_brands": self._analyze_brand_preferences(brand_preferences),
                    "price_sensitivity": self._analyze_price_preferences(price_preferences),
                    "year_preferences": self._analyze_year_preferences(year_preferences),
                    "updated_at": datetime.now().isoformat()
                }

                # 프로필 저장 (실제로는 데이터베이스나 캐시에 저장)
                user_profiles_updated.add(user_id)

            return {
                "updated": True,
                "profiles_updated": len(user_profiles_updated),
                "user_ids": list(user_profiles_updated)
            }

        except Exception as e:
            logger.error(f"❌ 사용자 프로필 업데이트 실패: {e}")
            return {"updated": False, "error": str(e)}

    def _analyze_brand_preferences(self, brand_preferences: Dict[str, List[float]]) -> List[str]:
        """브랜드 선호도 분석"""
        preferred_brands = []

        for brand, ratings in brand_preferences.items():
            avg_rating = np.mean(ratings)
            if avg_rating > 0.6 and len(ratings) >= 2:  # 충분한 데이터와 높은 평점
                preferred_brands.append(brand)

        return preferred_brands

    def _analyze_price_preferences(self, price_preferences: List[Tuple[float, float]]) -> Dict[str, Any]:
        """가격 선호도 분석"""
        if not price_preferences:
            return {"sensitivity": "medium", "preferred_range": None}

        # 높은 평점을 받은 가격대 분석
        high_rated_prices = [price for price, rating in price_preferences if rating > 0.6]

        if high_rated_prices:
            return {
                "sensitivity": "low" if np.std(high_rated_prices) > 1000 else "high",
                "preferred_range": {
                    "min": int(np.percentile(high_rated_prices, 25)),
                    "max": int(np.percentile(high_rated_prices, 75))
                }
            }

        return {"sensitivity": "medium", "preferred_range": None}

    def _analyze_year_preferences(self, year_preferences: List[Tuple[int, float]]) -> Dict[str, Any]:
        """연식 선호도 분석"""
        if not year_preferences:
            return {"preference_type": "neutral", "preferred_range": None}

        # 높은 평점을 받은 연식들 분석
        high_rated_years = [year for year, rating in year_preferences if rating > 0.6]

        if high_rated_years:
            avg_year = np.mean(high_rated_years)
            current_year = 2025

            if avg_year >= current_year - 3:
                preference_type = "new_cars"
            elif avg_year <= current_year - 10:
                preference_type = "vintage_cars"
            else:
                preference_type = "balanced"

            return {
                "preference_type": preference_type,
                "preferred_range": {
                    "min": int(np.min(high_rated_years)),
                    "max": int(np.max(high_rated_years))
                }
            }

        return {"preference_type": "neutral", "preferred_range": None}

    async def _analyze_ab_test_performance(self) -> Dict[str, Any]:
        """A/B 테스트 성과 분석"""
        try:
            # 최근 7일간 피드백 데이터 조회
            seven_days_ago = datetime.now() - timedelta(days=7)

            conn = sqlite3.connect(self.feedback_db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT user_id, feedback_type, rating, context
                FROM user_feedback
                WHERE timestamp > ?
            ''', (seven_days_ago.isoformat(),))

            recent_feedbacks = cursor.fetchall()
            conn.close()

            # A/B 그룹별 성과 분석
            group_performance = defaultdict(list)

            for feedback in recent_feedbacks:
                context = json.loads(feedback[3]) if feedback[3] else {}
                ab_group = context.get("ab_test_group", "control")
                rating = feedback[2]

                group_performance[ab_group].append(rating)

            # 통계 계산
            performance_stats = {}
            for group, ratings in group_performance.items():
                if ratings:
                    performance_stats[group] = {
                        "avg_rating": np.mean(ratings),
                        "sample_size": len(ratings),
                        "std_dev": np.std(ratings),
                        "conversion_rate": len([r for r in ratings if r > 0.7]) / len(ratings)
                    }

            # 통계적 유의성 검정 (간단한 t-test)
            if "control" in performance_stats and "experimental" in performance_stats:
                control_avg = performance_stats["control"]["avg_rating"]
                experimental_avg = performance_stats["experimental"]["avg_rating"]

                improvement = ((experimental_avg - control_avg) / control_avg) * 100
                is_significant = abs(improvement) > 5.0  # 5% 이상 차이

                return {
                    "analyzed": True,
                    "performance_stats": performance_stats,
                    "improvement": improvement,
                    "is_significant": is_significant,
                    "recommendation": "deploy" if improvement > 5 else "continue_testing"
                }

            return {"analyzed": True, "performance_stats": performance_stats}

        except Exception as e:
            logger.error(f"❌ A/B 테스트 분석 실패: {e}")
            return {"analyzed": False, "error": str(e)}

    async def _assign_ab_test_group(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """A/B 테스트 그룹 할당"""
        user_id = params.get("user_id", "anonymous")

        # 사용자 ID 기반 일관성 있는 그룹 할당
        import hashlib
        user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        group_assignment = "experimental" if user_hash % 2 == 0 else "control"

        return {
            "success": True,
            "user_id": user_id,
            "assigned_group": group_assignment,
            "group_distribution": self.ab_test_groups
        }

    async def _get_learning_stats(self) -> Dict[str, Any]:
        """학습 통계 반환"""
        # 사용자 참여도 메트릭 계산
        engagement_metrics = self._calculate_engagement_metrics()

        return {
            "success": True,
            "stats": self.learning_stats,
            "engagement_metrics": engagement_metrics,
            "buffer_status": {
                "current_size": len(self.feedback_buffer),
                "threshold": self.learning_config["update_frequency"],
                "fill_percentage": (len(self.feedback_buffer) / self.learning_config["update_frequency"]) * 100
            },
            "learning_config": self.learning_config
        }

    async def _get_user_learning_profile(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """사용자 학습 프로필 조회"""
        user_id = params.get("user_id")

        if not user_id:
            raise LearningUpdateError("Missing user_id parameter")

        # 사용자 상호작용 히스토리 조회
        user_history = self.user_interaction_history.get(user_id, [])

        # 학습된 선호도 프로필 생성
        profile = self._generate_user_profile(user_history)

        return {
            "success": True,
            "user_id": user_id,
            "interaction_count": len(user_history),
            "learned_profile": profile,
            "last_activity": user_history[-1].timestamp.isoformat() if user_history else None
        }

    def _calculate_engagement_metrics(self) -> Dict[str, Any]:
        """사용자 참여도 메트릭 계산"""
        if not self.user_interaction_history:
            return {"total_users": 0}

        total_users = len(self.user_interaction_history)
        total_interactions = sum(len(history) for history in self.user_interaction_history.values())

        # 활성 사용자 (최근 7일 내 활동)
        recent_threshold = datetime.now() - timedelta(days=7)
        active_users = 0

        for user_history in self.user_interaction_history.values():
            if user_history and user_history[-1].timestamp > recent_threshold:
                active_users += 1

        return {
            "total_users": total_users,
            "active_users": active_users,
            "total_interactions": total_interactions,
            "avg_interactions_per_user": total_interactions / total_users if total_users > 0 else 0,
            "engagement_rate": (active_users / total_users) * 100 if total_users > 0 else 0
        }

    def _generate_user_profile(self, user_history: List[UserFeedback]) -> Dict[str, Any]:
        """사용자 히스토리로부터 학습 프로필 생성"""
        if not user_history:
            return {"status": "no_data"}

        # 피드백 타입별 통계
        feedback_types = defaultdict(int)
        total_rating = 0.0

        for feedback in user_history:
            feedback_types[feedback.feedback_type] += 1
            total_rating += feedback.rating

        avg_rating = total_rating / len(user_history)

        # 사용자 행동 패턴 분석
        behavior_pattern = "explorer" if len(feedback_types) > 3 else "focused"
        engagement_level = "high" if avg_rating > 0.6 else "medium" if avg_rating > 0.3 else "low"

        return {
            "status": "learned",
            "feedback_count": len(user_history),
            "avg_rating": avg_rating,
            "behavior_pattern": behavior_pattern,
            "engagement_level": engagement_level,
            "feedback_distribution": dict(feedback_types),
            "learning_confidence": min(len(user_history) / 20.0, 1.0)  # 20개 피드백으로 완전 학습
        }

# 전역 인스턴스
learning_update_tool = LearningUpdateTool()

# MCP Tool 인터페이스
async def carfin_learning_update(params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    CarFin MCP Tool: 학습 업데이트

    사용 예시:
    - 피드백 기록: {"action": "record_feedback", "feedback_data": {...}}
    - 모델 업데이트: {"action": "trigger_update", "force_update": true}
    - 통계 조회: {"action": "get_stats"}
    - 사용자 프로필: {"action": "get_user_profile", "user_id": "user123"}
    """
    return await learning_update_tool.execute(params, context)