"""
CarFin MCP Tool: Recommendation Fuse
멀티 에이전트 추천 결과 지능형 융합
"""

import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import entropy

logger = logging.getLogger("CarFin-MCP.RecommendationFuse")

class RecommendationFuseError(Exception):
    """추천 융합 관련 에러"""
    pass

@dataclass
class AgentRecommendation:
    agent_name: str
    recommendations: List[Dict[str, Any]]
    confidence: float
    execution_time: float
    reasoning: str
    metadata: Dict[str, Any]

@dataclass
class FusionWeights:
    vehicle_expert: float = 0.3
    finance_expert: float = 0.25
    gemini_agent: float = 0.3
    ncf_model: float = 0.15

class RecommendationFuseTool:
    """멀티 에이전트 추천 결과 융합 도구"""

    def __init__(self):
        # 융합 전략 설정
        self.fusion_strategies = {
            "weighted_average": self._weighted_average_fusion,
            "borda_count": self._borda_count_fusion,
            "condorcet": self._condorcet_fusion,
            "entropy_weighted": self._entropy_weighted_fusion,
            "hybrid": self._hybrid_fusion
        }

        # 동적 가중치 학습
        self.agent_performance_history = {
            "vehicle_expert": {"accuracy": [], "user_feedback": []},
            "finance_expert": {"accuracy": [], "user_feedback": []},
            "gemini_agent": {"accuracy": [], "user_feedback": []},
            "ncf_model": {"accuracy": [], "user_feedback": []}
        }

        # 융합 통계
        self.fusion_stats = {
            "total_fusions": 0,
            "strategy_usage": {},
            "avg_fusion_time": 0.0,
            "consensus_rate": 0.0
        }

        logger.info("✅ RecommendationFuse Tool 초기화 완료")

    async def execute(self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        추천 융합 실행

        params:
            agent_results: 각 에이전트의 추천 결과
            ncf_predictions: NCF 모델 예측 결과
            fusion_strategy: 융합 전략 선택
            user_profile: 사용자 프로필 (개인화 융합용)
            diversity_factor: 다양성 촉진 요소
        """
        try:
            start_time = datetime.now()

            agent_results = params.get("agent_results", [])
            ncf_predictions = params.get("ncf_predictions", {})
            fusion_strategy = params.get("fusion_strategy", "hybrid")
            user_profile = params.get("user_profile", {})
            diversity_factor = params.get("diversity_factor", 0.1)

            # 입력 검증
            if not agent_results and not ncf_predictions:
                raise RecommendationFuseError("No agent results or NCF predictions provided")

            # 에이전트 결과 구조화
            structured_results = await self._structure_agent_results(agent_results, ncf_predictions)

            # 융합 전략 실행
            if fusion_strategy not in self.fusion_strategies:
                fusion_strategy = "hybrid"

            fusion_func = self.fusion_strategies[fusion_strategy]
            fused_recommendations = await fusion_func(structured_results, user_profile, diversity_factor)

            # 후처리: 다양성 및 신뢰도 조정
            final_recommendations = await self._post_process_recommendations(
                fused_recommendations, user_profile, diversity_factor
            )

            # 융합 메타데이터 생성
            fusion_metadata = await self._generate_fusion_metadata(
                structured_results, final_recommendations, fusion_strategy
            )

            # 통계 업데이트
            execution_time = (datetime.now() - start_time).total_seconds()
            self._update_fusion_stats(fusion_strategy, execution_time, len(final_recommendations))

            logger.info(f"🔄 추천 융합 완료: {fusion_strategy} 전략, {len(final_recommendations)}개 결과, {execution_time:.3f}초")

            return {
                "success": True,
                "fused_recommendations": final_recommendations,
                "fusion_strategy": fusion_strategy,
                "fusion_metadata": fusion_metadata,
                "execution_time": execution_time,
                "diversity_score": self._calculate_diversity_score(final_recommendations)
            }

        except Exception as e:
            logger.error(f"❌ 추천 융합 실패: {e}")
            raise RecommendationFuseError(f"Fusion execution failed: {e}")

    async def _structure_agent_results(
        self,
        agent_results: List[Dict[str, Any]],
        ncf_predictions: Dict[str, Any]
    ) -> List[AgentRecommendation]:
        """에이전트 결과 구조화"""
        structured = []

        # 에이전트 결과 처리
        for result in agent_results:
            if isinstance(result, dict) and "agent_name" in result:
                structured.append(AgentRecommendation(
                    agent_name=result.get("agent_name", "unknown"),
                    recommendations=result.get("recommendations", []),
                    confidence=result.get("confidence", 0.5),
                    execution_time=result.get("execution_time", 0.0),
                    reasoning=result.get("reasoning", ""),
                    metadata=result.get("metadata", {})
                ))

        # NCF 결과 처리
        if ncf_predictions and "predictions" in ncf_predictions:
            ncf_recs = []
            for pred in ncf_predictions["predictions"]:
                ncf_recs.append({
                    "vehicle_id": pred.get("item_id"),
                    "score": pred.get("score", 0.0),
                    "ncf_score": pred.get("raw_ncf_score", 0.0),
                    "source": "ncf_model"
                })

            structured.append(AgentRecommendation(
                agent_name="ncf_model",
                recommendations=ncf_recs,
                confidence=0.8,
                execution_time=ncf_predictions.get("execution_time", 0.0),
                reasoning="Neural Collaborative Filtering",
                metadata=ncf_predictions.get("model_info", {})
            ))

        return structured

    async def _weighted_average_fusion(
        self,
        structured_results: List[AgentRecommendation],
        user_profile: Dict[str, Any],
        diversity_factor: float
    ) -> List[Dict[str, Any]]:
        """가중평균 융합"""
        # 동적 가중치 계산
        weights = await self._calculate_dynamic_weights(structured_results, user_profile)

        # 아이템별 점수 집계
        item_scores = {}

        for agent_result in structured_results:
            agent_weight = weights.get(agent_result.agent_name, 0.25)

            for rec in agent_result.recommendations:
                item_id = rec.get("vehicle_id") or rec.get("item_id")
                if not item_id:
                    continue

                score = rec.get("score", 0.0)
                weighted_score = score * agent_weight * agent_result.confidence

                if item_id not in item_scores:
                    item_scores[item_id] = {
                        "total_score": 0.0,
                        "contributing_agents": [],
                        "raw_scores": {},
                        "item_data": rec
                    }

                item_scores[item_id]["total_score"] += weighted_score
                item_scores[item_id]["contributing_agents"].append(agent_result.agent_name)
                item_scores[item_id]["raw_scores"][agent_result.agent_name] = score

        # 결과 정렬 및 반환
        sorted_items = sorted(
            item_scores.items(),
            key=lambda x: x[1]["total_score"],
            reverse=True
        )

        fused_recs = []
        for item_id, data in sorted_items:
            fused_recs.append({
                "vehicle_id": item_id,
                "fused_score": data["total_score"],
                "contributing_agents": data["contributing_agents"],
                "agent_scores": data["raw_scores"],
                "consensus_level": len(data["contributing_agents"]) / len(structured_results),
                **data["item_data"]  # 원본 아이템 데이터 포함
            })

        return fused_recs

    async def _borda_count_fusion(
        self,
        structured_results: List[AgentRecommendation],
        user_profile: Dict[str, Any],
        diversity_factor: float
    ) -> List[Dict[str, Any]]:
        """보르다 카운트 융합 (순위 기반)"""
        item_borda_scores = {}

        for agent_result in structured_results:
            recommendations = agent_result.recommendations
            num_items = len(recommendations)

            for rank, rec in enumerate(recommendations):
                item_id = rec.get("vehicle_id") or rec.get("item_id")
                if not item_id:
                    continue

                # 보르다 점수 계산 (높은 순위일수록 높은 점수)
                borda_score = (num_items - rank) * agent_result.confidence

                if item_id not in item_borda_scores:
                    item_borda_scores[item_id] = {
                        "borda_score": 0.0,
                        "appearing_agents": [],
                        "item_data": rec
                    }

                item_borda_scores[item_id]["borda_score"] += borda_score
                item_borda_scores[item_id]["appearing_agents"].append(agent_result.agent_name)

        # 결과 정렬
        sorted_items = sorted(
            item_borda_scores.items(),
            key=lambda x: x[1]["borda_score"],
            reverse=True
        )

        fused_recs = []
        for item_id, data in sorted_items:
            fused_recs.append({
                "vehicle_id": item_id,
                "borda_score": data["borda_score"],
                "appearing_agents": data["appearing_agents"],
                "consensus_level": len(data["appearing_agents"]) / len(structured_results),
                **data["item_data"]
            })

        return fused_recs

    async def _entropy_weighted_fusion(
        self,
        structured_results: List[AgentRecommendation],
        user_profile: Dict[str, Any],
        diversity_factor: float
    ) -> List[Dict[str, Any]]:
        """엔트로피 기반 가중 융합"""
        # 각 에이전트의 예측 분포에서 엔트로피 계산
        agent_entropies = {}

        for agent_result in structured_results:
            scores = [rec.get("score", 0.0) for rec in agent_result.recommendations]
            if scores:
                # 확률 분포로 정규화
                scores_array = np.array(scores)
                probabilities = scores_array / np.sum(scores_array) if np.sum(scores_array) > 0 else scores_array
                agent_entropy = entropy(probabilities + 1e-10)  # 로그(0) 방지
                agent_entropies[agent_result.agent_name] = agent_entropy

        # 낮은 엔트로피 = 높은 확신 = 높은 가중치
        max_entropy = max(agent_entropies.values()) if agent_entropies else 1.0
        entropy_weights = {
            agent: (max_entropy - ent + 0.1) / (max_entropy + 0.1)
            for agent, ent in agent_entropies.items()
        }

        # 가중평균 융합에 엔트로피 가중치 적용
        weighted_results = []
        for agent_result in structured_results:
            entropy_weight = entropy_weights.get(agent_result.agent_name, 0.25)
            for rec in agent_result.recommendations:
                weighted_rec = rec.copy()
                weighted_rec["entropy_adjusted_score"] = rec.get("score", 0.0) * entropy_weight
                weighted_rec["source_agent"] = agent_result.agent_name
                weighted_results.append(weighted_rec)

        return weighted_results

    async def _hybrid_fusion(
        self,
        structured_results: List[AgentRecommendation],
        user_profile: Dict[str, Any],
        diversity_factor: float
    ) -> List[Dict[str, Any]]:
        """하이브리드 융합 (여러 전략 결합)"""
        # 1. 가중평균 결과
        weighted_results = await self._weighted_average_fusion(structured_results, user_profile, diversity_factor)

        # 2. 보르다 카운트 결과
        borda_results = await self._borda_count_fusion(structured_results, user_profile, diversity_factor)

        # 3. 엔트로피 가중 결과
        entropy_results = await self._entropy_weighted_fusion(structured_results, user_profile, diversity_factor)

        # 결과 통합
        item_hybrid_scores = {}

        # 가중평균 점수 통합
        for i, rec in enumerate(weighted_results[:20]):  # 상위 20개
            item_id = rec["vehicle_id"]
            rank_score = (20 - i) / 20.0  # 순위 점수

            if item_id not in item_hybrid_scores:
                item_hybrid_scores[item_id] = {
                    "weighted_rank": 0,
                    "borda_rank": 0,
                    "entropy_rank": 0,
                    "item_data": rec
                }

            item_hybrid_scores[item_id]["weighted_rank"] = rank_score

        # 보르다 카운트 점수 통합
        for i, rec in enumerate(borda_results[:20]):
            item_id = rec["vehicle_id"]
            rank_score = (20 - i) / 20.0

            if item_id in item_hybrid_scores:
                item_hybrid_scores[item_id]["borda_rank"] = rank_score

        # 최종 하이브리드 점수 계산
        for item_data in item_hybrid_scores.values():
            hybrid_score = (
                0.4 * item_data["weighted_rank"] +
                0.4 * item_data["borda_rank"] +
                0.2 * item_data["entropy_rank"]
            )
            item_data["hybrid_score"] = hybrid_score

        # 결과 정렬
        sorted_items = sorted(
            item_hybrid_scores.items(),
            key=lambda x: x[1]["hybrid_score"],
            reverse=True
        )

        hybrid_recs = []
        for item_id, data in sorted_items:
            hybrid_recs.append({
                "vehicle_id": item_id,
                "hybrid_score": data["hybrid_score"],
                "weighted_rank": data["weighted_rank"],
                "borda_rank": data["borda_rank"],
                "fusion_method": "hybrid",
                **data["item_data"]
            })

        return hybrid_recs

    async def _calculate_dynamic_weights(
        self,
        structured_results: List[AgentRecommendation],
        user_profile: Dict[str, Any]
    ) -> Dict[str, float]:
        """동적 가중치 계산"""
        base_weights = {
            "vehicle_expert": 0.3,
            "finance_expert": 0.25,
            "gemini_agent": 0.3,
            "ncf_model": 0.15
        }

        # 사용자 프로필 기반 조정
        user_focus = user_profile.get("focus", "balanced")

        if user_focus == "budget_focused":
            base_weights["finance_expert"] += 0.1
            base_weights["vehicle_expert"] -= 0.05
            base_weights["gemini_agent"] -= 0.05

        elif user_focus == "performance_focused":
            base_weights["vehicle_expert"] += 0.1
            base_weights["finance_expert"] -= 0.05
            base_weights["gemini_agent"] -= 0.05

        elif user_focus == "ai_powered":
            base_weights["ncf_model"] += 0.1
            base_weights["gemini_agent"] += 0.05
            base_weights["vehicle_expert"] -= 0.075
            base_weights["finance_expert"] -= 0.075

        # 에이전트 성능 이력 기반 조정
        for agent_result in structured_results:
            agent_name = agent_result.agent_name
            if agent_name in self.agent_performance_history:
                history = self.agent_performance_history[agent_name]
                if history["user_feedback"]:
                    avg_feedback = np.mean(history["user_feedback"])
                    if avg_feedback > 0.7:  # 좋은 피드백
                        base_weights[agent_name] *= 1.1
                    elif avg_feedback < 0.3:  # 나쁜 피드백
                        base_weights[agent_name] *= 0.9

        # 가중치 정규화
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v / total_weight for k, v in base_weights.items()}

        return normalized_weights

    async def _post_process_recommendations(
        self,
        fused_recommendations: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        diversity_factor: float
    ) -> List[Dict[str, Any]]:
        """추천 결과 후처리"""
        if not fused_recommendations:
            return []

        # 다양성 증진
        if diversity_factor > 0:
            fused_recommendations = await self._enhance_diversity(fused_recommendations, diversity_factor)

        # 신뢰도 점수 추가
        for rec in fused_recommendations:
            rec["confidence_score"] = self._calculate_confidence_score(rec)

        # 개인화 점수 추가
        for rec in fused_recommendations:
            rec["personalization_score"] = self._calculate_personalization_score(rec, user_profile)

        # 최종 정렬 (신뢰도 + 개인화 고려)
        for rec in fused_recommendations:
            final_score = (
                rec.get("fused_score", rec.get("hybrid_score", 0.0)) * 0.6 +
                rec.get("confidence_score", 0.0) * 0.25 +
                rec.get("personalization_score", 0.0) * 0.15
            )
            rec["final_score"] = final_score

        fused_recommendations.sort(key=lambda x: x.get("final_score", 0.0), reverse=True)

        return fused_recommendations[:20]  # 상위 20개만 반환

    async def _enhance_diversity(
        self,
        recommendations: List[Dict[str, Any]],
        diversity_factor: float
    ) -> List[Dict[str, Any]]:
        """다양성 증진"""
        if diversity_factor <= 0 or len(recommendations) <= 1:
            return recommendations

        enhanced_recs = [recommendations[0]]  # 최고 점수는 항상 포함

        for candidate in recommendations[1:]:
            # 기존 선택된 아이템들과의 유사도 계산
            similarity_penalty = 0.0

            for selected in enhanced_recs:
                similarity = self._calculate_item_similarity(candidate, selected)
                similarity_penalty += similarity

            # 다양성 보정된 점수 계산
            original_score = candidate.get("fused_score", candidate.get("hybrid_score", 0.0))
            diversity_adjusted_score = original_score - (similarity_penalty * diversity_factor)
            candidate["diversity_adjusted_score"] = diversity_adjusted_score

            enhanced_recs.append(candidate)

        # 다양성 보정 점수로 재정렬
        enhanced_recs.sort(key=lambda x: x.get("diversity_adjusted_score", 0.0), reverse=True)

        return enhanced_recs

    def _calculate_item_similarity(self, item1: Dict[str, Any], item2: Dict[str, Any]) -> float:
        """아이템 간 유사도 계산"""
        similarity = 0.0

        # 브랜드 유사도
        if item1.get("manufacturer") == item2.get("manufacturer"):
            similarity += 0.3

        # 가격대 유사도
        price1 = item1.get("price", 0)
        price2 = item2.get("price", 0)
        if price1 > 0 and price2 > 0:
            price_diff = abs(price1 - price2) / max(price1, price2)
            price_similarity = max(0, 1 - price_diff * 2)  # 50% 차이까지는 유사
            similarity += price_similarity * 0.25

        # 연식 유사도
        year1 = item1.get("modelyear", 2020)
        year2 = item2.get("modelyear", 2020)
        year_diff = abs(year1 - year2)
        year_similarity = max(0, 1 - year_diff / 10.0)  # 10년 차이까지
        similarity += year_similarity * 0.2

        # 차종 유사도
        if item1.get("cartype") == item2.get("cartype"):
            similarity += 0.25

        return min(similarity, 1.0)

    def _calculate_confidence_score(self, recommendation: Dict[str, Any]) -> float:
        """신뢰도 점수 계산"""
        confidence = 0.5  # 기본값

        # 컨센서스 레벨 기반
        consensus_level = recommendation.get("consensus_level", 0.0)
        confidence += consensus_level * 0.3

        # 기여 에이전트 수 기반
        contributing_agents = recommendation.get("contributing_agents", [])
        if len(contributing_agents) >= 3:
            confidence += 0.2
        elif len(contributing_agents) >= 2:
            confidence += 0.1

        return min(confidence, 1.0)

    def _calculate_personalization_score(
        self,
        recommendation: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> float:
        """개인화 점수 계산"""
        personalization = 0.5  # 기본값

        # 예산 적합성
        budget = user_profile.get("budget", {})
        item_price = recommendation.get("price", 0)

        if budget:
            min_budget = budget.get("min", 0)
            max_budget = budget.get("max", float('inf'))

            if min_budget <= item_price <= max_budget:
                personalization += 0.2
            elif item_price > max_budget:
                personalization -= 0.3

        # 브랜드 선호도
        preferred_brands = user_profile.get("preferred_brands", [])
        item_brand = recommendation.get("manufacturer", "")

        if item_brand in preferred_brands:
            personalization += 0.15

        # 연식 선호도
        preferred_year_range = user_profile.get("preferred_year_range", {})
        item_year = recommendation.get("modelyear", 2020)

        if preferred_year_range:
            min_year = preferred_year_range.get("min", 2000)
            max_year = preferred_year_range.get("max", 2025)

            if min_year <= item_year <= max_year:
                personalization += 0.15

        return max(0.0, min(personalization, 1.0))

    def _calculate_diversity_score(self, recommendations: List[Dict[str, Any]]) -> float:
        """추천 리스트의 다양성 점수 계산"""
        if len(recommendations) <= 1:
            return 1.0

        total_similarity = 0.0
        comparisons = 0

        for i in range(len(recommendations)):
            for j in range(i + 1, len(recommendations)):
                similarity = self._calculate_item_similarity(recommendations[i], recommendations[j])
                total_similarity += similarity
                comparisons += 1

        avg_similarity = total_similarity / comparisons if comparisons > 0 else 0.0
        diversity_score = 1.0 - avg_similarity

        return max(0.0, min(diversity_score, 1.0))

    async def _generate_fusion_metadata(
        self,
        structured_results: List[AgentRecommendation],
        final_recommendations: List[Dict[str, Any]],
        fusion_strategy: str
    ) -> Dict[str, Any]:
        """융합 메타데이터 생성"""
        return {
            "fusion_strategy": fusion_strategy,
            "participating_agents": [agent.agent_name for agent in structured_results],
            "agent_contributions": {
                agent.agent_name: {
                    "confidence": agent.confidence,
                    "num_recommendations": len(agent.recommendations),
                    "execution_time": agent.execution_time
                }
                for agent in structured_results
            },
            "fusion_quality_metrics": {
                "diversity_score": self._calculate_diversity_score(final_recommendations),
                "consensus_rate": np.mean([rec.get("consensus_level", 0) for rec in final_recommendations]),
                "avg_confidence": np.mean([rec.get("confidence_score", 0) for rec in final_recommendations])
            },
            "timestamp": datetime.now().isoformat()
        }

    def _update_fusion_stats(self, strategy: str, execution_time: float, num_results: int):
        """융합 통계 업데이트"""
        self.fusion_stats["total_fusions"] += 1

        if strategy not in self.fusion_stats["strategy_usage"]:
            self.fusion_stats["strategy_usage"][strategy] = 0
        self.fusion_stats["strategy_usage"][strategy] += 1

        # 평균 실행 시간 업데이트
        total_fusions = self.fusion_stats["total_fusions"]
        current_avg = self.fusion_stats["avg_fusion_time"]
        self.fusion_stats["avg_fusion_time"] = (
            (current_avg * (total_fusions - 1) + execution_time) / total_fusions
        )

    async def update_agent_performance(self, agent_name: str, user_feedback: float, accuracy: float):
        """에이전트 성능 이력 업데이트"""
        if agent_name in self.agent_performance_history:
            self.agent_performance_history[agent_name]["user_feedback"].append(user_feedback)
            self.agent_performance_history[agent_name]["accuracy"].append(accuracy)

            # 이력 크기 제한 (최근 100개)
            for metric in ["user_feedback", "accuracy"]:
                if len(self.agent_performance_history[agent_name][metric]) > 100:
                    self.agent_performance_history[agent_name][metric] = \
                        self.agent_performance_history[agent_name][metric][-100:]

# 전역 인스턴스
recommendation_fuse_tool = RecommendationFuseTool()

# MCP Tool 인터페이스
async def carfin_recommendation_fuse(params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    CarFin MCP Tool: 추천 융합

    사용 예시:
    - 기본 융합: {"agent_results": [...], "ncf_predictions": {...}, "fusion_strategy": "hybrid"}
    - 다양성 강화: {"agent_results": [...], "diversity_factor": 0.3}
    - 개인화 융합: {"agent_results": [...], "user_profile": {...}}
    """
    return await recommendation_fuse_tool.execute(params, context)