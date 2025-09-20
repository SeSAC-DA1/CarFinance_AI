#!/usr/bin/env python3
"""
CarFin-MCP Server - 멀티에이전트 협업 및 NCF 통합 서버
48시간 베타 버전 - 핵심 기능 우선 구현
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("CarFin-MCP")

# MCP 요청/응답 모델
class MCPRequest(BaseModel):
    tool_name: str = Field(..., description="MCP Tool 이름")
    params: Dict[str, Any] = Field(..., description="Tool 파라미터")
    context: Optional[Dict[str, Any]] = Field(None, description="실행 컨텍스트")
    user_id: Optional[str] = Field(None, description="사용자 ID")

class MCPResponse(BaseModel):
    success: bool = Field(..., description="실행 성공 여부")
    result: Optional[Dict[str, Any]] = Field(None, description="실행 결과")
    error: Optional[str] = Field(None, description="에러 메시지")
    execution_time: float = Field(..., description="실행 시간 (초)")
    tool_name: str = Field(..., description="실행된 Tool 이름")

class RecommendationRequest(BaseModel):
    user_profile: Dict[str, Any] = Field(..., description="사용자 프로필")
    request_type: str = Field("full_recommendation", description="요청 타입")
    limit: int = Field(10, description="추천 결과 개수")

@dataclass
class AgentResult:
    agent_name: str
    result: Dict[str, Any]
    execution_time: float
    confidence: float
    timestamp: datetime

class CarFinMCPServer:
    """CarFin MCP 서버 - 멀티에이전트 협업 오케스트레이터"""

    def __init__(self):
        self.app = FastAPI(
            title="CarFin-MCP Server",
            description="멀티에이전트 협업 및 NCF 딥러닝 통합 서버",
            version="1.0.0-beta"
        )

        # CORS 설정
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:3000", "http://localhost:8000"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # MCP Tools 레지스트리
        self.tools = {}
        self.agent_pool = {}
        self.session_contexts = {}

        # 라우터 설정
        self._setup_routes()

        logger.info("🚀 CarFin-MCP Server 초기화 완료")

    def _setup_routes(self):
        """API 라우터 설정"""

        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "server": "CarFin-MCP",
                "version": "1.0.0-beta",
                "tools_registered": len(self.tools),
                "timestamp": datetime.now().isoformat()
            }

        @self.app.post("/mcp/execute", response_model=MCPResponse)
        async def execute_mcp_tool(request: MCPRequest):
            """MCP Tool 실행"""
            start_time = datetime.now()

            try:
                if request.tool_name not in self.tools:
                    raise HTTPException(
                        status_code=404,
                        detail=f"MCP Tool '{request.tool_name}' not found"
                    )

                tool_func = self.tools[request.tool_name]
                result = await tool_func(request.params, request.context)

                execution_time = (datetime.now() - start_time).total_seconds()

                return MCPResponse(
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    tool_name=request.tool_name
                )

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"❌ MCP Tool '{request.tool_name}' 실행 실패: {e}")

                return MCPResponse(
                    success=False,
                    error=str(e),
                    execution_time=execution_time,
                    tool_name=request.tool_name
                )

        @self.app.post("/mcp/recommend")
        async def orchestrate_recommendation(request: RecommendationRequest):
            """멀티에이전트 협업 추천 오케스트레이션"""
            start_time = datetime.now()

            try:
                logger.info(f"🎯 추천 오케스트레이션 시작: 사용자 {request.user_profile.get('user_id', 'anonymous')}")

                # 1. 3개 에이전트 병렬 실행
                agent_tasks = [
                    self._execute_agent("vehicle_expert", request.user_profile),
                    self._execute_agent("finance_expert", request.user_profile),
                    self._execute_agent("gemini_multi_agent", request.user_profile)
                ]

                # 2. NCF 모델 병렬 추론
                ncf_task = self._execute_ncf_prediction(request.user_profile)

                # 3. 모든 결과 수집
                agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
                ncf_result = await ncf_task

                # 4. 결과 융합
                final_recommendation = await self._fuse_recommendations(
                    agent_results, ncf_result, request.user_profile
                )

                execution_time = (datetime.now() - start_time).total_seconds()

                logger.info(f"✅ 추천 완료: {execution_time:.2f}초")

                return {
                    "success": True,
                    "recommendations": final_recommendation,
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                logger.error(f"❌ 추천 오케스트레이션 실패: {e}")

                return {
                    "success": False,
                    "error": str(e),
                    "execution_time": execution_time,
                    "timestamp": datetime.now().isoformat()
                }

    async def _execute_agent(self, agent_name: str, user_profile: Dict[str, Any]) -> AgentResult:
        """개별 에이전트 실행"""
        start_time = datetime.now()

        try:
            # 에이전트별 실행 로직 (현재는 모킹)
            if agent_name == "vehicle_expert":
                result = await self._mock_vehicle_expert(user_profile)
            elif agent_name == "finance_expert":
                result = await self._mock_finance_expert(user_profile)
            elif agent_name == "gemini_multi_agent":
                result = await self._mock_gemini_agent(user_profile)
            else:
                raise ValueError(f"Unknown agent: {agent_name}")

            execution_time = (datetime.now() - start_time).total_seconds()

            return AgentResult(
                agent_name=agent_name,
                result=result,
                execution_time=execution_time,
                confidence=result.get("confidence", 0.8),
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"❌ 에이전트 '{agent_name}' 실행 실패: {e}")
            return AgentResult(
                agent_name=agent_name,
                result={"error": str(e)},
                execution_time=(datetime.now() - start_time).total_seconds(),
                confidence=0.0,
                timestamp=datetime.now()
            )

    async def _execute_ncf_prediction(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """NCF 모델 추론 실행"""
        try:
            # NCF 모델 추론 (현재는 모킹)
            await asyncio.sleep(0.5)  # 모델 추론 시뮬레이션

            return {
                "algorithm": "NCF",
                "predictions": [
                    {"vehicle_id": f"ncf_{i}", "score": 0.9 - i * 0.1}
                    for i in range(5)
                ],
                "confidence": 0.85,
                "model_version": "v1.0-beta"
            }

        except Exception as e:
            logger.error(f"❌ NCF 모델 추론 실패: {e}")
            return {"error": str(e), "confidence": 0.0}

    async def _fuse_recommendations(
        self,
        agent_results: List[AgentResult],
        ncf_result: Dict[str, Any],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """에이전트 결과 융합"""
        try:
            # 가중평균 기반 결과 융합
            final_recommendations = []

            # 각 에이전트 결과에서 추천 추출
            for agent_result in agent_results:
                if isinstance(agent_result, AgentResult) and "recommendations" in agent_result.result:
                    agent_recs = agent_result.result["recommendations"]
                    for rec in agent_recs[:3]:  # 상위 3개
                        rec["source"] = agent_result.agent_name
                        rec["weight"] = agent_result.confidence
                        final_recommendations.append(rec)

            # NCF 결과 추가
            if "predictions" in ncf_result:
                for pred in ncf_result["predictions"][:3]:
                    final_recommendations.append({
                        "vehicle_id": pred["vehicle_id"],
                        "score": pred["score"],
                        "source": "ncf_model",
                        "weight": ncf_result.get("confidence", 0.8)
                    })

            # 중복 제거 및 정렬
            unique_recs = {}
            for rec in final_recommendations:
                vid = rec.get("vehicle_id")
                if vid and vid not in unique_recs:
                    unique_recs[vid] = rec
                elif vid in unique_recs:
                    # 가중평균으로 점수 업데이트
                    existing = unique_recs[vid]
                    new_weight = (existing.get("weight", 0) + rec.get("weight", 0)) / 2
                    existing["weight"] = new_weight
                    existing["sources"] = existing.get("sources", [existing.get("source", "")]) + [rec.get("source", "")]

            sorted_recs = sorted(
                unique_recs.values(),
                key=lambda x: x.get("weight", 0) * x.get("score", 0),
                reverse=True
            )

            return {
                "vehicles": sorted_recs[:10],  # 상위 10개
                "fusion_method": "weighted_average",
                "agent_contributions": {
                    res.agent_name: res.confidence
                    for res in agent_results if isinstance(res, AgentResult)
                },
                "ncf_contribution": ncf_result.get("confidence", 0.0)
            }

        except Exception as e:
            logger.error(f"❌ 결과 융합 실패: {e}")
            return {"error": str(e), "vehicles": []}

    # Mock 에이전트 함수들 (실제 구현 전 테스트용)
    async def _mock_vehicle_expert(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """차량 전문가 에이전트 모킹"""
        await asyncio.sleep(0.3)
        return {
            "recommendations": [
                {"vehicle_id": "ve_001", "score": 0.95, "reason": "예산 적합"},
                {"vehicle_id": "ve_002", "score": 0.90, "reason": "브랜드 선호"},
                {"vehicle_id": "ve_003", "score": 0.85, "reason": "연비 우수"}
            ],
            "confidence": 0.9,
            "agent": "vehicle_expert"
        }

    async def _mock_finance_expert(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """금융 전문가 에이전트 모킹"""
        await asyncio.sleep(0.4)
        return {
            "recommendations": [
                {"vehicle_id": "fe_001", "score": 0.92, "reason": "대출 조건 우수"},
                {"vehicle_id": "fe_002", "score": 0.88, "reason": "할부 가능"},
                {"vehicle_id": "fe_003", "score": 0.83, "reason": "리스 적합"}
            ],
            "confidence": 0.85,
            "agent": "finance_expert"
        }

    async def _mock_gemini_agent(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Gemini 멀티에이전트 모킹"""
        await asyncio.sleep(0.6)
        return {
            "recommendations": [
                {"vehicle_id": "gm_001", "score": 0.93, "reason": "종합 분석 결과"},
                {"vehicle_id": "gm_002", "score": 0.87, "reason": "사용자 패턴 매칭"},
                {"vehicle_id": "gm_003", "score": 0.82, "reason": "시장 트렌드 반영"}
            ],
            "confidence": 0.88,
            "agent": "gemini_multi_agent"
        }

    def register_tool(self, name: str, func):
        """MCP Tool 등록"""
        self.tools[name] = func
        logger.info(f"✅ MCP Tool 등록: {name}")

    def run(self, host: str = "0.0.0.0", port: int = 9000):
        """MCP 서버 실행"""
        logger.info(f"🚀 CarFin-MCP Server 시작: http://{host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

# 서버 인스턴스 생성
mcp_server = CarFinMCPServer()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CarFin-MCP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=9000, help="Port number")

    args = parser.parse_args()

    mcp_server.run(host=args.host, port=args.port)