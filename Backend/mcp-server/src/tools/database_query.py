"""
CarFin MCP Tool: Database Query
PostgreSQL AWS RDS 안전한 쿼리 실행 및 데이터 접근
"""

import asyncio
import asyncpg
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import os
from contextlib import asynccontextmanager

logger = logging.getLogger("CarFin-MCP.DatabaseQuery")

class DatabaseQueryError(Exception):
    """데이터베이스 쿼리 관련 에러"""
    pass

class DatabaseQueryTool:
    """PostgreSQL 데이터베이스 쿼리 도구"""

    def __init__(self):
        # 데이터베이스 연결 설정
        self.db_config = {
            "host": "carfin-db.cbkayiqs4div.ap-northeast-2.rds.amazonaws.com",
            "port": 5432,
            "database": "carfin",
            "user": os.getenv("DB_USER", "carfin_user"),
            "password": os.getenv("DB_PASSWORD", "your_password")
        }

        self.connection_pool = None
        self.query_cache = {}  # 쿼리 캐싱
        self.safe_queries = self._init_safe_queries()

        logger.info("✅ DatabaseQuery Tool 초기화 완료")

    async def initialize_pool(self):
        """연결 풀 초기화"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                **self.db_config,
                min_size=5,
                max_size=20,
                command_timeout=30
            )
            logger.info("✅ 데이터베이스 연결 풀 초기화 완료")
        except Exception as e:
            logger.error(f"❌ 데이터베이스 연결 풀 초기화 실패: {e}")
            raise DatabaseQueryError(f"Failed to initialize connection pool: {e}")

    def _init_safe_queries(self) -> Dict[str, str]:
        """안전한 사전 정의 쿼리들"""
        return {
            # 차량 검색 쿼리
            "search_vehicles": """
                SELECT vehicleid, manufacturer, model, modelyear, price, distance,
                       fueltype, cartype, location, detailurl, photo
                FROM vehicles
                WHERE price BETWEEN $1 AND $2
                AND distance <= $3
                AND modelyear >= $4
                AND cartype NOT LIKE '%화물%'
                AND cartype NOT LIKE '%버스%'
                AND cartype NOT LIKE '%특수%'
                ORDER BY price ASC, distance ASC
                LIMIT $5
            """,

            # 인기 차량 조회
            "popular_vehicles": """
                SELECT v.vehicleid, v.manufacturer, v.model, v.modelyear, v.price,
                       COUNT(*) OVER (PARTITION BY v.manufacturer, v.model) as popularity_count
                FROM vehicles v
                WHERE v.price BETWEEN 1000 AND 8000
                AND v.distance <= 100000
                AND v.modelyear >= 2018
                ORDER BY popularity_count DESC, v.modelyear DESC
                LIMIT $1
            """,

            # 브랜드별 통계
            "brand_statistics": """
                SELECT manufacturer,
                       COUNT(*) as total_count,
                       AVG(price) as avg_price,
                       MIN(price) as min_price,
                       MAX(price) as max_price,
                       AVG(distance) as avg_mileage
                FROM vehicles
                WHERE price >= 50 AND price <= 50000
                GROUP BY manufacturer
                ORDER BY total_count DESC
                LIMIT $1
            """,

            # 가격대별 분포
            "price_distribution": """
                SELECT
                    CASE
                        WHEN price < 1000 THEN '1000만원 미만'
                        WHEN price < 3000 THEN '1000-3000만원'
                        WHEN price < 5000 THEN '3000-5000만원'
                        WHEN price < 10000 THEN '5000만원-1억'
                        ELSE '1억 이상'
                    END as price_range,
                    COUNT(*) as count,
                    AVG(distance) as avg_mileage
                FROM vehicles
                WHERE price >= 50 AND price <= 50000
                GROUP BY price_range
                ORDER BY MIN(price)
            """,

            # 연식별 트렌드
            "year_trends": """
                SELECT modelyear,
                       COUNT(*) as count,
                       AVG(price) as avg_price,
                       array_agg(DISTINCT manufacturer) as brands
                FROM vehicles
                WHERE modelyear >= $1 AND modelyear <= $2
                AND price >= 50 AND price <= 50000
                GROUP BY modelyear
                ORDER BY modelyear DESC
            """,

            # 지역별 매물 분포
            "location_distribution": """
                SELECT location,
                       COUNT(*) as count,
                       AVG(price) as avg_price
                FROM vehicles
                WHERE location IS NOT NULL
                AND price >= 50 AND price <= 50000
                GROUP BY location
                ORDER BY count DESC
                LIMIT $1
            """,

            # 특정 차량 상세 정보
            "vehicle_details": """
                SELECT vehicleid, manufacturer, model, modelyear, price, distance,
                       fueltype, cartype, location, detailurl, photo
                FROM vehicles
                WHERE vehicleid = $1
            """,

            # 유사 차량 검색
            "similar_vehicles": """
                SELECT vehicleid, manufacturer, model, modelyear, price, distance,
                       fueltype, cartype, location, detailurl, photo,
                       ABS(price - $2) as price_diff
                FROM vehicles
                WHERE manufacturer = $1
                AND modelyear BETWEEN $3 AND $4
                AND price BETWEEN $5 AND $6
                AND vehicleid != $7
                ORDER BY price_diff ASC, distance ASC
                LIMIT $8
            """,

            # 데이터 품질 체크
            "data_quality_check": """
                SELECT
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN price < 50 OR price > 50000 THEN 1 END) as price_outliers,
                    COUNT(CASE WHEN distance > 500000 THEN 1 END) as mileage_outliers,
                    COUNT(CASE WHEN modelyear < 1990 THEN 1 END) as old_vehicles,
                    COUNT(CASE WHEN vehicleid IS NULL THEN 1 END) as null_ids
                FROM vehicles
            """
        }

    async def execute(self, params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        데이터베이스 쿼리 실행

        params:
            query_type: 사전 정의된 쿼리 타입
            parameters: 쿼리 파라미터
            cache_key: 캐시 키 (선택적)
            use_cache: 캐시 사용 여부 (기본값: True)
        """
        try:
            query_type = params.get("query_type")
            parameters = params.get("parameters", [])
            cache_key = params.get("cache_key")
            use_cache = params.get("use_cache", True)

            if not query_type:
                raise DatabaseQueryError("Missing required parameter: query_type")

            if query_type not in self.safe_queries:
                raise DatabaseQueryError(f"Unknown query type: {query_type}")

            # 캐시 확인
            if use_cache and cache_key and cache_key in self.query_cache:
                cached_result = self.query_cache[cache_key]
                logger.info(f"📋 캐시에서 결과 반환: {cache_key}")
                return {
                    "success": True,
                    "data": cached_result["data"],
                    "cached": True,
                    "cache_timestamp": cached_result["timestamp"]
                }

            # 연결 풀 초기화 (필요시)
            if not self.connection_pool:
                await self.initialize_pool()

            # 쿼리 실행
            query = self.safe_queries[query_type]
            result = await self._execute_query(query, parameters)

            # 결과 후처리
            processed_result = self._process_result(result, query_type)

            # 캐시 저장
            if use_cache and cache_key:
                self.query_cache[cache_key] = {
                    "data": processed_result,
                    "timestamp": datetime.now().isoformat()
                }

            logger.info(f"✅ 쿼리 실행 완료: {query_type} - {len(processed_result) if isinstance(processed_result, list) else 1}건")

            return {
                "success": True,
                "data": processed_result,
                "query_type": query_type,
                "cached": False
            }

        except Exception as e:
            logger.error(f"❌ 데이터베이스 쿼리 실패: {e}")
            raise DatabaseQueryError(f"Query execution failed: {e}")

    async def _execute_query(self, query: str, parameters: List[Any]) -> List[Dict[str, Any]]:
        """실제 쿼리 실행"""
        async with self.connection_pool.acquire() as connection:
            rows = await connection.fetch(query, *parameters)
            return [dict(row) for row in rows]

    def _process_result(self, result: List[Dict[str, Any]], query_type: str) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """결과 후처리"""
        if not result:
            return []

        # 쿼리 타입에 따른 특별 처리
        if query_type == "data_quality_check":
            return result[0] if result else {}

        elif query_type in ["brand_statistics", "price_distribution", "location_distribution"]:
            # 통계 데이터는 숫자 값 정리
            for row in result:
                for key, value in row.items():
                    if isinstance(value, (int, float)) and key != 'count':
                        row[key] = round(float(value), 2)

        elif query_type in ["search_vehicles", "popular_vehicles", "similar_vehicles"]:
            # 차량 데이터는 추가 정보 계산
            for row in result:
                row["age"] = 2025 - (row.get("modelyear", 2025))
                row["price_per_km"] = row.get("price", 0) / max(row.get("distance", 1), 1)

        return result

    # 편의 메서드들
    async def search_vehicles_by_criteria(self, criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """기준에 따른 차량 검색"""
        params = {
            "query_type": "search_vehicles",
            "parameters": [
                criteria.get("min_price", 50),
                criteria.get("max_price", 50000),
                criteria.get("max_mileage", 200000),
                criteria.get("min_year", 2005),
                criteria.get("limit", 50)
            ],
            "cache_key": f"search_{hash(str(sorted(criteria.items())))}"
        }

        result = await self.execute(params)
        return result.get("data", [])

    async def get_popular_vehicles(self, limit: int = 20) -> List[Dict[str, Any]]:
        """인기 차량 조회"""
        params = {
            "query_type": "popular_vehicles",
            "parameters": [limit],
            "cache_key": f"popular_{limit}"
        }

        result = await self.execute(params)
        return result.get("data", [])

    async def get_brand_statistics(self, limit: int = 20) -> List[Dict[str, Any]]:
        """브랜드별 통계"""
        params = {
            "query_type": "brand_statistics",
            "parameters": [limit],
            "cache_key": f"brand_stats_{limit}"
        }

        result = await self.execute(params)
        return result.get("data", [])

    async def get_vehicle_details(self, vehicle_id: str) -> Optional[Dict[str, Any]]:
        """특정 차량 상세 정보"""
        params = {
            "query_type": "vehicle_details",
            "parameters": [vehicle_id],
            "cache_key": f"vehicle_{vehicle_id}"
        }

        result = await self.execute(params)
        data = result.get("data", [])
        return data[0] if data else None

    async def get_similar_vehicles(self, reference_vehicle: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """유사 차량 검색"""
        params = {
            "query_type": "similar_vehicles",
            "parameters": [
                reference_vehicle.get("manufacturer"),
                reference_vehicle.get("price"),
                reference_vehicle.get("modelyear", 2020) - 2,  # ±2년
                reference_vehicle.get("modelyear", 2020) + 2,
                reference_vehicle.get("price", 3000) * 0.8,  # ±20%
                reference_vehicle.get("price", 3000) * 1.2,
                reference_vehicle.get("vehicleid"),
                limit
            ]
        }

        result = await self.execute(params)
        return result.get("data", [])

    async def cleanup(self):
        """리소스 정리"""
        if self.connection_pool:
            await self.connection_pool.close()
            logger.info("🔄 데이터베이스 연결 풀 종료")

# 전역 인스턴스
database_query_tool = DatabaseQueryTool()

# MCP Tool 인터페이스
async def carfin_database_query(params: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    CarFin MCP Tool: 데이터베이스 쿼리

    사용 예시:
    - 차량 검색: {"query_type": "search_vehicles", "parameters": [1000, 5000, 100000, 2018, 20]}
    - 인기 차량: {"query_type": "popular_vehicles", "parameters": [10]}
    - 브랜드 통계: {"query_type": "brand_statistics", "parameters": [15]}
    """
    return await database_query_tool.execute(params, context)