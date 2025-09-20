// Next.js API Route: 추천 시스템 서빙
import { NextRequest, NextResponse } from 'next/server';
import { mockRecommendationEngine } from '@/lib/recommendation/mock-recommendation-engine';
import { experimentManager } from '@/lib/ab-testing/experiment-manager';
import {
  RecommendationRequest,
  RecommendationResponse,
  UserProfile as RecommendationUserProfile
} from '@/types/recommendation';

export async function POST(request: NextRequest) {
  const startTime = Date.now();

  try {
    // 요청 파싱
    const body = await request.json();
    console.log('📥 추천 요청:', body);

    // 요청 검증
    if (!body.userId || !body.userProfile) {
      return NextResponse.json(
        { error: 'userId와 userProfile이 필요합니다' },
        { status: 400 }
      );
    }

    // A/B 테스트: 사용자를 실험군에 할당
    const experimentConfig = experimentManager.getUserExperimentConfig(body.userId);
    console.log('🧪 A/B 테스트 설정:', experimentConfig);

    // 사용자 프로필 변환 및 예산 범위 추정
    const userProfile: RecommendationUserProfile = {
      ...body.userProfile,
      budgetRange: body.userProfile.budgetRange || estimateBudgetRange(body.userProfile),
      createdAt: new Date(),
      lastActiveAt: new Date()
    };

    // 추천 요청 구성
    const recommendationRequest: RecommendationRequest = {
      userId: body.userId,
      userProfile,
      context: {
        type: body.recommendationType || 'personalized',
        currentVehicleId: body.currentVehicleId,
        limit: body.limit || 12,
        searchFilters: body.searchFilters
      },
      excludeVehicleIds: body.excludeVehicleIds || []
    };

    // Enhanced FastAPI 백엔드 호출
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    try {
      const response = await fetch(`${apiUrl}/api/recommendations`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_profile: {
            user_id: recommendationRequest.userId,
            name: userProfile.name || 'Guest',
            age: userProfile.age || 30,
            income: userProfile.income || 4000,
            preferences: userProfile.preferences || ['연비'],
            purpose: userProfile.purpose || 'commute',
            budget_range: userProfile.budgetRange
          },
          recommendation_type: recommendationRequest.context.type,
          limit: recommendationRequest.context.limit,
          exclude_vehicle_ids: recommendationRequest.excludeVehicleIds || []
        })
      });

      if (!response.ok) {
        throw new Error(`FastAPI Error: ${response.statusText}`);
      }

      const recommendations: RecommendationResponse = await response.json();

      // FastAPI 응답을 Next.js 형식으로 변환
      if (recommendations.recommendations) {
        recommendations.recommendations = recommendations.recommendations.map((rec: any) => ({
          ...rec,
          vehicle_id: rec.vehicle_id || `car_${rec.car_id}`,
          score: rec.score || 0.8,
          reasons: Array.isArray(rec.reasons) ? rec.reasons : [rec.reason || '추천 차량'],
          confidence: rec.confidence || rec.score || 0.8
        }));
      }
    } catch (fetchError) {
      console.warn('FastAPI 호출 실패, Mock 엔진 사용:', fetchError);

      // Fallback to mock engine
      recommendations = await mockRecommendationEngine.getRecommendations(
        recommendationRequest
      );
    }

    // A/B 테스트 메트릭 기록 (노출)
    experimentManager.recordMetric(
      'algorithm_comparison_v1',
      body.userId,
      'impression',
      recommendations.recommendations.length
    );

    // 응답 시간 기록
    const processingTime = Date.now() - startTime;
    recommendations.metadata.processingTime = processingTime;

    console.log('✅ 추천 완료:', {
      userId: body.userId,
      count: recommendations.recommendations.length,
      processingTime: `${processingTime}ms`,
      experimentVariant: experimentConfig.algorithm_comparison_v1?.variant
    });

    return NextResponse.json(recommendations);

  } catch (error) {
    console.error('❌ 추천 API 오류:', error);

    return NextResponse.json(
      {
        error: '추천 생성 중 오류가 발생했습니다',
        message: error instanceof Error ? error.message : '알 수 없는 오류',
        timestamp: new Date().toISOString()
      },
      { status: 500 }
    );
  }
}

// GET 요청: 사용자별 빠른 추천 (캐시된 결과)
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const userId = searchParams.get('userId');
  const type = searchParams.get('type') || 'personalized';
  const limit = parseInt(searchParams.get('limit') || '6');

  if (!userId) {
    return NextResponse.json(
      { error: 'userId 파라미터가 필요합니다' },
      { status: 400 }
    );
  }

  try {
    // 간단한 게스트 프로필 생성
    const guestProfile: RecommendationUserProfile = {
      user_id: userId,
      name: 'Guest User',
      budgetRange: { min: 2000, max: 5000 }, // 기본 예산 범위
      guest: true,
      createdAt: new Date(),
      lastActiveAt: new Date()
    };

    // 빠른 추천 요청
    const recommendations = await mockRecommendationEngine.getRecommendations({
      userId,
      userProfile: guestProfile,
      context: {
        type: type as any,
        limit
      }
    });

    return NextResponse.json(recommendations);

  } catch (error) {
    console.error('❌ 빠른 추천 오류:', error);

    return NextResponse.json(
      { error: '추천 생성 실패' },
      { status: 500 }
    );
  }
}

// 예산 범위 추정 함수
function estimateBudgetRange(userProfile: any): { min: number; max: number } {
  const income = userProfile.income || 3000; // 기본값: 3000만원
  const age = userProfile.age || 30;
  const purpose = userProfile.purpose;

  // 연소득 기반 예산 추정
  let minBudget = Math.max(1000, income * 0.2); // 최소 연소득의 20%
  let maxBudget = income * 0.8; // 최대 연소득의 80%

  // 연령대별 조정
  if (age < 30) {
    minBudget *= 0.8; // 젊은 층은 더 저렴한 차량 선호
    maxBudget *= 0.9;
  } else if (age > 45) {
    minBudget *= 1.2; // 중년층은 더 비싼 차량 선호
    maxBudget *= 1.1;
  }

  // 용도별 조정
  switch (purpose) {
    case 'business':
      minBudget *= 1.3;
      maxBudget *= 1.2;
      break;
    case 'family':
      minBudget *= 1.1;
      maxBudget *= 1.1;
      break;
    case 'commute':
      minBudget *= 0.9;
      maxBudget *= 0.95;
      break;
    case 'leisure':
      minBudget *= 1.0;
      maxBudget *= 1.1;
      break;
  }

  return {
    min: Math.round(minBudget),
    max: Math.round(maxBudget)
  };
}