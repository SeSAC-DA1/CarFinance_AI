// Next.js API Route: 사용자 행동 추적
import { NextRequest, NextResponse } from 'next/server';
import { experimentManager } from '@/lib/ab-testing/experiment-manager';

interface InteractionData {
  sessionId: string;
  userId: string;
  interactions: Array<{
    id: string;
    userId: string;
    vehicleId: string;
    type: string;
    timestamp: Date;
    confidence: number;
    duration?: number;
    context?: any;
  }>;
  context: {
    sessionId: string;
    currentPage: string;
    viewedVehicles: string[];
    searchQueries: string[];
    timeSpent: number;
    clickPattern: Array<{
      vehicleId: string;
      timestamp: Date;
      action: string;
    }>;
  };
  preferences: {
    totalInteractions: number;
    uniqueVehiclesViewed: number;
    averageViewDuration: number;
    topPreferences: Array<{
      vehicleId: string;
      score: number;
      viewCount: number;
    }>;
    engagementLevel: 'low' | 'medium' | 'high' | 'very_high';
  };
}

export async function POST(request: NextRequest) {
  try {
    const data: InteractionData = await request.json();
    console.log('📊 사용자 행동 데이터 수신:', {
      userId: data.userId,
      sessionId: data.sessionId,
      interactionCount: data.interactions.length,
      engagementLevel: data.preferences.engagementLevel
    });

    // 🔄 상호작용 데이터 처리
    await processInteractionData(data);

    // 🧪 A/B 테스트 메트릭 기록
    await recordExperimentMetrics(data);

    // 📈 실시간 추천 개선을 위한 데이터 저장
    await updateUserModel(data);

    return NextResponse.json({
      success: true,
      message: '행동 데이터가 성공적으로 기록되었습니다',
      processed: {
        interactions: data.interactions.length,
        sessionDuration: data.context.timeSpent,
        engagementLevel: data.preferences.engagementLevel
      }
    });

  } catch (error) {
    console.error('❌ 행동 추적 API 오류:', error);

    return NextResponse.json(
      {
        error: '행동 데이터 처리 중 오류가 발생했습니다',
        message: error instanceof Error ? error.message : '알 수 없는 오류'
      },
      { status: 500 }
    );
  }
}

// 상호작용 데이터 처리
async function processInteractionData(data: InteractionData) {
  const { userId, interactions, preferences } = data;

  // 📊 사용자 선호도 업데이트
  console.log('🔍 사용자 선호도 분석:', {
    userId,
    totalInteractions: preferences.totalInteractions,
    uniqueVehicles: preferences.uniqueVehiclesViewed,
    avgViewDuration: preferences.averageViewDuration,
    engagementLevel: preferences.engagementLevel
  });

  // 💾 실제 환경에서는 데이터베이스에 저장
  // await db.interactions.createMany({ data: interactions });
  // await db.userPreferences.upsert({ where: { userId }, data: preferences });

  // 📈 높은 참여도 사용자 감지
  if (preferences.engagementLevel === 'very_high') {
    console.log('🔥 고참여 사용자 감지:', userId);
    // 실시간 개인화 추천 트리거 가능
  }

  // 🎯 관심 차량 패턴 분석
  const topVehicles = preferences.topPreferences.slice(0, 5);
  if (topVehicles.length > 0) {
    console.log('🚗 관심 차량 TOP 5:', topVehicles);
    // 유사 차량 추천 준비
  }
}

// A/B 테스트 메트릭 기록
async function recordExperimentMetrics(data: InteractionData) {
  const { userId, interactions } = data;

  // 클릭 수 기록
  const clicks = interactions.filter(i => i.type === 'view').length;
  if (clicks > 0) {
    experimentManager.recordMetric(
      'algorithm_comparison_v1',
      userId,
      'click',
      clicks
    );
  }

  // 좋아요 수 기록
  const likes = interactions.filter(i => i.type === 'like').length;
  if (likes > 0) {
    experimentManager.recordMetric(
      'algorithm_comparison_v1',
      userId,
      'like',
      likes
    );
  }

  // 문의 수 기록
  const inquiries = interactions.filter(i => i.type === 'inquiry').length;
  if (inquiries > 0) {
    experimentManager.recordMetric(
      'algorithm_comparison_v1',
      userId,
      'inquiry',
      inquiries
    );
  }

  console.log('🧪 A/B 테스트 메트릭 기록:', {
    userId,
    clicks,
    likes,
    inquiries
  });
}

// 사용자 모델 업데이트
async function updateUserModel(data: InteractionData) {
  const { userId, preferences, context } = data;

  // 🧠 실시간 학습: 사용자 선호도 패턴 업데이트
  const learningSignals = {
    // 시간 기반 패턴
    activeTimeOfDay: getActiveTimeOfDay(data.interactions),

    // 선호 차량 특성
    preferredBrands: extractPreferredBrands(preferences.topPreferences),
    preferredPriceRange: estimatePreferredPriceRange(preferences.topPreferences),

    // 행동 패턴
    browsingStyle: analyzeBrowsingStyle(context.clickPattern),
    decisionSpeed: analyzeDecisionSpeed(data.interactions),

    // 참여도 지표
    engagementTrend: preferences.engagementLevel,
    sessionActivity: context.timeSpent > 300000 // 5분 이상
  };

  console.log('🧠 사용자 모델 업데이트:', {
    userId,
    learningSignals
  });

  // 📈 실제 환경에서는 추천 모델에 피드백
  // await updateCollaborativeFilteringModel(userId, learningSignals);
  // await updateContentBasedModel(userId, learningSignals);
}

// 헬퍼 함수들
function getActiveTimeOfDay(interactions: any[]): string {
  const hourCounts = new Map<number, number>();

  interactions.forEach(interaction => {
    const hour = new Date(interaction.timestamp).getHours();
    hourCounts.set(hour, (hourCounts.get(hour) || 0) + 1);
  });

  const mostActiveHour = Array.from(hourCounts.entries())
    .sort((a, b) => b[1] - a[1])[0]?.[0] || 12;

  if (mostActiveHour >= 6 && mostActiveHour < 12) return 'morning';
  if (mostActiveHour >= 12 && mostActiveHour < 18) return 'afternoon';
  if (mostActiveHour >= 18 && mostActiveHour < 22) return 'evening';
  return 'night';
}

function extractPreferredBrands(topPreferences: any[]): string[] {
  // 실제로는 vehicleId로부터 브랜드 정보 추출
  return ['현대', '기아', 'BMW']; // 예시
}

function estimatePreferredPriceRange(topPreferences: any[]): { min: number; max: number } {
  // 실제로는 선호 차량들의 가격 분석
  return { min: 2000, max: 5000 }; // 예시
}

function analyzeBrowsingStyle(clickPattern: any[]): 'quick_scanner' | 'detail_oriented' | 'comparison_focused' {
  const avgTimePerVehicle = clickPattern.length > 0
    ? clickPattern.reduce((sum, click, index, arr) => {
        if (index === 0) return 0;
        return sum + (new Date(click.timestamp).getTime() - new Date(arr[index-1].timestamp).getTime());
      }, 0) / Math.max(1, clickPattern.length - 1)
    : 0;

  if (avgTimePerVehicle < 5000) return 'quick_scanner'; // 5초 미만
  if (avgTimePerVehicle > 30000) return 'detail_oriented'; // 30초 이상
  return 'comparison_focused';
}

function analyzeDecisionSpeed(interactions: any[]): 'impulsive' | 'deliberate' | 'researcher' {
  const totalDuration = interactions.reduce((sum, i) => sum + (i.duration || 0), 0);
  const avgDuration = totalDuration / Math.max(1, interactions.length);

  if (avgDuration < 10000) return 'impulsive'; // 10초 미만
  if (avgDuration > 60000) return 'researcher'; // 1분 이상
  return 'deliberate';
}

// GET 요청: 사용자 행동 분석 조회
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams;
  const userId = searchParams.get('userId');

  if (!userId) {
    return NextResponse.json(
      { error: 'userId 파라미터가 필요합니다' },
      { status: 400 }
    );
  }

  // 🔍 사용자 행동 분석 결과 반환 (실제로는 DB에서 조회)
  const mockAnalysis = {
    userId,
    summary: {
      totalSessions: 5,
      totalInteractions: 47,
      averageSessionDuration: 8.5, // 분
      engagementLevel: 'high',
      preferredBrowsingTime: 'evening'
    },
    preferences: {
      brands: ['현대', 'BMW', '기아'],
      priceRange: { min: 2500, max: 6000 },
      categories: ['sedan', 'suv'],
      features: ['연비', '안전성', '디자인']
    },
    recommendations: {
      algorithmPerformance: {
        collaborative: { ctr: 0.15, conversionRate: 0.08 },
        content: { ctr: 0.12, conversionRate: 0.06 },
        hybrid: { ctr: 0.18, conversionRate: 0.10 }
      },
      bestAlgorithm: 'hybrid'
    }
  };

  return NextResponse.json(mockAnalysis);
}