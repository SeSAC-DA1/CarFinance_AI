'use client';

// 기존 시스템 통합한 Mock 추천 엔진
import { CollaborativeFilteringEngine } from '@/lib/collaborative-filtering/similarity-engine';
import { mockDataGenerator, MOCK_CARS, MOCK_USERS, MOCK_INTERACTIONS } from '@/lib/data/mock-data';
import {
  RecommendationRequest,
  RecommendationResponse,
  RecommendationResult,
  RecommendationReason,
  UserProfile,
  VehicleFeatures
} from '@/types/recommendation';

export class MockRecommendationEngine {
  private cfEngine: CollaborativeFilteringEngine;
  private isInitialized = false;

  constructor() {
    this.cfEngine = new CollaborativeFilteringEngine();
  }

  // 📚 학습 포인트: 초기화에서 기존 상호작용 데이터로 CF 모델 학습
  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    console.log('🔄 Mock 추천 엔진 초기화 중...');

    // 기존 상호작용 데이터를 CF 엔진에 학습시키기
    const userInteractionMap = new Map<string, Map<string, any[]>>();

    // 1단계: 사용자별 상호작용 그룹핑
    MOCK_INTERACTIONS.forEach(interaction => {
      if (!userInteractionMap.has(interaction.userId)) {
        userInteractionMap.set(interaction.userId, new Map());
      }

      const userMap = userInteractionMap.get(interaction.userId)!;
      const carId = interaction.target.carId;

      if (!userMap.has(carId)) {
        userMap.set(carId, []);
      }

      userMap.get(carId)!.push(interaction);
    });

    // 2단계: CF 엔진에 학습 데이터 주입
    for (const [userId, carInteractions] of userInteractionMap) {
      for (const [carId, interactions] of carInteractions) {
        this.cfEngine.updateUserCarInteraction(userId, carId, interactions);
      }
    }

    // 3단계: 차량 특성 벡터 설정
    MOCK_CARS.forEach(car => {
      if (car.similarity_vector) {
        this.cfEngine.setCarFeatures(car.id, car.similarity_vector);
      }
    });

    this.isInitialized = true;
    console.log('✅ Mock 추천 엔진 초기화 완료!');
    console.log(`📊 학습된 데이터: ${MOCK_USERS.length}명 사용자, ${MOCK_CARS.length}개 차량, ${MOCK_INTERACTIONS.length}개 상호작용`);
  }

  // 🎯 메인 추천 함수: 여러 알고리즘을 조합한 하이브리드 추천
  async getRecommendations(request: RecommendationRequest): Promise<RecommendationResponse> {
    await this.initialize();

    const startTime = Date.now();

    try {
      let recommendations: RecommendationResult[] = [];

      // 추천 타입별 다른 전략 사용
      switch (request.context.type) {
        case 'homepage':
          recommendations = await this.getHomepageRecommendations(request);
          break;
        case 'similar':
          recommendations = await this.getSimilarVehicleRecommendations(request);
          break;
        case 'personalized':
          recommendations = await this.getPersonalizedRecommendations(request);
          break;
        case 'search':
          recommendations = await this.getSearchBasedRecommendations(request);
          break;
        default:
          recommendations = await this.getPersonalizedRecommendations(request);
      }

      // 제외할 차량 필터링
      if (request.excludeVehicleIds?.length) {
        recommendations = recommendations.filter(
          rec => !request.excludeVehicleIds!.includes(rec.vehicleId)
        );
      }

      // 요청된 개수만큼 제한
      recommendations = recommendations.slice(0, request.context.limit);

      const processingTime = Date.now() - startTime;

      return {
        recommendations,
        metadata: {
          totalCount: recommendations.length,
          processingTime,
          modelUsed: 'mock_hybrid_cf_v1.0',
          experimentId: this.generateExperimentId(request.userId)
        },
        debug: {
          userSegment: this.getUserSegment(request.userProfile),
          appliedFilters: request.context.searchFilters || {},
          candidateCount: MOCK_CARS.length
        }
      };

    } catch (error) {
      console.error('추천 엔진 오류:', error);
      return this.getFallbackRecommendations(request);
    }
  }

  // 📚 학습 포인트: 홈페이지 추천 = 인기도 + 개인화 혼합
  private async getHomepageRecommendations(request: RecommendationRequest): Promise<RecommendationResult[]> {
    const personalizedRecs = await this.getPersonalizedRecommendations(request);
    const popularRecs = await this.getPopularRecommendations(request.context.limit);

    // 70% 개인화 + 30% 인기 차량 혼합
    const combined = [
      ...personalizedRecs.slice(0, Math.floor(request.context.limit * 0.7)),
      ...popularRecs.slice(0, Math.floor(request.context.limit * 0.3))
    ];

    return this.rerank(combined, request);
  }

  // 📚 학습 포인트: 개인화 추천 = User-Based CF + Content-Based 조합
  private async getPersonalizedRecommendations(request: RecommendationRequest): Promise<RecommendationResult[]> {
    const userId = request.userId;
    const recommendations: RecommendationResult[] = [];

    // 1. User-Based Collaborative Filtering
    const similarUsers = await this.cfEngine.findSimilarUsers(userId, 10);
    console.log(`🔍 발견된 유사 사용자: ${similarUsers.length}명`);

    for (const similarUser of similarUsers.slice(0, 5)) {
      const userStats = this.cfEngine.getUserStats(similarUser.userId);

      // 유사 사용자가 좋아한 차량들 찾기
      const similarUserCars = this.getUserFavoriteVehicles(similarUser.userId);

      for (const carId of similarUserCars.slice(0, 3)) {
        const car = MOCK_CARS.find(c => c.id === carId);
        if (!car) continue;

        const cfScore = similarUser.similarity * 0.8; // 유사도에 따른 가중치

        recommendations.push({
          vehicleId: carId,
          score: cfScore,
          rank: recommendations.length + 1,
          scores: {
            collaborative: cfScore,
            contentBased: 0,
            popularity: this.getPopularityScore(carId),
            recency: this.getRecencyScore(car)
          },
          reasons: [{
            type: 'similar_users',
            description: `${Math.round(similarUser.similarity * 100)}% 유사한 취향의 사용자들이 선택한 차량입니다`,
            confidence: similarUser.similarity,
            evidence: {
              similarUserIds: [similarUser.userId]
            }
          }],
          modelVersion: 'user_cf_v1.0',
          timestamp: new Date(),
          confidence: similarUser.similarity
        });
      }
    }

    // 2. Content-Based Filtering (사용자 선호도 기반)
    const contentRecommendations = await this.getContentBasedRecommendations(request);
    recommendations.push(...contentRecommendations.slice(0, 5));

    // 3. 점수 순으로 정렬 및 중복 제거
    const uniqueRecs = new Map<string, RecommendationResult>();

    recommendations.forEach(rec => {
      const existing = uniqueRecs.get(rec.vehicleId);
      if (!existing || rec.score > existing.score) {
        uniqueRecs.set(rec.vehicleId, rec);
      }
    });

    return Array.from(uniqueRecs.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, request.context.limit);
  }

  // 📚 학습 포인트: 유사 차량 추천 = Item-Based CF
  private async getSimilarVehicleRecommendations(request: RecommendationRequest): Promise<RecommendationResult[]> {
    const targetCarId = request.context.currentVehicleId;
    if (!targetCarId) return [];

    const similarCars = await this.cfEngine.findSimilarCars(targetCarId, request.context.limit);
    console.log(`🚗 "${targetCarId}"와 유사한 차량: ${similarCars.length}개 발견`);

    return similarCars.map((similarCar, index) => ({
      vehicleId: similarCar.carId,
      score: similarCar.similarity,
      rank: index + 1,
      scores: {
        collaborative: 0,
        contentBased: similarCar.similarity,
        popularity: this.getPopularityScore(similarCar.carId),
        recency: this.getRecencyScore(MOCK_CARS.find(c => c.id === similarCar.carId)!)
      },
      reasons: [{
        type: 'similar_vehicles',
        description: `선택하신 차량과 ${Math.round(similarCar.similarity * 100)}% 유사한 특성을 가진 차량입니다`,
        confidence: similarCar.similarity,
        evidence: {
          similarVehicleIds: [targetCarId]
        }
      }],
      modelVersion: 'item_cf_v1.0',
      timestamp: new Date(),
      confidence: similarCar.similarity
    }));
  }

  // 콘텐츠 기반 추천 (차량 특성 매칭)
  private async getContentBasedRecommendations(request: RecommendationRequest): Promise<RecommendationResult[]> {
    const userProfile = request.userProfile;
    const recommendations: RecommendationResult[] = [];

    // 사용자 예산 범위 내 차량 필터링
    const budgetRange = userProfile.budgetRange;
    let candidateCars = MOCK_CARS;

    if (budgetRange) {
      candidateCars = MOCK_CARS.filter(car =>
        car.price >= budgetRange.min && car.price <= budgetRange.max
      );
    }

    // 사용자 선호도와 차량 특성 매칭
    for (const car of candidateCars.slice(0, 20)) {
      const matchScore = this.calculateContentMatchScore(userProfile, car);

      if (matchScore > 0.3) { // 최소 임계값
        recommendations.push({
          vehicleId: car.id,
          score: matchScore,
          rank: recommendations.length + 1,
          scores: {
            collaborative: 0,
            contentBased: matchScore,
            popularity: this.getPopularityScore(car.id),
            recency: this.getRecencyScore(car)
          },
          reasons: this.generateContentBasedReasons(userProfile, car, matchScore),
          modelVersion: 'content_based_v1.0',
          timestamp: new Date(),
          confidence: matchScore
        });
      }
    }

    return recommendations.sort((a, b) => b.score - a.score);
  }

  // 📚 학습 포인트: 콘텐츠 매칭 점수 계산
  private calculateContentMatchScore(userProfile: UserProfile, car: any): number {
    let score = 0;
    let factors = 0;

    // 예산 매칭 (가장 중요)
    if (userProfile.budgetRange) {
      const { min, max } = userProfile.budgetRange;
      if (car.price >= min && car.price <= max) {
        score += 0.4; // 40% 가중치
      } else {
        score += Math.max(0, 0.4 - Math.abs(car.price - (min + max) / 2) / (max - min));
      }
      factors += 0.4;
    }

    // 용도 매칭
    if (userProfile.purpose && car.semantic_features?.usage_pattern) {
      const purposeMatch = this.matchPurposeToUsage(userProfile.purpose, car.semantic_features.usage_pattern);
      score += purposeMatch * 0.3;
      factors += 0.3;
    }

    // 선호 브랜드 매칭
    if (userProfile.preferences?.includes(car.brand)) {
      score += 0.2;
      factors += 0.2;
    }

    // 연령대별 선호도 매칭
    if (userProfile.age && car.semantic_features?.target_demographic) {
      const ageMatch = this.matchAgeToTarget(userProfile.age, car.semantic_features.target_demographic);
      score += ageMatch * 0.1;
      factors += 0.1;
    }

    return factors > 0 ? score / factors : 0;
  }

  // 인기도 기반 추천
  private async getPopularRecommendations(limit: number): Promise<RecommendationResult[]> {
    // 상호작용 빈도로 인기도 계산
    const popularityMap = new Map<string, number>();

    MOCK_INTERACTIONS.forEach(interaction => {
      const carId = interaction.target.carId;
      const current = popularityMap.get(carId) || 0;
      const weight = this.getActionWeight(interaction.action);
      popularityMap.set(carId, current + weight);
    });

    // 인기도 순으로 정렬
    const sortedByPopularity = Array.from(popularityMap.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit);

    return sortedByPopularity.map(([carId, popularity], index) => ({
      vehicleId: carId,
      score: this.normalizePopularity(popularity),
      rank: index + 1,
      scores: {
        collaborative: 0,
        contentBased: 0,
        popularity: this.normalizePopularity(popularity),
        recency: this.getRecencyScore(MOCK_CARS.find(c => c.id === carId)!)
      },
      reasons: [{
        type: 'popular',
        description: `많은 사용자들이 관심을 보인 인기 차량입니다`,
        confidence: Math.min(1, popularity / 100)
      }],
      modelVersion: 'popularity_v1.0',
      timestamp: new Date(),
      confidence: Math.min(1, popularity / 100)
    }));
  }

  // 헬퍼 함수들
  private getUserFavoriteVehicles(userId: string): string[] {
    return MOCK_INTERACTIONS
      .filter(i => i.userId === userId && ['like', 'save', 'detail_view'].includes(i.action))
      .map(i => i.target.carId)
      .slice(0, 10);
  }

  private getPopularityScore(carId: string): number {
    const interactions = MOCK_INTERACTIONS.filter(i => i.target.carId === carId);
    return Math.min(1, interactions.length / 20); // 20개 상호작용을 만점으로
  }

  private getRecencyScore(car: any): number {
    const daysSinceListed = (Date.now() - car.lastUpdated) / (1000 * 60 * 60 * 24);
    return Math.max(0, 1 - daysSinceListed / 30); // 30일을 기준으로
  }

  private getActionWeight(action: string): number {
    const weights: Record<string, number> = {
      'click': 1,
      'long_hover': 2,
      'like': 5,
      'save': 8,
      'detail_view': 3,
      'compare_add': 4,
      'skip': -1
    };
    return weights[action] || 1;
  }

  private normalizePopularity(popularity: number): number {
    return Math.min(1, popularity / 100);
  }

  private matchPurposeToUsage(purpose: string, usage: string): number {
    const matches: Record<string, string[]> = {
      'commute': ['city_driving', 'business'],
      'family': ['family_transport'],
      'leisure': ['recreation', 'highway_cruising'],
      'business': ['business', 'city_driving']
    };

    return matches[purpose]?.includes(usage) ? 1 : 0.3;
  }

  private matchAgeToTarget(age: number, target: string): number {
    if (age < 30 && target === 'young_professional') return 1;
    if (age >= 30 && age < 50 && target === 'family') return 1;
    if (age >= 50 && target === 'senior') return 1;
    return 0.5;
  }

  private generateContentBasedReasons(userProfile: UserProfile, car: any, score: number): RecommendationReason[] {
    const reasons: RecommendationReason[] = [];

    if (userProfile.budgetRange && car.price >= userProfile.budgetRange.min && car.price <= userProfile.budgetRange.max) {
      reasons.push({
        type: 'price_match',
        description: `설정하신 예산 범위(${userProfile.budgetRange.min/10000}-${userProfile.budgetRange.max/10000}억원)에 적합합니다`,
        confidence: 0.9
      });
    }

    if (userProfile.purpose) {
      reasons.push({
        type: 'user_preference',
        description: `${userProfile.purpose} 용도에 적합한 차량입니다`,
        confidence: 0.8
      });
    }

    return reasons;
  }

  private getUserSegment(userProfile: UserProfile): string {
    if (!userProfile.budgetRange) return 'unknown';

    const avgBudget = (userProfile.budgetRange.min + userProfile.budgetRange.max) / 2;

    if (avgBudget < 3000) return 'budget';
    if (avgBudget < 6000) return 'mid-range';
    return 'premium';
  }

  private generateExperimentId(userId: string): string {
    return `exp_${Date.now()}_${userId.slice(-4)}`;
  }

  // 폴백 추천 (오류 시)
  private getFallbackRecommendations(request: RecommendationRequest): RecommendationResponse {
    const fallbackCars = MOCK_CARS.slice(0, request.context.limit);

    return {
      recommendations: fallbackCars.map((car, index) => ({
        vehicleId: car.id,
        score: 0.5,
        rank: index + 1,
        scores: {
          collaborative: 0,
          contentBased: 0,
          popularity: 0.5,
          recency: 0.5
        },
        reasons: [{
          type: 'popular',
          description: '추천 엔진 오류로 인한 기본 추천입니다',
          confidence: 0.3
        }],
        modelVersion: 'fallback_v1.0',
        timestamp: new Date(),
        confidence: 0.3
      })),
      metadata: {
        totalCount: fallbackCars.length,
        processingTime: 0,
        modelUsed: 'fallback'
      }
    };
  }

  // 재순위 조정 (다양성 보장)
  private rerank(recommendations: RecommendationResult[], request: RecommendationRequest): RecommendationResult[] {
    // 브랜드 다양성 보장
    const seenBrands = new Set<string>();
    const diversified: RecommendationResult[] = [];

    for (const rec of recommendations) {
      const car = MOCK_CARS.find(c => c.id === rec.vehicleId);
      if (!car) continue;

      if (!seenBrands.has(car.brand) || diversified.length < 3) {
        diversified.push(rec);
        seenBrands.add(car.brand);
      }
    }

    // 남은 자리는 점수 순으로 채움
    const remaining = recommendations.filter(rec =>
      !diversified.some(d => d.vehicleId === rec.vehicleId)
    );

    return [...diversified, ...remaining].slice(0, request.context.limit);
  }

  // 검색 기반 추천 (추후 구현)
  private async getSearchBasedRecommendations(request: RecommendationRequest): Promise<RecommendationResult[]> {
    // 일단 개인화 추천으로 대체
    return this.getPersonalizedRecommendations(request);
  }
}

// 싱글톤 인스턴스
export const mockRecommendationEngine = new MockRecommendationEngine();

// 편의 함수들
export const getPersonalizedRecommendations = async (
  userId: string,
  userProfile: UserProfile,
  limit: number = 10
): Promise<RecommendationResponse> => {
  return mockRecommendationEngine.getRecommendations({
    userId,
    userProfile,
    context: {
      type: 'personalized',
      limit
    }
  });
};

export const getSimilarVehicleRecommendations = async (
  userId: string,
  userProfile: UserProfile,
  currentVehicleId: string,
  limit: number = 10
): Promise<RecommendationResponse> => {
  return mockRecommendationEngine.getRecommendations({
    userId,
    userProfile,
    context: {
      type: 'similar',
      currentVehicleId,
      limit
    }
  });
};