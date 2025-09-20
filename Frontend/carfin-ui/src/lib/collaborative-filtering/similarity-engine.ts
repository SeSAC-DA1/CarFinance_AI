// 협업 필터링 유사도 엔진
import { ProcessedCarData, UserProfile, UserInteraction } from '@/types';

export interface SimilarityScore {
  userId: string;
  similarity: number;
  sharedPreferences: string[];
}

export interface CarSimilarity {
  carId: string;
  similarity: number;
  commonFeatures: string[];
}

export class CollaborativeFilteringEngine {
  private userCarMatrix: Map<string, Map<string, number>> = new Map(); // userId -> carId -> rating
  private carFeatureMatrix: Map<string, number[]> = new Map(); // carId -> feature vector
  private userSimilarityCache: Map<string, SimilarityScore[]> = new Map();
  private carSimilarityCache: Map<string, CarSimilarity[]> = new Map();

  // 사용자-차량 상호작용을 평점으로 변환
  updateUserCarInteraction(userId: string, carId: string, interactions: UserInteraction[]): void {
    const rating = this.calculateImplicitRating(interactions);

    if (!this.userCarMatrix.has(userId)) {
      this.userCarMatrix.set(userId, new Map());
    }

    const userRatings = this.userCarMatrix.get(userId)!;
    userRatings.set(carId, rating);

    // 캐시 무효화
    this.userSimilarityCache.delete(userId);
    this.invalidateCarSimilarityCache(carId);
  }

  // 암시적 피드백을 평점으로 변환
  private calculateImplicitRating(interactions: UserInteraction[]): number {
    let score = 0;

    for (const interaction of interactions) {
      switch (interaction.action) {
        case 'click':
          score += 0.5;
          break;
        case 'long_hover':
          score += 0.3 * (interaction.context?.duration || 1000) / 1000; // 초당 0.3점
          break;
        case 'like':
          score += 2.0;
          break;
        case 'save':
          score += 3.0;
          break;
        case 'share':
          score += 2.5;
          break;
        case 'detail_view':
          score += 1.0;
          break;
        case 'compare_add':
          score += 1.5;
          break;
        case 'skip':
          score -= 0.5;
          break;
        case 'quick_dismiss':
          score -= 1.0;
          break;
      }
    }

    // 0-5 스케일로 정규화
    return Math.max(0, Math.min(5, score));
  }

  // User-Based 협업 필터링
  async findSimilarUsers(targetUserId: string, limit: number = 20): Promise<SimilarityScore[]> {
    if (this.userSimilarityCache.has(targetUserId)) {
      return this.userSimilarityCache.get(targetUserId)!.slice(0, limit);
    }

    const targetUserRatings = this.userCarMatrix.get(targetUserId);
    if (!targetUserRatings) {
      return [];
    }

    const similarities: SimilarityScore[] = [];

    for (const [userId, userRatings] of this.userCarMatrix) {
      if (userId === targetUserId) continue;

      const similarity = this.calculateUserSimilarity(targetUserRatings, userRatings);

      if (similarity > 0.1) { // 최소 임계값
        similarities.push({
          userId,
          similarity,
          sharedPreferences: this.findSharedPreferences(targetUserRatings, userRatings)
        });
      }
    }

    // 유사도 순으로 정렬
    similarities.sort((a, b) => b.similarity - a.similarity);

    this.userSimilarityCache.set(targetUserId, similarities);
    return similarities.slice(0, limit);
  }

  // Item-Based 협업 필터링
  async findSimilarCars(targetCarId: string, limit: number = 20): Promise<CarSimilarity[]> {
    if (this.carSimilarityCache.has(targetCarId)) {
      return this.carSimilarityCache.get(targetCarId)!.slice(0, limit);
    }

    const similarities: CarSimilarity[] = [];
    const targetCarRatings = this.getCarRatings(targetCarId);

    for (const [carId] of this.carFeatureMatrix) {
      if (carId === targetCarId) continue;

      const carRatings = this.getCarRatings(carId);
      const similarity = this.calculateCarSimilarity(targetCarRatings, carRatings);

      if (similarity > 0.1) {
        similarities.push({
          carId,
          similarity,
          commonFeatures: this.findCommonFeatures(targetCarId, carId)
        });
      }
    }

    similarities.sort((a, b) => b.similarity - a.similarity);

    this.carSimilarityCache.set(targetCarId, similarities);
    return similarities.slice(0, limit);
  }

  // 사용자 유사도 계산 (Pearson 상관계수)
  private calculateUserSimilarity(
    userA: Map<string, number>,
    userB: Map<string, number>
  ): number {
    const commonItems: string[] = [];

    // 공통으로 평가한 차량 찾기
    for (const carId of userA.keys()) {
      if (userB.has(carId)) {
        commonItems.push(carId);
      }
    }

    if (commonItems.length < 2) return 0;

    // Pearson 상관계수 계산
    let sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, pSum = 0;

    for (const carId of commonItems) {
      const rating1 = userA.get(carId)!;
      const rating2 = userB.get(carId)!;

      sum1 += rating1;
      sum2 += rating2;
      sum1Sq += rating1 * rating1;
      sum2Sq += rating2 * rating2;
      pSum += rating1 * rating2;
    }

    const num = pSum - (sum1 * sum2 / commonItems.length);
    const den = Math.sqrt(
      (sum1Sq - sum1 * sum1 / commonItems.length) *
      (sum2Sq - sum2 * sum2 / commonItems.length)
    );

    return den === 0 ? 0 : num / den;
  }

  // 차량 유사도 계산 (코사인 유사도)
  private calculateCarSimilarity(
    ratingsA: Map<string, number>,
    ratingsB: Map<string, number>
  ): number {
    const commonUsers: string[] = [];

    for (const userId of ratingsA.keys()) {
      if (ratingsB.has(userId)) {
        commonUsers.push(userId);
      }
    }

    if (commonUsers.length < 2) return 0;

    // 코사인 유사도 계산
    let dotProduct = 0, normA = 0, normB = 0;

    for (const userId of commonUsers) {
      const ratingA = ratingsA.get(userId)!;
      const ratingB = ratingsB.get(userId)!;

      dotProduct += ratingA * ratingB;
      normA += ratingA * ratingA;
      normB += ratingB * ratingB;
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dotProduct / denominator;
  }

  // 차량에 대한 모든 사용자 평점 가져오기
  private getCarRatings(carId: string): Map<string, number> {
    const ratings = new Map<string, number>();

    for (const [userId, userRatings] of this.userCarMatrix) {
      if (userRatings.has(carId)) {
        ratings.set(userId, userRatings.get(carId)!);
      }
    }

    return ratings;
  }

  // 공통 선호사항 찾기
  private findSharedPreferences(
    userA: Map<string, number>,
    userB: Map<string, number>
  ): string[] {
    const shared: string[] = [];

    for (const [carId, ratingA] of userA) {
      const ratingB = userB.get(carId);
      if (ratingB && Math.abs(ratingA - ratingB) < 1.0) {
        shared.push(carId);
      }
    }

    return shared;
  }

  // 공통 특성 찾기
  private findCommonFeatures(carA: string, carB: string): string[] {
    const featuresA = this.carFeatureMatrix.get(carA);
    const featuresB = this.carFeatureMatrix.get(carB);

    if (!featuresA || !featuresB) return [];

    const common: string[] = [];

    // 특성 벡터 비교 (간단히 구현)
    for (let i = 0; i < Math.min(featuresA.length, featuresB.length); i++) {
      if (Math.abs(featuresA[i] - featuresB[i]) < 0.1) {
        common.push(`feature_${i}`);
      }
    }

    return common;
  }

  // 캐시 무효화
  private invalidateCarSimilarityCache(carId: string): void {
    this.carSimilarityCache.delete(carId);
    // 다른 차량들의 캐시에서도 해당 차량 관련 정보 제거
    for (const [cacheCarId, similarities] of this.carSimilarityCache) {
      const filtered = similarities.filter(sim => sim.carId !== carId);
      if (filtered.length !== similarities.length) {
        this.carSimilarityCache.set(cacheCarId, filtered);
      }
    }
  }

  // 차량 특성 벡터 설정
  setCarFeatures(carId: string, features: number[]): void {
    this.carFeatureMatrix.set(carId, features);
    this.invalidateCarSimilarityCache(carId);
  }

  // 사용자 통계
  getUserStats(userId: string): {
    totalRatings: number;
    avgRating: number;
    preferredCategories: string[]
  } {
    const userRatings = this.userCarMatrix.get(userId);
    if (!userRatings) {
      return { totalRatings: 0, avgRating: 0, preferredCategories: [] };
    }

    const ratings = Array.from(userRatings.values());
    const totalRatings = ratings.length;
    const avgRating = ratings.reduce((sum, rating) => sum + rating, 0) / totalRatings;

    return {
      totalRatings,
      avgRating,
      preferredCategories: [] // TODO: 구현
    };
  }
}