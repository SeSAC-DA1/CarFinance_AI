// 벡터 데이터베이스 통합 시스템
import { ProcessedCarData, UserProfile, UserInteraction } from '@/types';

export interface VectorEmbedding {
  id: string;
  vector: number[];
  metadata: Record<string, any>;
  timestamp: number;
}

export interface CarVectorData extends VectorEmbedding {
  carId: string;
  features: {
    // 기본 차량 정보
    brand: string;
    model: string;
    year: number;
    price: number;
    mileage: number;

    // 정규화된 특성
    normalized_features: {
      age_score: number;
      price_score: number;
      mileage_score: number;
      fuel_efficiency_score: number;
      depreciation_score: number;
    };

    // 의미적 특성 (Gemini AI 분석)
    semantic_features: {
      target_demographic_vector: number[];
      usage_pattern_vector: number[];
      style_category_vector: number[];
      reliability_vector: number[];
    };

    // 시장 특성
    market_features: {
      popularity_score: number;
      price_trend_vector: number[];
      availability_score: number;
      dealer_rating_score: number;
    };
  };
  source_urls: string[];
  last_updated: number;
}

export interface UserVectorData extends VectorEmbedding {
  userId: string;
  preference_vector: number[];
  behavioral_vector: number[];
  demographic_vector: number[];
  interaction_history: {
    total_interactions: number;
    preference_confidence: number;
    last_active: number;
    session_count: number;
  };
}

export class VectorDatabaseManager {
  private carVectors: Map<string, CarVectorData> = new Map();
  private userVectors: Map<string, UserVectorData> = new Map();
  private similarityIndex: Map<string, string[]> = new Map(); // 빠른 검색을 위한 인덱스

  // 추후 Pinecone, Weaviate, Chroma 등으로 교체 가능
  private vectorDimension = 512; // 임베딩 차원

  // 차량 데이터를 벡터로 변환하여 저장
  async storeCarVector(carData: ProcessedCarData): Promise<void> {
    const carVector = await this.generateCarVector(carData);

    const vectorData: CarVectorData = {
      id: `car_${carData.id}`,
      carId: carData.id,
      vector: carVector,
      features: {
        brand: carData.brand,
        model: carData.model,
        year: carData.year,
        price: carData.price,
        mileage: carData.mileage,

        normalized_features: {
          age_score: this.normalizeAge(carData.year),
          price_score: this.normalizePrice(carData.price),
          mileage_score: this.normalizeMileage(carData.mileage),
          fuel_efficiency_score: carData.normalized_features?.fuel_efficiency || 0,
          depreciation_score: carData.normalized_features?.depreciation_rate || 0
        },

        semantic_features: {
          target_demographic_vector: await this.generateDemographicVector(carData),
          usage_pattern_vector: await this.generateUsageVector(carData),
          style_category_vector: await this.generateStyleVector(carData),
          reliability_vector: await this.generateReliabilityVector(carData)
        },

        market_features: {
          popularity_score: await this.calculatePopularityScore(carData),
          price_trend_vector: await this.generatePriceTrendVector(carData),
          availability_score: await this.calculateAvailabilityScore(carData),
          dealer_rating_score: await this.calculateDealerRating(carData)
        }
      },
      source_urls: [carData.sourceUrl || ''],
      last_updated: Date.now(),
      metadata: {
        brand: carData.brand,
        model: carData.model,
        year: carData.year,
        price_range: this.getPriceRange(carData.price),
        body_type: carData.bodyType,
        fuel_type: carData.fuelType
      },
      timestamp: Date.now()
    };

    this.carVectors.set(carData.id, vectorData);
    await this.updateSimilarityIndex(carData.id, carVector);

    // 추후 실제 벡터 DB 연동
    // await this.vectorDB.upsert(vectorData);
  }

  // 사용자 프로필을 벡터로 변환하여 저장
  async storeUserVector(userProfile: UserProfile): Promise<void> {
    const preferenceVector = await this.generateUserPreferenceVector(userProfile);
    const behavioralVector = await this.generateBehavioralVector(userProfile);
    const demographicVector = await this.generateUserDemographicVector(userProfile);

    const userVector: UserVectorData = {
      id: `user_${userProfile.id}`,
      userId: userProfile.id,
      vector: [...preferenceVector, ...behavioralVector, ...demographicVector],
      preference_vector: preferenceVector,
      behavioral_vector: behavioralVector,
      demographic_vector: demographicVector,
      interaction_history: userProfile.learning_metadata,
      metadata: {
        user_segment: this.classifyUserSegment(userProfile),
        preference_confidence: userProfile.learning_metadata.preference_confidence,
        activity_level: this.calculateActivityLevel(userProfile)
      },
      timestamp: Date.now()
    };

    this.userVectors.set(userProfile.id, userVector);

    // 추후 실제 벡터 DB 연동
    // await this.vectorDB.upsert(userVector);
  }

  // 벡터 유사도 기반 차량 검색
  async searchSimilarCars(
    queryVector: number[],
    filters?: {
      price_range?: [number, number];
      year_range?: [number, number];
      brands?: string[];
      body_types?: string[];
    },
    topK: number = 20
  ): Promise<CarVectorData[]> {

    let candidates = Array.from(this.carVectors.values());

    // 필터 적용
    if (filters) {
      candidates = candidates.filter(car => {
        if (filters.price_range &&
            (car.features.price < filters.price_range[0] || car.features.price > filters.price_range[1])) {
          return false;
        }
        if (filters.year_range &&
            (car.features.year < filters.year_range[0] || car.features.year > filters.year_range[1])) {
          return false;
        }
        if (filters.brands && !filters.brands.includes(car.features.brand)) {
          return false;
        }
        if (filters.body_types && !filters.body_types.includes(car.metadata.body_type)) {
          return false;
        }
        return true;
      });
    }

    // 코사인 유사도 계산
    const similarities = candidates.map(car => ({
      car,
      similarity: this.calculateCosineSimilarity(queryVector, car.vector)
    }));

    // 유사도 순으로 정렬하여 상위 K개 반환
    return similarities
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, topK)
      .map(item => item.car);
  }

  // 사용자 기반 개인화 추천
  async getPersonalizedRecommendations(
    userId: string,
    excludeCarIds?: string[],
    topK: number = 20
  ): Promise<CarVectorData[]> {

    const userVector = this.userVectors.get(userId);
    if (!userVector) {
      throw new Error(`User vector not found for user: ${userId}`);
    }

    // 사용자 벡터를 차량 검색 쿼리로 변환
    const searchVector = this.adaptUserVectorForCarSearch(userVector);

    // 제외할 차량 필터링
    let candidates = Array.from(this.carVectors.values());
    if (excludeCarIds) {
      candidates = candidates.filter(car => !excludeCarIds.includes(car.carId));
    }

    return this.searchSimilarCars(searchVector, undefined, topK);
  }

  // 하이브리드 추천 (협업 필터링 + 벡터 검색 + 콘텐츠 기반)
  async getHybridRecommendations(
    userId: string,
    contextFilters?: any,
    topK: number = 20
  ): Promise<{
    cars: CarVectorData[];
    scores: {
      collaborative_filtering: number;
      vector_similarity: number;
      content_based: number;
      hybrid_score: number;
    }[];
  }> {

    // 1. 벡터 기반 추천
    const vectorRecommendations = await this.getPersonalizedRecommendations(userId, [], topK * 2);

    // 2. 각 추천에 대한 다양한 점수 계산
    const scoredRecommendations = await Promise.all(
      vectorRecommendations.map(async (car) => {
        const scores = {
          collaborative_filtering: await this.getCollaborativeFilteringScore(userId, car.carId),
          vector_similarity: await this.getVectorSimilarityScore(userId, car.carId),
          content_based: await this.getContentBasedScore(userId, car.carId),
          hybrid_score: 0
        };

        // 하이브리드 점수 계산 (가중 평균)
        scores.hybrid_score =
          scores.collaborative_filtering * 0.4 +
          scores.vector_similarity * 0.35 +
          scores.content_based * 0.25;

        return { car, scores };
      })
    );

    // 하이브리드 점수로 정렬
    scoredRecommendations.sort((a, b) => b.scores.hybrid_score - a.scores.hybrid_score);

    return {
      cars: scoredRecommendations.slice(0, topK).map(item => item.car),
      scores: scoredRecommendations.slice(0, topK).map(item => item.scores)
    };
  }

  // 차량 특성 벡터 생성
  private async generateCarVector(carData: ProcessedCarData): Promise<number[]> {
    const vector = new Array(this.vectorDimension).fill(0);

    // 기본 특성 (0-100 인덱스)
    vector[0] = this.normalizeAge(carData.year);
    vector[1] = this.normalizePrice(carData.price);
    vector[2] = this.normalizeMileage(carData.mileage);
    vector[3] = this.encodeBrand(carData.brand);
    vector[4] = this.encodeBodyType(carData.bodyType);

    // 의미적 특성 (100-300 인덱스)
    if (carData.semantic_features) {
      const semanticVector = await this.generateSemanticEmbedding(carData.semantic_features);
      for (let i = 0; i < Math.min(200, semanticVector.length); i++) {
        vector[100 + i] = semanticVector[i];
      }
    }

    // 시장 특성 (300-400 인덱스)
    // TODO: 시장 데이터 기반 벡터 생성

    // 나머지는 패딩 또는 미래 확장용
    return vector;
  }

  // 사용자 선호도 벡터 생성
  private async generateUserPreferenceVector(userProfile: UserProfile): Promise<number[]> {
    const vector = new Array(256).fill(0); // 선호도는 256차원

    // 선호도 정보를 벡터로 변환
    let index = 0;
    for (const [feature, preference] of Object.entries(userProfile.preferences)) {
      if (index < 256) {
        vector[index++] = preference;
      }
    }

    return vector;
  }

  // 행동 패턴 벡터 생성
  private async generateBehavioralVector(userProfile: UserProfile): Promise<number[]> {
    const vector = new Array(128).fill(0); // 행동은 128차원

    // 상호작용 패턴 분석
    const interactions = userProfile.interaction_history;

    // 행동 패턴을 벡터로 인코딩
    vector[0] = interactions.length / 100; // 정규화된 상호작용 수
    vector[1] = userProfile.learning_metadata.session_count / 50;
    vector[2] = userProfile.learning_metadata.preference_confidence;

    // TODO: 더 복잡한 행동 패턴 분석

    return vector;
  }

  // 코사인 유사도 계산
  private calculateCosineSimilarity(vectorA: number[], vectorB: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < Math.min(vectorA.length, vectorB.length); i++) {
      dotProduct += vectorA[i] * vectorB[i];
      normA += vectorA[i] * vectorA[i];
      normB += vectorB[i] * vectorB[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dotProduct / denominator;
  }

  // 정규화 헬퍼 함수들
  private normalizeAge(year: number): number {
    const currentYear = new Date().getFullYear();
    const age = currentYear - year;
    return Math.max(0, Math.min(1, (20 - age) / 20)); // 20년 이상은 0, 신차는 1
  }

  private normalizePrice(price: number): number {
    return Math.max(0, Math.min(1, (price - 500) / (10000 - 500))); // 500만원-1억원 범위
  }

  private normalizeMileage(mileage: number): number {
    return Math.max(0, Math.min(1, (300000 - mileage) / 300000)); // 낮은 주행거리일수록 높은 점수
  }

  private encodeBrand(brand: string): number {
    // 브랜드를 숫자로 인코딩 (추후 더 정교한 임베딩으로 교체)
    const brands = ['현대', '기아', '삼성', 'BMW', '벤츠', '아우디', '토요타', '혼다'];
    const index = brands.indexOf(brand);
    return index === -1 ? 0.5 : (index + 1) / brands.length;
  }

  private encodeBodyType(bodyType: string): number {
    const types = ['sedan', 'suv', 'hatchback', 'coupe', 'wagon'];
    const index = types.indexOf(bodyType);
    return index === -1 ? 0.5 : (index + 1) / types.length;
  }

  // 추후 구현할 메서드들
  private async generateSemanticEmbedding(semanticFeatures: any): Promise<number[]> {
    // Gemini AI를 통한 의미적 임베딩 생성
    return new Array(200).fill(0.5); // 임시
  }

  private async getCollaborativeFilteringScore(userId: string, carId: string): Promise<number> {
    // 협업 필터링 점수 계산
    return 0.5; // 임시
  }

  private async getVectorSimilarityScore(userId: string, carId: string): Promise<number> {
    // 벡터 유사도 점수 계산
    return 0.5; // 임시
  }

  private async getContentBasedScore(userId: string, carId: string): Promise<number> {
    // 콘텐츠 기반 점수 계산
    return 0.5; // 임시
  }

  // 기타 헬퍼 메서드들
  private getPriceRange(price: number): string {
    if (price < 1000) return 'budget';
    if (price < 3000) return 'mid-range';
    if (price < 6000) return 'premium';
    return 'luxury';
  }

  private classifyUserSegment(userProfile: UserProfile): string {
    // 사용자 세그먼트 분류 로직
    return 'general'; // 임시
  }

  private calculateActivityLevel(userProfile: UserProfile): string {
    const interactions = userProfile.interaction_history.length;
    if (interactions > 100) return 'high';
    if (interactions > 20) return 'medium';
    return 'low';
  }

  private adaptUserVectorForCarSearch(userVector: UserVectorData): number[] {
    // 사용자 벡터를 차량 검색에 적합한 형태로 변환
    return userVector.preference_vector; // 임시
  }

  private async updateSimilarityIndex(carId: string, vector: number[]): Promise<void> {
    // 빠른 검색을 위한 인덱스 업데이트
    // TODO: 구현
  }

  // 추후 실제 벡터 DB 연동을 위한 인터페이스
  async connectToVectorDB(config: {
    provider: 'pinecone' | 'weaviate' | 'chroma' | 'qdrant';
    apiKey: string;
    endpoint: string;
    indexName: string;
  }): Promise<void> {
    // 실제 벡터 DB 연결 로직
    console.log(`Connecting to ${config.provider} vector database...`);
  }

  // 기타 생성 메서드들 (임시 구현)
  private async generateDemographicVector(carData: ProcessedCarData): Promise<number[]> {
    return new Array(50).fill(0.5);
  }

  private async generateUsageVector(carData: ProcessedCarData): Promise<number[]> {
    return new Array(30).fill(0.5);
  }

  private async generateStyleVector(carData: ProcessedCarData): Promise<number[]> {
    return new Array(40).fill(0.5);
  }

  private async generateReliabilityVector(carData: ProcessedCarData): Promise<number[]> {
    return new Array(20).fill(0.5);
  }

  private async calculatePopularityScore(carData: ProcessedCarData): Promise<number> {
    return 0.5;
  }

  private async generatePriceTrendVector(carData: ProcessedCarData): Promise<number[]> {
    return new Array(10).fill(0.5);
  }

  private async calculateAvailabilityScore(carData: ProcessedCarData): Promise<number> {
    return 0.5;
  }

  private async calculateDealerRating(carData: ProcessedCarData): Promise<number> {
    return 0.5;
  }

  private async generateUserDemographicVector(userProfile: UserProfile): Promise<number[]> {
    return new Array(128).fill(0.5);
  }
}