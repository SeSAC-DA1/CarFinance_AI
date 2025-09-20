// 추천시스템 핵심 데이터 타입 정의

export interface UserInteraction {
  id: string;
  userId: string;
  vehicleId: string;
  type: 'view' | 'like' | 'inquiry' | 'test_drive' | 'share';
  timestamp: Date;
  duration?: number; // 조회 시간 (초)
  confidence: number; // 관심도 점수 (0-1)
  context?: {
    source: 'search' | 'recommendation' | 'similar' | 'popular';
    position?: number; // 리스트에서의 위치
    searchQuery?: string;
  };
}

export interface VehicleFeatures {
  vehicleId: string;

  // 기본 속성
  price: number;
  year: number;
  mileage: number;
  brand: string;
  model: string;
  fuelType: 'gasoline' | 'hybrid' | 'electric' | 'diesel';
  transmission: 'manual' | 'automatic';

  // 카테고리
  category: 'sedan' | 'suv' | 'hatchback' | 'coupe' | 'truck' | 'van';
  size: 'compact' | 'midsize' | 'fullsize' | 'luxury';

  // 성능
  fuelEfficiency: number; // km/L
  engineSize: number; // cc
  horsepower?: number;

  // 상태
  condition: 'excellent' | 'good' | 'fair' | 'poor';
  accidentHistory: boolean;
  ownerCount: number;

  // 위치/시간
  location: {
    city: string;
    district: string;
    latitude?: number;
    longitude?: number;
  };
  listedDate: Date;

  // 계산된 특성
  depreciation: number; // 감가상각률
  marketValue: number; // 시장 가치 점수
  popularityScore: number; // 인기도 점수
}

export interface UserProfile {
  user_id: string;
  name?: string;
  email?: string;

  // 기본 정보
  age?: number;
  income?: number;
  location?: {
    city: string;
    district: string;
  };

  // 선호도
  purpose?: 'commute' | 'family' | 'leisure' | 'business';
  preferences?: string[];
  budgetRange?: {
    min: number;
    max: number;
  };

  // 행동 패턴
  searchHistory?: string[];
  viewHistory?: string[];
  favoriteVehicles?: string[];

  // 세그먼트
  segment?: 'budget' | 'value' | 'premium' | 'luxury';
  creditScore?: number;

  // 메타데이터
  createdAt: Date;
  lastActiveAt: Date;
  guest?: boolean;
}

export interface RecommendationRequest {
  userId: string;
  userProfile: UserProfile;
  context: {
    type: 'homepage' | 'search' | 'similar' | 'personalized';
    currentVehicleId?: string; // 유사 차량 추천용
    searchFilters?: VehicleSearchFilters;
    limit: number;
  };
  excludeVehicleIds?: string[];
}

export interface VehicleSearchFilters {
  priceRange?: { min: number; max: number };
  yearRange?: { min: number; max: number };
  mileageMax?: number;
  brands?: string[];
  categories?: string[];
  fuelTypes?: string[];
  location?: string;
  sortBy?: 'price' | 'year' | 'mileage' | 'popularity' | 'recommendation';
}

export interface RecommendationResult {
  vehicleId: string;
  score: number; // 추천 점수 (0-1)
  rank: number;

  // 점수 분해
  scores: {
    collaborative: number; // 협업 필터링 점수
    contentBased: number; // 콘텐츠 기반 점수
    popularity: number; // 인기도 점수
    recency: number; // 최신성 점수
  };

  // 설명 가능성
  reasons: RecommendationReason[];

  // 메타데이터
  modelVersion: string;
  timestamp: Date;
  confidence: number; // 예측 신뢰도
}

export interface RecommendationReason {
  type: 'similar_users' | 'similar_vehicles' | 'user_preference' | 'popular' | 'price_match' | 'location_nearby';
  description: string;
  confidence: number;
  evidence?: {
    similarUserIds?: string[];
    similarVehicleIds?: string[];
    matchedPreferences?: string[];
  };
}

export interface RecommendationResponse {
  recommendations: RecommendationResult[];
  metadata: {
    totalCount: number;
    processingTime: number;
    modelUsed: string;
    experimentId?: string; // A/B 테스트용
  };
  debug?: {
    userSegment: string;
    appliedFilters: any;
    candidateCount: number;
  };
}

// Gemini Agent 응답 타입
export interface AgentRecommendationInsight {
  agent: 'data_collector' | 'vehicle_expert' | 'finance_expert' | 'consultant';
  analysis: string;
  recommendations: {
    vehicleId: string;
    score: number;
    reasoning: string;
  }[];
  confidence: number;
  processingTime: number;
}

export interface MultiAgentRecommendation {
  vehicleId: string;
  finalScore: number;
  agentInsights: {
    technical?: string; // vehicle_expert
    financial?: string; // finance_expert
    behavioral?: string; // data_collector
    summary?: string; // consultant
  };
  consensusLevel: number; // 에이전트 간 합의 정도
}

// 실시간 추론용 타입
export interface RealtimeRecommendationContext {
  sessionId: string;
  currentPage: string;
  viewedVehicles: string[];
  searchQueries: string[];
  timeSpent: number;
  clickPattern: {
    vehicleId: string;
    timestamp: Date;
    action: string;
  }[];
}

// A/B 테스트용 타입
export interface RecommendationExperiment {
  experimentId: string;
  variant: 'control' | 'ncf_only' | 'wide_deep' | 'multi_agent';
  userId: string;
  startTime: Date;
  endTime?: Date;
  metrics: {
    impressions: number;
    clicks: number;
    inquiries: number;
    conversions: number;
  };
}

// 성능 모니터링용 타입
export interface RecommendationMetrics {
  timestamp: Date;
  modelVersion: string;
  userSegment: string;

  // 정확도 지표
  precision_at_5: number;
  precision_at_10: number;
  recall_at_5: number;
  recall_at_10: number;
  ndcg_at_10: number;

  // 비즈니스 지표
  click_through_rate: number;
  conversion_rate: number;
  revenue_per_recommendation: number;

  // 성능 지표
  response_time_ms: number;
  cache_hit_rate: number;
  error_rate: number;
}