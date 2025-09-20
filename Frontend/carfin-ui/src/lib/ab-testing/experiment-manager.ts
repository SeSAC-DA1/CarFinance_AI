'use client';

// A/B 테스트 실험 관리자
import { RecommendationExperiment } from '@/types/recommendation';

export interface ExperimentConfig {
  experimentId: string;
  name: string;
  description: string;
  variants: ExperimentVariant[];
  trafficSplit: number[]; // 각 변형에 할당할 트래픽 비율
  startDate: Date;
  endDate?: Date;
  targetMetrics: string[];
  enabled: boolean;
}

export interface ExperimentVariant {
  id: string;
  name: string;
  description: string;
  config: {
    algorithm: 'ncf_only' | 'wide_deep' | 'multi_agent' | 'collaborative_only' | 'content_only';
    parameters?: Record<string, any>;
  };
}

export interface ExperimentResult {
  experimentId: string;
  variant: string;
  userId: string;
  metrics: {
    impressions: number;
    clicks: number;
    likes: number;
    inquiries: number;
    conversions: number;
    ctr: number; // Click Through Rate
    conversionRate: number;
    engagementScore: number;
    timeSpent: number;
  };
  timestamp: Date;
}

export class ExperimentManager {
  private experiments = new Map<string, ExperimentConfig>();
  private userAssignments = new Map<string, Map<string, string>>(); // userId -> experimentId -> variant
  private results = new Map<string, ExperimentResult[]>(); // experimentId -> results

  constructor() {
    this.initializeDefaultExperiments();
  }

  // 📚 학습 포인트: 기본 A/B 테스트 실험 설정
  private initializeDefaultExperiments() {
    // 실험 1: 추천 알고리즘 비교
    const algorithmComparisonExperiment: ExperimentConfig = {
      experimentId: 'algorithm_comparison_v1',
      name: '추천 알고리즘 성능 비교',
      description: 'Neural CF vs Wide&Deep vs Multi-Agent 알고리즘 성능 비교',
      variants: [
        {
          id: 'control',
          name: '기존 방식 (Control)',
          description: '기본 인기도 기반 추천',
          config: {
            algorithm: 'collaborative_only'
          }
        },
        {
          id: 'ncf',
          name: 'Neural Collaborative Filtering',
          description: 'NCF 기반 개인화 추천',
          config: {
            algorithm: 'ncf_only',
            parameters: {
              embedding_dim: 64,
              learning_rate: 0.001
            }
          }
        },
        {
          id: 'wide_deep',
          name: 'Wide & Deep Learning',
          description: 'Wide & Deep 하이브리드 모델',
          config: {
            algorithm: 'wide_deep',
            parameters: {
              wide_columns: ['brand', 'price_range', 'category'],
              deep_columns: ['user_preferences', 'vehicle_features']
            }
          }
        },
        {
          id: 'multi_agent',
          name: 'Multi-Agent AI',
          description: 'Gemini 멀티 에이전트 협업 추천',
          config: {
            algorithm: 'multi_agent',
            parameters: {
              agents: ['data_collector', 'vehicle_expert', 'finance_expert'],
              ensemble_weights: [0.3, 0.4, 0.3]
            }
          }
        }
      ],
      trafficSplit: [0.25, 0.25, 0.25, 0.25], // 균등 분할
      startDate: new Date(),
      endDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30일 후
      targetMetrics: ['ctr', 'conversion_rate', 'engagement_score'],
      enabled: true
    };

    // 실험 2: UI/UX 변형 테스트
    const uiVariationExperiment: ExperimentConfig = {
      experimentId: 'ui_variation_v1',
      name: '추천 결과 UI 변형 테스트',
      description: '추천 이유 표시 방식과 카드 레이아웃 비교',
      variants: [
        {
          id: 'minimal',
          name: '최소한 정보',
          description: '기본 차량 정보만 표시',
          config: {
            algorithm: 'multi_agent',
            parameters: {
              show_recommendation_reason: false,
              show_confidence_score: false,
              card_layout: 'compact'
            }
          }
        },
        {
          id: 'detailed',
          name: '상세 정보',
          description: '추천 이유와 신뢰도 점수 표시',
          config: {
            algorithm: 'multi_agent',
            parameters: {
              show_recommendation_reason: true,
              show_confidence_score: true,
              card_layout: 'detailed'
            }
          }
        }
      ],
      trafficSplit: [0.5, 0.5],
      startDate: new Date(),
      endDate: new Date(Date.now() + 14 * 24 * 60 * 60 * 1000), // 14일 후
      targetMetrics: ['engagement_score', 'time_spent', 'ctr'],
      enabled: true
    };

    this.experiments.set(algorithmComparisonExperiment.experimentId, algorithmComparisonExperiment);
    this.experiments.set(uiVariationExperiment.experimentId, uiVariationExperiment);
  }

  // 사용자를 실험 변형에 할당
  assignUserToVariant(userId: string, experimentId: string): string {
    const experiment = this.experiments.get(experimentId);
    if (!experiment || !experiment.enabled) {
      return experiment?.variants[0]?.id || 'control';
    }

    // 이미 할당된 사용자인지 확인
    if (this.userAssignments.has(userId)) {
      const userExperiments = this.userAssignments.get(userId)!;
      if (userExperiments.has(experimentId)) {
        return userExperiments.get(experimentId)!;
      }
    }

    // 새로운 할당: 해시 기반 일관된 할당
    const hash = this.hashUserId(userId, experimentId);
    const variantIndex = this.selectVariantByTrafficSplit(hash, experiment.trafficSplit);
    const selectedVariant = experiment.variants[variantIndex].id;

    // 할당 결과 저장
    if (!this.userAssignments.has(userId)) {
      this.userAssignments.set(userId, new Map());
    }
    this.userAssignments.get(userId)!.set(experimentId, selectedVariant);

    console.log('🧪 A/B 테스트 할당:', {
      userId,
      experimentId,
      variant: selectedVariant,
      hash: hash.toFixed(3)
    });

    return selectedVariant;
  }

  // 📚 학습 포인트: 해시 기반 일관된 사용자 할당
  private hashUserId(userId: string, experimentId: string): number {
    const input = `${userId}_${experimentId}`;
    let hash = 0;

    for (let i = 0; i < input.length; i++) {
      const char = input.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // 32비트 정수로 변환
    }

    // 0-1 사이의 값으로 정규화
    return Math.abs(hash) / 2147483647;
  }

  // 트래픽 분할에 따른 변형 선택
  private selectVariantByTrafficSplit(hash: number, trafficSplit: number[]): number {
    let cumulative = 0;

    for (let i = 0; i < trafficSplit.length; i++) {
      cumulative += trafficSplit[i];
      if (hash <= cumulative) {
        return i;
      }
    }

    return trafficSplit.length - 1; // 마지막 변형
  }

  // 사용자의 현재 실험 설정 가져오기
  getUserExperimentConfig(userId: string): Record<string, any> {
    const config: Record<string, any> = {};

    for (const [experimentId, experiment] of this.experiments) {
      if (!experiment.enabled) continue;

      const variantId = this.assignUserToVariant(userId, experimentId);
      const variant = experiment.variants.find(v => v.id === variantId);

      if (variant) {
        config[experimentId] = {
          variant: variantId,
          config: variant.config
        };
      }
    }

    return config;
  }

  // 📊 실험 결과 기록
  recordMetric(
    experimentId: string,
    userId: string,
    metricType: 'impression' | 'click' | 'like' | 'inquiry' | 'conversion',
    value: number = 1,
    metadata?: Record<string, any>
  ) {
    const variant = this.getUserAssignedVariant(userId, experimentId);
    if (!variant) return;

    if (!this.results.has(experimentId)) {
      this.results.set(experimentId, []);
    }

    const results = this.results.get(experimentId)!;

    // 기존 결과 찾기 또는 새로 생성
    let userResult = results.find(r => r.userId === userId && r.variant === variant);

    if (!userResult) {
      userResult = {
        experimentId,
        variant,
        userId,
        metrics: {
          impressions: 0,
          clicks: 0,
          likes: 0,
          inquiries: 0,
          conversions: 0,
          ctr: 0,
          conversionRate: 0,
          engagementScore: 0,
          timeSpent: 0
        },
        timestamp: new Date()
      };
      results.push(userResult);
    }

    // 메트릭 업데이트
    switch (metricType) {
      case 'impression':
        userResult.metrics.impressions += value;
        break;
      case 'click':
        userResult.metrics.clicks += value;
        break;
      case 'like':
        userResult.metrics.likes += value;
        break;
      case 'inquiry':
        userResult.metrics.inquiries += value;
        break;
      case 'conversion':
        userResult.metrics.conversions += value;
        break;
    }

    // 계산된 메트릭 업데이트
    userResult.metrics.ctr = userResult.metrics.impressions > 0
      ? userResult.metrics.clicks / userResult.metrics.impressions
      : 0;

    userResult.metrics.conversionRate = userResult.metrics.clicks > 0
      ? userResult.metrics.conversions / userResult.metrics.clicks
      : 0;

    userResult.metrics.engagementScore = (
      userResult.metrics.clicks * 1 +
      userResult.metrics.likes * 2 +
      userResult.metrics.inquiries * 5 +
      userResult.metrics.conversions * 10
    ) / Math.max(1, userResult.metrics.impressions);

    console.log('📊 실험 메트릭 기록:', {
      experimentId,
      variant,
      metricType,
      value,
      newMetrics: userResult.metrics
    });
  }

  // 사용자의 할당된 변형 가져오기
  private getUserAssignedVariant(userId: string, experimentId: string): string | null {
    return this.userAssignments.get(userId)?.get(experimentId) || null;
  }

  // 📈 실험 결과 분석
  getExperimentAnalysis(experimentId: string): {
    variants: Array<{
      variant: string;
      userCount: number;
      avgMetrics: Record<string, number>;
    }>;
    summary: {
      totalUsers: number;
      bestVariant: string;
      significantDifference: boolean;
    };
  } {
    const results = this.results.get(experimentId) || [];
    const variantGroups = new Map<string, ExperimentResult[]>();

    // 변형별 결과 그룹핑
    results.forEach(result => {
      if (!variantGroups.has(result.variant)) {
        variantGroups.set(result.variant, []);
      }
      variantGroups.get(result.variant)!.push(result);
    });

    // 변형별 평균 메트릭 계산
    const variantAnalysis = Array.from(variantGroups.entries()).map(([variant, results]) => {
      const avgMetrics: Record<string, number> = {};
      const metricKeys = Object.keys(results[0]?.metrics || {});

      metricKeys.forEach(key => {
        const values = results.map(r => (r.metrics as any)[key] || 0);
        avgMetrics[key] = values.reduce((sum, val) => sum + val, 0) / values.length;
      });

      return {
        variant,
        userCount: results.length,
        avgMetrics
      };
    });

    // 최고 성능 변형 찾기 (CTR 기준)
    const bestVariant = variantAnalysis.reduce((best, current) =>
      (current.avgMetrics.ctr || 0) > (best.avgMetrics.ctr || 0) ? current : best
    );

    return {
      variants: variantAnalysis,
      summary: {
        totalUsers: results.length,
        bestVariant: bestVariant.variant,
        significantDifference: this.checkStatisticalSignificance(variantAnalysis)
      }
    };
  }

  // 간단한 통계적 유의성 검사
  private checkStatisticalSignificance(variants: any[]): boolean {
    if (variants.length < 2) return false;

    // 최고 CTR과 두 번째 CTR 비교
    const sorted = variants.sort((a, b) => (b.avgMetrics.ctr || 0) - (a.avgMetrics.ctr || 0));
    const best = sorted[0];
    const second = sorted[1];

    // 간단한 효과 크기 비교 (실제로는 t-test 등 통계 검정 필요)
    const improvement = (best.avgMetrics.ctr - second.avgMetrics.ctr) / second.avgMetrics.ctr;
    return improvement > 0.1 && best.userCount > 30; // 10% 이상 개선, 30명 이상 샘플
  }

  // 모든 활성 실험 목록
  getActiveExperiments(): ExperimentConfig[] {
    return Array.from(this.experiments.values()).filter(exp => exp.enabled);
  }

  // 실험 활성화/비활성화
  toggleExperiment(experimentId: string, enabled: boolean) {
    const experiment = this.experiments.get(experimentId);
    if (experiment) {
      experiment.enabled = enabled;
    }
  }

  // 📤 결과를 외부 분석 도구로 내보내기 (추후 구현)
  exportResults(experimentId: string): string {
    const results = this.results.get(experimentId) || [];
    return JSON.stringify({
      experimentId,
      exportDate: new Date().toISOString(),
      results,
      analysis: this.getExperimentAnalysis(experimentId)
    }, null, 2);
  }
}

// 싱글톤 인스턴스
export const experimentManager = new ExperimentManager();

// 편의 함수들
export const assignUserToExperiment = (userId: string, experimentId: string) =>
  experimentManager.assignUserToVariant(userId, experimentId);

export const recordExperimentMetric = (
  experimentId: string,
  userId: string,
  metricType: 'impression' | 'click' | 'like' | 'inquiry' | 'conversion',
  value?: number
) => experimentManager.recordMetric(experimentId, userId, metricType, value);

export const getUserExperimentConfig = (userId: string) =>
  experimentManager.getUserExperimentConfig(userId);