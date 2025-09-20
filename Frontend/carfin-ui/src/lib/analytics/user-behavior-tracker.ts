'use client';

import { UserInteraction, RealtimeRecommendationContext } from '@/types/recommendation';

interface TrackingEvent {
  vehicleId?: string;
  action: 'view' | 'click' | 'like' | 'inquiry' | 'share' | 'filter' | 'search';
  duration?: number;
  context?: Record<string, any>;
}

export class UserBehaviorTracker {
  private sessionId: string;
  private userId: string;
  private interactions: UserInteraction[] = [];
  private sessionStartTime: number;
  private currentVehicleViewStart?: number;
  private realtimeContext: RealtimeRecommendationContext;

  constructor(userId: string) {
    this.userId = userId;
    this.sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    this.sessionStartTime = Date.now();

    this.realtimeContext = {
      sessionId: this.sessionId,
      currentPage: typeof window !== 'undefined' ? window.location.pathname : '',
      viewedVehicles: [],
      searchQueries: [],
      timeSpent: 0,
      clickPattern: []
    };

    // 페이지 이탈 시 데이터 저장
    if (typeof window !== 'undefined') {
      window.addEventListener('beforeunload', () => {
        this.flushInteractions();
      });

      // 페이지 변경 추적
      window.addEventListener('popstate', () => {
        this.updateCurrentPage();
      });
    }
  }

  // 차량 조회 추적
  trackVehicleView(vehicleId: string, context?: { source: string; position?: number }) {
    if (this.currentVehicleViewStart) {
      // 이전 차량 조회 종료
      this.endVehicleView();
    }

    this.currentVehicleViewStart = Date.now();

    // 실시간 컨텍스트 업데이트
    if (!this.realtimeContext.viewedVehicles.includes(vehicleId)) {
      this.realtimeContext.viewedVehicles.push(vehicleId);
    }

    const interaction: UserInteraction = {
      id: this.generateInteractionId(),
      userId: this.userId,
      vehicleId,
      type: 'view',
      timestamp: new Date(),
      confidence: this.calculateInitialConfidence('view'),
      context: {
        source: context?.source || 'unknown',
        position: context?.position
      }
    };

    this.interactions.push(interaction);
    this.updateClickPattern(vehicleId, 'view');
  }

  // 차량 조회 종료
  endVehicleView() {
    if (!this.currentVehicleViewStart || this.interactions.length === 0) return;

    const duration = (Date.now() - this.currentVehicleViewStart) / 1000; // 초 단위
    const lastInteraction = this.interactions[this.interactions.length - 1];

    if (lastInteraction.type === 'view') {
      lastInteraction.duration = duration;
      lastInteraction.confidence = this.calculateConfidenceWithDuration(duration);
    }

    this.currentVehicleViewStart = undefined;
  }

  // 좋아요/찜 추적
  trackLike(vehicleId: string) {
    const interaction: UserInteraction = {
      id: this.generateInteractionId(),
      userId: this.userId,
      vehicleId,
      type: 'like',
      timestamp: new Date(),
      confidence: 0.8, // 명시적 행동이므로 높은 신뢰도
      context: {
        source: 'user_action'
      }
    };

    this.interactions.push(interaction);
    this.updateClickPattern(vehicleId, 'like');
  }

  // 문의 추적
  trackInquiry(vehicleId: string, inquiryType: 'phone' | 'message' | 'test_drive') {
    const interaction: UserInteraction = {
      id: this.generateInteractionId(),
      userId: this.userId,
      vehicleId,
      type: 'inquiry',
      timestamp: new Date(),
      confidence: 0.9, // 매우 높은 관심도
      context: {
        source: 'user_action',
        inquiryType
      }
    };

    this.interactions.push(interaction);
    this.updateClickPattern(vehicleId, 'inquiry');
  }

  // 검색 추적
  trackSearch(query: string, filters?: Record<string, any>, resultCount?: number) {
    if (!this.realtimeContext.searchQueries.includes(query)) {
      this.realtimeContext.searchQueries.push(query);
    }

    // 검색 결과 클릭률 분석을 위한 데이터
    localStorage.setItem('lastSearch', JSON.stringify({
      query,
      filters,
      resultCount,
      timestamp: Date.now()
    }));
  }

  // 클릭 패턴 업데이트
  private updateClickPattern(vehicleId: string, action: string) {
    this.realtimeContext.clickPattern.push({
      vehicleId,
      timestamp: new Date(),
      action
    });

    // 최근 50개 액션만 유지
    if (this.realtimeContext.clickPattern.length > 50) {
      this.realtimeContext.clickPattern = this.realtimeContext.clickPattern.slice(-50);
    }
  }

  // 페이지 업데이트
  private updateCurrentPage() {
    if (typeof window !== 'undefined') {
      this.realtimeContext.currentPage = window.location.pathname;
      this.realtimeContext.timeSpent = Date.now() - this.sessionStartTime;
    }
  }

  // 신뢰도 계산 (조회 시간 기반)
  private calculateConfidenceWithDuration(duration: number): number {
    // 조회 시간에 따른 관심도 점수 계산
    if (duration < 5) return 0.1;       // 5초 미만: 낮은 관심
    if (duration < 15) return 0.3;      // 15초 미만: 보통 관심
    if (duration < 30) return 0.5;      // 30초 미만: 높은 관심
    if (duration < 60) return 0.7;      // 1분 미만: 매우 높은 관심
    return 0.9;                         // 1분 이상: 구매 고려
  }

  // 초기 신뢰도 계산
  private calculateInitialConfidence(type: UserInteraction['type']): number {
    const baseConfidence = {
      'view': 0.2,
      'like': 0.8,
      'inquiry': 0.9,
      'test_drive': 0.95,
      'share': 0.6
    };

    return baseConfidence[type] || 0.1;
  }

  // 상호작용 ID 생성
  private generateInteractionId(): string {
    return `int_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  // 실시간 컨텍스트 가져오기
  getRealtimeContext(): RealtimeRecommendationContext {
    this.updateCurrentPage();
    return { ...this.realtimeContext };
  }

  // 사용자 선호도 분석
  analyzeUserPreferences() {
    const vehicleViewCounts = new Map<string, number>();
    const totalViews = this.interactions.filter(i => i.type === 'view').length;

    // 차량별 조회 빈도 계산
    this.interactions.forEach(interaction => {
      if (interaction.type === 'view') {
        const count = vehicleViewCounts.get(interaction.vehicleId) || 0;
        vehicleViewCounts.set(interaction.vehicleId, count + 1);
      }
    });

    // 선호도 점수 계산
    const preferences = Array.from(vehicleViewCounts.entries()).map(([vehicleId, count]) => ({
      vehicleId,
      score: count / totalViews,
      viewCount: count
    })).sort((a, b) => b.score - a.score);

    return {
      totalInteractions: this.interactions.length,
      uniqueVehiclesViewed: vehicleViewCounts.size,
      averageViewDuration: this.calculateAverageViewDuration(),
      topPreferences: preferences.slice(0, 10),
      engagementLevel: this.calculateEngagementLevel()
    };
  }

  // 평균 조회 시간 계산
  private calculateAverageViewDuration(): number {
    const viewsWithDuration = this.interactions.filter(i => i.type === 'view' && i.duration);
    if (viewsWithDuration.length === 0) return 0;

    const totalDuration = viewsWithDuration.reduce((sum, i) => sum + (i.duration || 0), 0);
    return totalDuration / viewsWithDuration.length;
  }

  // 참여도 레벨 계산
  private calculateEngagementLevel(): 'low' | 'medium' | 'high' | 'very_high' {
    const totalInteractions = this.interactions.length;
    const explicitActions = this.interactions.filter(i => ['like', 'inquiry', 'test_drive'].includes(i.type)).length;

    const explicitRatio = totalInteractions > 0 ? explicitActions / totalInteractions : 0;

    if (explicitRatio > 0.3) return 'very_high';
    if (explicitRatio > 0.1) return 'high';
    if (totalInteractions > 10) return 'medium';
    return 'low';
  }

  // 서버로 데이터 전송
  async flushInteractions() {
    if (this.interactions.length === 0) return;

    try {
      const response = await fetch('/api/analytics/interactions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sessionId: this.sessionId,
          userId: this.userId,
          interactions: this.interactions,
          context: this.getRealtimeContext(),
          preferences: this.analyzeUserPreferences()
        }),
      });

      if (response.ok) {
        this.interactions = []; // 전송 성공 시 초기화
      }
    } catch (error) {
      console.error('Failed to flush interactions:', error);

      // 로컬 스토리지에 백업
      const backup = JSON.parse(localStorage.getItem('interaction_backup') || '[]');
      backup.push(...this.interactions);
      localStorage.setItem('interaction_backup', JSON.stringify(backup.slice(-1000))); // 최대 1000개 유지
    }
  }

  // 주기적 데이터 전송 (5분마다)
  startPeriodicFlush() {
    setInterval(() => {
      this.flushInteractions();
    }, 5 * 60 * 1000); // 5분
  }

  // 현재 세션 통계
  getSessionStats() {
    return {
      sessionId: this.sessionId,
      duration: Date.now() - this.sessionStartTime,
      interactions: this.interactions.length,
      uniqueVehicles: new Set(this.interactions.map(i => i.vehicleId)).size,
      averageConfidence: this.interactions.reduce((sum, i) => sum + i.confidence, 0) / this.interactions.length || 0
    };
  }
}

// 싱글톤 인스턴스 관리
let trackerInstance: UserBehaviorTracker | null = null;

export const getUserBehaviorTracker = (userId: string): UserBehaviorTracker => {
  if (!trackerInstance || trackerInstance['userId'] !== userId) {
    trackerInstance = new UserBehaviorTracker(userId);
    trackerInstance.startPeriodicFlush();
  }
  return trackerInstance;
};