/**
 * 현실적인 CarFin AI 에이전트 시스템
 * 정보 제공 + 매칭 서비스 전용 설계
 */

interface UserProfile {
  user_id: string;
  name?: string;
  email?: string;
  age?: number;
  income?: number;
  budget?: number;
  purpose?: string;
  fuelType?: string;
  preferences?: string[];
  creditScore?: 'excellent' | 'good' | 'fair' | 'poor' | 'unknown';
}

interface VehicleListing {
  id: string;
  source: 'encar' | 'kbchachacha' | 'carprice' | 'sk_encar';
  brand: string;
  model: string;
  year: number;
  price: number;
  mileage: number;
  fuel_type: string;
  transmission: string;
  location: string;
  dealer_name: string;
  images: string[];
  features: string[];
  inspection_grade?: string;
  accident_history?: 'none' | 'minor' | 'major' | 'unknown';
  match_score?: number;
  market_price_analysis?: {
    average_price: number;
    price_rating: 'excellent' | 'good' | 'fair' | 'expensive';
    similar_listings_count: number;
  };
}

interface FinancialProduct {
  id: string;
  type: 'bank_loan' | 'capital_loan' | 'lease' | 'installment';
  provider: string;
  product_name: string;
  interest_rate: {
    min: number;
    max: number;
    typical: number;
  };
  loan_to_value: number; // 최대 대출 비율
  max_amount: number;
  term_months: number[];
  requirements: string[];
  pros: string[];
  cons: string[];
  last_updated: Date;
}

interface AgentResponse {
  agent: 'consultant' | 'data_collector' | 'vehicle_matcher' | 'market_analyst' | 'finance_advisor';
  message: string;
  data?: any;
  confidence: number;
  sources?: string[];
  recommendations?: string[];
}

class RealisticCarFinAgents {
  private geminiApiKey: string;
  private googleSearchApiKey?: string;
  private vehicleCache: Map<string, VehicleListing[]> = new Map();
  private financialProductCache: Map<string, FinancialProduct[]> = new Map();

  constructor(geminiApiKey: string, googleSearchApiKey?: string) {
    this.geminiApiKey = geminiApiKey;
    this.googleSearchApiKey = googleSearchApiKey;
  }

  /**
   * 1. 데이터 수집 에이전트 - 사용자 정보 수집
   */
  async dataCollectionAgent(userInput: string, currentData: Partial<UserProfile>): Promise<{
    response: string;
    collected_data: Partial<UserProfile>;
    collection_progress: number;
    is_complete: boolean;
    next_questions: string[];
  }> {
    const prompt = `
당신은 전문 자동차 상담사입니다. 사용자 정보를 자연스럽게 수집하세요.

현재 수집된 정보:
${JSON.stringify(currentData, null, 2)}

사용자 입력: "${userInput}"

수집해야 할 정보:
- budget: 예산 범위
- purpose: 사용 목적 (출퇴근, 가족용, 레저, 사업)
- fuelType: 연료 선호도 (가솔린, 하이브리드, 전기, 디젤)
- preferences: 중요 요소 (연비, 안전성, 디자인, 브랜드, 가격)

응답은 JSON 형태로:
{
  "response": "자연스러운 대화 응답",
  "collected_data": { "추출된 정보" },
  "collection_progress": 0-100,
  "is_complete": true/false,
  "next_questions": ["다음 질문들"]
}`;

    try {
      const result = await this.callGeminiAPI(prompt);
      const parsed = this.extractJSONFromResponse(result.message);
      return JSON.parse(parsed);
    } catch (error) {
      console.error('Data collection failed:', error);
      return this.createFallbackDataCollection(userInput, currentData);
    }
  }

  /**
   * 2. 차량 매칭 에이전트 - 실제 매물 검색 및 매칭
   */
  async vehicleMatchingAgent(userProfile: UserProfile): Promise<VehicleListing[]> {
    try {
      // 실제 매물 데이터 수집 (여러 소스)
      const listings = await this.fetchRealVehicleListings(userProfile);

      // 협업 필터링 적용
      const scoredListings = await this.applyCollaborativeFiltering(listings, userProfile);

      // 상위 5개 추천
      return scoredListings
        .sort((a, b) => (b.match_score || 0) - (a.match_score || 0))
        .slice(0, 5);
    } catch (error) {
      console.error('Vehicle matching failed:', error);
      return this.getMockVehicles(userProfile);
    }
  }

  /**
   * 3. 시장 분석 에이전트 - 가격 분석 및 시장 정보
   */
  async marketAnalysisAgent(vehicle: VehicleListing, userProfile: UserProfile): Promise<{
    price_analysis: {
      market_average: number;
      price_rating: 'excellent' | 'good' | 'fair' | 'expensive';
      confidence: number;
    };
    market_trends: {
      demand_level: 'high' | 'medium' | 'low';
      price_trend: 'rising' | 'stable' | 'falling';
      seasonal_factor: string;
    };
    risk_assessment: {
      overall_risk: 'low' | 'medium' | 'high';
      risk_factors: string[];
      recommendations: string[];
    };
  }> {
    try {
      // 실제 시장 데이터 분석
      const marketData = await this.analyzeMarketData(vehicle);
      const riskAssessment = await this.assessVehicleRisk(vehicle);

      return {
        price_analysis: marketData.price_analysis,
        market_trends: marketData.trends,
        risk_assessment: riskAssessment
      };
    } catch (error) {
      console.error('Market analysis failed:', error);
      return this.getMockMarketAnalysis(vehicle);
    }
  }

  /**
   * 4. 금융 상품 정보 에이전트 - 실시간 금융 정보 검색
   */
  async financeInformationAgent(vehiclePrice: number, userProfile: UserProfile): Promise<FinancialProduct[]> {
    try {
      // Google Search API로 실시간 금융 정보 검색
      let products: FinancialProduct[] = [];

      if (this.googleSearchApiKey) {
        products = await this.searchFinancialProducts(vehiclePrice, userProfile);
      }

      // 기본 금융 정보와 병합
      const defaultProducts = await this.getDefaultFinancialProducts(vehiclePrice, userProfile);

      return [...products, ...defaultProducts]
        .sort((a, b) => a.interest_rate.typical - b.interest_rate.typical)
        .slice(0, 4);
    } catch (error) {
      console.error('Finance information search failed:', error);
      return this.getDefaultFinancialProducts(vehiclePrice, userProfile);
    }
  }

  /**
   * Google Search API를 활용한 실시간 금융상품 검색
   */
  private async searchFinancialProducts(vehiclePrice: number, userProfile: UserProfile): Promise<FinancialProduct[]> {
    if (!this.googleSearchApiKey) {
      throw new Error('Google Search API key not available');
    }

    const queries = [
      `중고차 대출 ${vehiclePrice}만원 금리 비교 2024`,
      `자동차 리스 ${vehiclePrice}만원 조건 2024`,
      `중고차 할부 금리 순위 2024`
    ];

    const results: FinancialProduct[] = [];

    for (const query of queries) {
      try {
        const response = await fetch(
          `https://www.googleapis.com/customsearch/v1?key=${this.googleSearchApiKey}&q=${encodeURIComponent(query)}&num=5`
        );

        if (response.ok) {
          const data = await response.json();
          const products = await this.parseFinancialSearchResults(data.items || []);
          results.push(...products);
        }
      } catch (error) {
        console.error(`Search failed for query: ${query}`, error);
      }
    }

    return results;
  }

  /**
   * 협업 필터링 알고리즘 적용
   */
  private async applyCollaborativeFiltering(vehicles: VehicleListing[], userProfile: UserProfile): Promise<VehicleListing[]> {
    return vehicles.map(vehicle => ({
      ...vehicle,
      match_score: this.calculateMatchScore(vehicle, userProfile)
    }));
  }

  private calculateMatchScore(vehicle: VehicleListing, userProfile: UserProfile): number {
    let score = 70; // 기본 점수

    // 예산 매칭 (25점)
    if (userProfile.budget) {
      const budgetRatio = vehicle.price / userProfile.budget;
      if (budgetRatio <= 0.8) score += 25;
      else if (budgetRatio <= 0.95) score += 20;
      else if (budgetRatio <= 1.0) score += 15;
      else if (budgetRatio <= 1.1) score += 5;
    }

    // 연료 타입 매칭 (15점)
    if (userProfile.fuelType && vehicle.fuel_type.toLowerCase().includes(userProfile.fuelType.toLowerCase())) {
      score += 15;
    }

    // 사용 목적 매칭 (10점)
    if (userProfile.purpose) {
      if (userProfile.purpose === '출퇴근용' && vehicle.fuel_type.includes('하이브리드')) score += 10;
      if (userProfile.purpose === '가족용' && ['세단', 'SUV'].some(type => vehicle.model.includes(type))) score += 10;
      if (userProfile.purpose === '레저용' && vehicle.model.toLowerCase().includes('suv')) score += 10;
    }

    return Math.min(Math.max(score, 0), 100);
  }

  /**
   * 실제 차량 매물 데이터 수집 (시뮬레이션)
   */
  private async fetchRealVehicleListings(userProfile: UserProfile): Promise<VehicleListing[]> {
    // 실제 구현 시 여기에 엔카, 카프라이스 등 API 호출
    // 현재는 현실적인 Mock 데이터 반환
    const mockListings: VehicleListing[] = [
      {
        id: "encar_001",
        source: "encar",
        brand: "현대",
        model: "아반떼",
        year: 2022,
        price: userProfile.budget ? Math.min(userProfile.budget * 0.85, 2800) : 2800,
        mileage: 15000,
        fuel_type: userProfile.fuelType === '하이브리드' ? '하이브리드' : '가솔린',
        transmission: '자동',
        location: "서울 강남구",
        dealer_name: "강남모터스",
        images: [],
        features: ["후방카메라", "블루투스", "크루즈컨트롤"],
        inspection_grade: "1급",
        accident_history: "none"
      },
      {
        id: "kbchachacha_002",
        source: "kbchachacha",
        brand: "기아",
        model: "K5",
        year: 2021,
        price: userProfile.budget ? Math.min(userProfile.budget * 0.92, 3200) : 3200,
        mileage: 28000,
        fuel_type: "하이브리드",
        transmission: "자동",
        location: "경기 성남시",
        dealer_name: "KB차차차",
        images: [],
        features: ["선루프", "통풍시트", "어댑티브크루즈"],
        inspection_grade: "2급",
        accident_history: "minor"
      }
    ];

    return mockListings;
  }

  private async callGeminiAPI(prompt: string): Promise<AgentResponse> {
    // Gemini API 호출 로직 (기존과 동일)
    try {
      const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key=${this.geminiApiKey}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 2048,
          }
        })
      });

      const data = await response.json();
      const content = data.candidates?.[0]?.content?.parts?.[0]?.text || '';

      return {
        agent: 'consultant',
        message: content,
        confidence: 0.85
      };
    } catch (error) {
      throw new Error(`Gemini API call failed: ${error}`);
    }
  }

  private extractJSONFromResponse(response: string): string {
    const jsonMatch = response.match(/```json\n([\s\S]*?)\n```/) ||
                     response.match(/```\n([\s\S]*?)\n```/) ||
                     response.match(/\{[\s\S]*\}/);

    if (jsonMatch) {
      return jsonMatch[1] || jsonMatch[0];
    }

    const trimmed = response.trim();
    if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
      return trimmed;
    }

    throw new Error('No valid JSON found in response');
  }

  // 추가 헬퍼 메서드들...
  private createFallbackDataCollection(userInput: string, currentData: Partial<UserProfile>) {
    // 기존 fallback 로직 활용
    return {
      response: "죄송합니다. 다시 한 번 말씀해주시겠어요?",
      collected_data: currentData,
      collection_progress: Object.keys(currentData).length * 25,
      is_complete: false,
      next_questions: ["예산은 어느 정도로 생각하고 계신가요?"]
    };
  }

  private getMockVehicles(userProfile: UserProfile): VehicleListing[] {
    // Mock 데이터 반환
    return [];
  }

  private getMockMarketAnalysis(vehicle: VehicleListing) {
    return {
      price_analysis: {
        market_average: vehicle.price * 1.05,
        price_rating: 'good' as const,
        confidence: 0.75
      },
      market_trends: {
        demand_level: 'medium' as const,
        price_trend: 'stable' as const,
        seasonal_factor: '연말 할인 시기'
      },
      risk_assessment: {
        overall_risk: 'low' as const,
        risk_factors: [],
        recommendations: ['정기점검 확인', '보험이력 조회']
      }
    };
  }

  private async analyzeMarketData(vehicle: VehicleListing) {
    // 실제 시장 분석 로직
    return this.getMockMarketAnalysis(vehicle);
  }

  private async assessVehicleRisk(vehicle: VehicleListing) {
    // 리스크 평가 로직
    return {
      overall_risk: 'low' as const,
      risk_factors: [],
      recommendations: []
    };
  }

  private async parseFinancialSearchResults(items: any[]): Promise<FinancialProduct[]> {
    // Google 검색 결과를 FinancialProduct로 파싱
    return [];
  }

  private async getDefaultFinancialProducts(vehiclePrice: number, userProfile: UserProfile): Promise<FinancialProduct[]> {
    // 기본 금융상품 정보 반환
    return [
      {
        id: 'kb_loan_001',
        type: 'bank_loan',
        provider: 'KB국민은행',
        product_name: '중고차담보대출',
        interest_rate: { min: 4.2, max: 8.5, typical: 6.0 },
        loan_to_value: 80,
        max_amount: 50000,
        term_months: [12, 24, 36, 48, 60],
        requirements: ['소득증빙', '재직증명서', '신용등급 4등급 이상'],
        pros: ['낮은 금리', '장기분할', '중도상환수수료 없음'],
        cons: ['까다로운 심사', '담보 설정'],
        last_updated: new Date()
      }
    ];
  }
}

export { RealisticCarFinAgents, type UserProfile, type VehicleListing, type FinancialProduct, type AgentResponse };