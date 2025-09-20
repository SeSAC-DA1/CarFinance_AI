/**
 * CarFin AI Core 3-Agent System
 * 고정 룰: 총괄 정보수집 AI + 차량전문가 + 금융전문가
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
  creditScore?: 'excellent' | 'good' | 'fair' | 'poor';
  location?: string;
}

interface VehicleData {
  id: string;
  brand: string;
  model: string;
  year: number;
  price: number;
  mileage: number;
  fuel_type: string;
  location: string;
  dealer_info: any;
  features: string[];
  inspection_grade?: string;
  match_score?: number;
  analysis_result?: VehicleAnalysis;
}

interface VehicleAnalysis {
  performance_score: number;
  price_appropriateness: 'excellent' | 'good' | 'fair' | 'expensive';
  condition_assessment: string;
  recommendation_reason: string[];
  risk_factors: string[];
}

interface FinancialProduct {
  id: string;
  provider: string;
  product_name: string;
  type: 'loan' | 'lease' | 'installment';
  interest_rate: {
    min: number;
    max: number;
    typical: number;
  };
  monthly_payment_estimate: number;
  total_cost_estimate: number;
  requirements: string[];
  pros: string[];
  cons: string[];
  match_score: number;
  source_url?: string;
}

interface AgentCommunication {
  id: string;
  from_agent: 'coordinator' | 'vehicle_expert' | 'finance_expert';
  to_agent: 'coordinator' | 'vehicle_expert' | 'finance_expert';
  message_type: 'data_share' | 'request' | 'analysis_result' | 'decision_support';
  content: string;
  data: any;
  timestamp: Date;
  priority: 'low' | 'medium' | 'high';
}

interface SharedContext {
  user_profile: UserProfile;
  conversation_history: string[];
  vehicle_candidates: VehicleData[];
  financial_options: FinancialProduct[];
  agent_communications: AgentCommunication[];
  current_analysis_state: 'collecting' | 'analyzing' | 'matching' | 'recommending';
  inference_results: any;
}

class CoreThreeAgentSystem {
  private geminiApiKey: string;
  private googleSearchApiKey?: string;
  private sharedContext: SharedContext;

  constructor(
    geminiApiKey: string,
    googleSearchApiKey?: string
  ) {
    this.geminiApiKey = geminiApiKey;
    this.googleSearchApiKey = googleSearchApiKey;
    this.initializeSharedContext();
  }

  private initializeSharedContext() {
    this.sharedContext = {
      user_profile: {} as UserProfile,
      conversation_history: [],
      vehicle_candidates: [],
      financial_options: [],
      agent_communications: [],
      current_analysis_state: 'collecting',
      inference_results: {}
    };
  }

  /**
   * 메인 진입점 - 사용자 입력 처리
   */
  async processUserInput(userInput: string): Promise<{
    response: string;
    agent: 'coordinator' | 'vehicle_expert' | 'finance_expert';
    next_actions: string[];
    recommendations?: any[];
  }> {
    // 1. 총괄 정보수집 AI가 입력 처리
    const coordinatorResult = await this.coordinatorAgent(userInput);

    // 2. 필요시 다른 에이전트들과 협업
    if (coordinatorResult.needs_vehicle_analysis) {
      await this.triggerVehicleExpertAnalysis(coordinatorResult.inference_data);
    }

    if (coordinatorResult.needs_financial_analysis) {
      await this.triggerFinanceExpertAnalysis(coordinatorResult.inference_data);
    }

    return {
      response: coordinatorResult.response,
      agent: 'coordinator',
      next_actions: coordinatorResult.next_actions,
      recommendations: coordinatorResult.recommendations
    };
  }

  /**
   * 1. 총괄 정보수집 AI 🔍
   * - 사용자 데이터 수집
   * - 추론 및 결론 도출
   * - 다른 에이전트 의사결정 지원
   */
  private async coordinatorAgent(userInput: string): Promise<{
    response: string;
    inference_data: any;
    needs_vehicle_analysis: boolean;
    needs_financial_analysis: boolean;
    next_actions: string[];
    recommendations?: any[];
  }> {
    const prompt = `
당신은 CarFin AI의 총괄 정보수집 AI입니다. 다음 역할을 수행하세요:

1. 사용자 데이터 수집 및 분석
2. 수집된 정보를 바탕으로 추론 및 결론 도출
3. 차량전문가/금융전문가 에이전트의 의사결정 지원

현재 수집된 사용자 정보:
${JSON.stringify(this.sharedContext.user_profile, null, 2)}

사용자 입력: "${userInput}"

대화 기록:
${this.sharedContext.conversation_history.slice(-3).join('\n')}

다음을 분석하여 JSON으로 응답하세요:
{
  "collected_data": "사용자 입력에서 추출한 새로운 정보",
  "inference_analysis": "수집된 정보를 바탕으로 한 추론",
  "conclusion": "도출된 결론",
  "response": "사용자에게 보낼 자연스러운 응답",
  "needs_vehicle_analysis": "차량전문가 에이전트가 필요한가? (true/false)",
  "needs_financial_analysis": "금융전문가 에이전트가 필요한가? (true/false)",
  "decision_support_data": "다른 에이전트들에게 전달할 의사결정 지원 데이터",
  "next_actions": ["다음 진행할 작업들"]
}`;

    try {
      const result = await this.callGeminiAPI(prompt);
      const parsed = JSON.parse(this.extractJSONFromResponse(result.message));

      // 사용자 프로필 업데이트
      if (parsed.collected_data) {
        this.updateUserProfile(parsed.collected_data);
      }

      // 추론 결과 저장
      this.sharedContext.inference_results = parsed.inference_analysis;
      this.sharedContext.conversation_history.push(userInput);

      // 다른 에이전트들에게 의사결정 지원 데이터 전달
      if (parsed.decision_support_data) {
        await this.shareDecisionSupportData(parsed.decision_support_data);
      }

      return {
        response: parsed.response,
        inference_data: parsed.decision_support_data,
        needs_vehicle_analysis: parsed.needs_vehicle_analysis,
        needs_financial_analysis: parsed.needs_financial_analysis,
        next_actions: parsed.next_actions || []
      };
    } catch (error) {
      console.error('Coordinator agent failed:', error);
      return this.createFallbackCoordinatorResponse(userInput);
    }
  }

  /**
   * 2. 차량전문가 에이전트 🚗
   * - PostgreSQL DB 연결
   * - 실시간 매물 데이터 가져오기
   * - 종합 차량 분석
   */
  private async vehicleExpertAgent(decisionSupportData: any): Promise<{
    analysis_result: VehicleAnalysis[];
    matched_vehicles: VehicleData[];
    recommendations: string[];
  }> {
    // 1. PostgreSQL에서 실시간 매물 데이터 가져오기
    const vehicleListings = await this.fetchVehicleDataFromDB(decisionSupportData);

    // 2. 종합 차량 분석 수행
    const analysisPrompt = `
당신은 차량전문가 에이전트입니다. 다음 역할을 수행하세요:

1. 실시간 매물 데이터 분석
2. 종합 차량 분석 (성능, 가격, 상태)
3. 사용자 조건에 맞는 차량 매칭

사용자 조건 (총괄 AI로부터 전달):
${JSON.stringify(decisionSupportData, null, 2)}

매물 데이터:
${JSON.stringify(vehicleListings, null, 2)}

각 차량에 대해 다음 분석을 수행하고 JSON으로 응답하세요:
{
  "analysis_results": [
    {
      "vehicle_id": "차량 ID",
      "performance_score": "성능 점수 (0-100)",
      "price_appropriateness": "가격 적정성 (excellent/good/fair/expensive)",
      "condition_assessment": "상태 평가",
      "recommendation_reason": ["추천 이유들"],
      "risk_factors": ["위험 요소들"],
      "match_score": "매칭 점수 (0-100)"
    }
  ],
  "top_recommendations": ["상위 추천 차량들"],
  "analysis_summary": "종합 분석 요약"
}`;

    try {
      const result = await this.callGeminiAPI(analysisPrompt);
      const parsed = JSON.parse(this.extractJSONFromResponse(result.message));

      // 결과를 shared context에 저장
      this.sharedContext.vehicle_candidates = vehicleListings.map((vehicle, index) => ({
        ...vehicle,
        analysis_result: parsed.analysis_results[index],
        match_score: parsed.analysis_results[index]?.match_score || 0
      }));

      // 총괄 AI에게 분석 결과 전달
      await this.sendAgentCommunication(
        'vehicle_expert',
        'coordinator',
        'analysis_result',
        '차량 분석이 완료되었습니다.',
        parsed
      );

      return {
        analysis_result: parsed.analysis_results,
        matched_vehicles: this.sharedContext.vehicle_candidates,
        recommendations: parsed.top_recommendations
      };
    } catch (error) {
      console.error('Vehicle expert analysis failed:', error);
      return this.createFallbackVehicleAnalysis();
    }
  }

  /**
   * 3. 금융전문가 에이전트 💳
   * - Google Search API 사용
   * - 실시간 인터넷 금융정보 검색
   * - 최저금리 및 사용자 맞춤 금융상품 매칭
   */
  private async financeExpertAgent(decisionSupportData: any): Promise<{
    financial_products: FinancialProduct[];
    best_options: FinancialProduct[];
    market_analysis: any;
  }> {
    if (!this.googleSearchApiKey) {
      console.warn('Google Search API key not available');
      return this.createFallbackFinanceAnalysis();
    }

    // 1. Google Search API로 실시간 금융정보 검색
    const searchQueries = this.buildFinanceSearchQueries(decisionSupportData);
    const searchResults = await Promise.all(
      searchQueries.map(query => this.performGoogleSearch(query))
    );

    // 2. 검색 결과 분석 및 금융상품 매칭
    const analysisPrompt = `
당신은 금융전문가 에이전트입니다. 다음 역할을 수행하세요:

1. 실시간 인터넷 금융정보 분석
2. 최저금리 금융상품 발굴
3. 사용자 맞춤 금융상품 매칭

사용자 조건 (총괄 AI로부터 전달):
${JSON.stringify(decisionSupportData, null, 2)}

실시간 검색 결과:
${JSON.stringify(searchResults, null, 2)}

다음 분석을 수행하고 JSON으로 응답하세요:
{
  "financial_products": [
    {
      "provider": "제공업체",
      "product_name": "상품명",
      "type": "대출 타입",
      "interest_rate": {"min": 0, "max": 0, "typical": 0},
      "monthly_payment_estimate": "월 납부 예상액",
      "total_cost_estimate": "총 비용 예상액",
      "requirements": ["필요 조건들"],
      "pros": ["장점들"],
      "cons": ["단점들"],
      "match_score": "매칭 점수 (0-100)",
      "source_url": "출처 URL"
    }
  ],
  "lowest_rate_options": ["최저금리 옵션들"],
  "personalized_recommendations": ["개인화 추천"],
  "market_analysis": "시장 분석 결과"
}`;

    try {
      const result = await this.callGeminiAPI(analysisPrompt);
      const parsed = JSON.parse(this.extractJSONFromResponse(result.message));

      // 결과를 shared context에 저장
      this.sharedContext.financial_options = parsed.financial_products;

      // 총괄 AI에게 분석 결과 전달
      await this.sendAgentCommunication(
        'finance_expert',
        'coordinator',
        'analysis_result',
        '금융상품 분석이 완료되었습니다.',
        parsed
      );

      return {
        financial_products: parsed.financial_products,
        best_options: parsed.lowest_rate_options,
        market_analysis: parsed.market_analysis
      };
    } catch (error) {
      console.error('Finance expert analysis failed:', error);
      return this.createFallbackFinanceAnalysis();
    }
  }

  /**
   * Backend API를 통해 PostgreSQL DB에서 실시간 매물 데이터 가져오기
   */
  private async fetchVehicleDataFromDB(criteria: any): Promise<VehicleData[]> {
    try {
      console.log('📊 Backend API 호출 - PostgreSQL DB에서 실시간 매물 조회');

      // 백엔드 서버 URL (개발환경: localhost:8000)
      const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

      const response = await fetch(`${apiBaseUrl}/vehicles/search/user`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          budget: criteria.budget,
          fuelType: criteria.fuelType,
          purpose: criteria.purpose,
          location: criteria.location,
          preferences: criteria.preferences
        })
      });

      if (!response.ok) {
        throw new Error(`Backend API error: ${response.status}`);
      }

      const vehicles = await response.json();

      // 백엔드 데이터를 프론트엔드 형식으로 변환
      return vehicles.map((vehicle: any) => ({
        id: vehicle.id,
        brand: vehicle.make || '현대',
        model: vehicle.model || '소나타',
        year: vehicle.year || 2020,
        price: vehicle.price || 2500,
        mileage: vehicle.mileage || 50000,
        fuel_type: vehicle.details?.fuel_type || criteria.fuelType || '가솔린',
        location: vehicle.details?.location || criteria.location || '서울',
        dealer_info: {
          name: vehicle.details?.dealer || '인증딜러',
          rating: vehicle.details?.dealer_rating || 4.0
        },
        features: vehicle.details?.features || ['네비게이션', '후방카메라'],
        inspection_grade: vehicle.details?.inspection_grade || '우수'
      }));

    } catch (error) {
      console.error('Backend API 호출 실패:', error);

      // Fallback: Mock 데이터 반환
      console.log('📊 Fallback - Mock 데이터 사용');
      const mockVehicles: VehicleData[] = [
        {
          id: 'db_001',
          brand: '현대',
          model: '아반떼',
          year: 2022,
          price: criteria.budget ? Math.min(criteria.budget * 0.9, 2800) : 2800,
          mileage: 15000,
          fuel_type: criteria.fuelType || '가솔린',
          location: '서울 강남구',
          dealer_info: { name: '강남모터스', rating: 4.5 },
          features: ['후방카메라', '블루투스', '크루즈컨트롤'],
          inspection_grade: '1급'
        },
        {
          id: 'db_002',
          brand: '기아',
          model: 'K5',
          year: 2021,
          price: criteria.budget ? Math.min(criteria.budget * 0.95, 3200) : 3200,
          mileage: 28000,
          fuel_type: '하이브리드',
          location: '경기 성남시',
          dealer_info: { name: 'K모터스', rating: 4.3 },
          features: ['선루프', '통풍시트', '어댑티브크루즈'],
          inspection_grade: '2급'
        }
      ];

      return mockVehicles;
    }
  }

  /**
   * Google Search API 실행
   */
  private async performGoogleSearch(query: string): Promise<any> {
    if (!this.googleSearchApiKey) {
      return { query, results: [] };
    }

    try {
      console.log(`🔍 Google Search API 실행: ${query}`);
      const response = await fetch(
        `https://www.googleapis.com/customsearch/v1?key=${this.googleSearchApiKey}&q=${encodeURIComponent(query)}&num=5`
      );

      if (!response.ok) {
        throw new Error(`Google Search API error: ${response.status}`);
      }

      const data = await response.json();
      return {
        query,
        results: data.items || []
      };
    } catch (error) {
      console.error(`Google Search failed for: ${query}`, error);
      return { query, results: [] };
    }
  }

  private buildFinanceSearchQueries(criteria: any): string[] {
    const budget = criteria.budget || 3000;
    const queries = [
      `중고차 대출 ${budget}만원 최저금리 2024`,
      `자동차 할부 금리 비교 2024`,
      `중고차 리스 조건 ${budget}만원 2024`,
      `자동차 금융 최신 금리 순위`
    ];

    if (criteria.creditScore) {
      queries.push(`${criteria.creditScore} 신용등급 자동차 대출 금리`);
    }

    return queries;
  }

  /**
   * 에이전트 간 소통
   */
  private async sendAgentCommunication(
    from: 'coordinator' | 'vehicle_expert' | 'finance_expert',
    to: 'coordinator' | 'vehicle_expert' | 'finance_expert',
    type: 'data_share' | 'request' | 'analysis_result' | 'decision_support',
    content: string,
    data: any
  ): Promise<void> {
    const communication: AgentCommunication = {
      id: Date.now().toString(),
      from_agent: from,
      to_agent: to,
      message_type: type,
      content,
      data,
      timestamp: new Date(),
      priority: 'medium'
    };

    this.sharedContext.agent_communications.push(communication);
    console.log(`📡 Agent Communication: ${from} → ${to}: ${content}`);
  }

  private async shareDecisionSupportData(data: any): Promise<void> {
    await this.sendAgentCommunication(
      'coordinator',
      'vehicle_expert',
      'decision_support',
      '사용자 분석 결과를 공유합니다.',
      data
    );

    await this.sendAgentCommunication(
      'coordinator',
      'finance_expert',
      'decision_support',
      '사용자 분석 결과를 공유합니다.',
      data
    );
  }

  private async triggerVehicleExpertAnalysis(data: any): Promise<void> {
    console.log('🚗 차량전문가 에이전트 활성화');
    await this.vehicleExpertAgent(data);
  }

  private async triggerFinanceExpertAnalysis(data: any): Promise<void> {
    console.log('💳 금융전문가 에이전트 활성화');
    await this.financeExpertAgent(data);
  }

  // 유틸리티 메서드들
  private updateUserProfile(newData: any): void {
    this.sharedContext.user_profile = { ...this.sharedContext.user_profile, ...newData };
  }

  private async callGeminiAPI(prompt: string): Promise<any> {
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

      return { message: content };
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

  // Fallback 메서드들
  private createFallbackCoordinatorResponse(userInput: string) {
    return {
      response: '정보를 분석 중입니다. 조금 더 자세히 말씀해주시겠어요?',
      inference_data: { userInput },
      needs_vehicle_analysis: false,
      needs_financial_analysis: false,
      next_actions: ['추가 정보 수집']
    };
  }

  private createFallbackVehicleAnalysis() {
    return {
      analysis_result: [],
      matched_vehicles: [],
      recommendations: ['추가 정보가 필요합니다']
    };
  }

  private createFallbackFinanceAnalysis() {
    return {
      financial_products: [],
      best_options: [],
      market_analysis: '분석 정보가 부족합니다'
    };
  }
}

export {
  CoreThreeAgentSystem,
  type UserProfile,
  type VehicleData,
  type FinancialProduct,
  type AgentCommunication,
  type SharedContext
};