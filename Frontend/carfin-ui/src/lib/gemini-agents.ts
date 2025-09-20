/**
 * Gemini Multi-Agent System for CarFin
 * 2개 에이전트 (차량 추천 + 금융 상담)의 실시간 협업 필터링 시스템
 */

interface UserProfile {
  user_id: string;
  name: string;
  email: string;
  age: number;
  income: number;
  preferences: string[];
  purpose: string;
  budget?: number;
  fuelType?: string;
}

interface Vehicle {
  id: string;
  brand: string;
  model: string;
  year: number;
  price: number;
  mileage: number;
  fuel_type: string;
  body_type: string;
  color: string;
  location: string;
  images: string[];
  features: string[];
  fuel_efficiency: number;
  safety_rating: number;
  match_score: number;
  description?: string;
}

interface AgentResponse {
  agent: 'vehicle_expert' | 'finance_expert';
  message: string;
  data?: any;
  confidence: number;
  next_questions?: string[];
}

interface CollaborativeResult {
  vehicles: Vehicle[];
  financial_options: FinancialOption[];
  recommendations: string[];
  agent_discussion: AgentResponse[];
}

interface FinancialOption {
  type: string;
  monthly_payment: number;
  total_cost: number;
  down_payment: number;
  description: string;
  pros: string[];
  cons: string[];
}

class GeminiMultiAgent {
  private apiKey: string;
  private googleSearchApiKey?: string;
  private customSearchEngineId?: string;
  private apiUrl = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent';

  constructor(apiKey: string, googleSearchApiKey?: string, customSearchEngineId?: string) {
    this.apiKey = apiKey;
    this.googleSearchApiKey = googleSearchApiKey;
    this.customSearchEngineId = customSearchEngineId;
  }

  /**
   * 2개 에이전트의 실시간 협업 대화 시뮬레이션
   */
  async collaborativeRecommendation(userInput: string, userProfile?: Partial<UserProfile>): Promise<CollaborativeResult> {
    const vehicleExpertPrompt = this.createVehicleExpertPrompt(userInput, userProfile);
    const financeExpertPrompt = this.createFinanceExpertPrompt(userInput, userProfile);

    try {
      // 두 에이전트 동시 실행
      const [vehicleResponse, financeResponse] = await Promise.all([
        this.callGeminiAPI(vehicleExpertPrompt, 'vehicle_expert'),
        this.callGeminiAPI(financeExpertPrompt, 'finance_expert')
      ]);

      // 에이전트 간 상호 검토 및 협업
      const collaborationPrompt = this.createCollaborationPrompt(
        vehicleResponse.message,
        financeResponse.message,
        userInput,
        userProfile
      );

      const finalResult = await this.callGeminiAPI(collaborationPrompt, 'collaboration');

      // 실시간 협업 필터링 결과 생성
      return this.processCollaborativeResult(vehicleResponse, financeResponse, finalResult, userProfile);

    } catch (error) {
      console.error('Collaborative recommendation failed:', error);
      throw new Error('멀티에이전트 협업 추천 중 오류가 발생했습니다.');
    }
  }

  /**
   * 상담사 AI - 초기 상담 및 환영
   */
  async consultantAgent(userInput: string): Promise<{
    response: string;
    agent: 'consultant';
  }> {
    const prompt = `
당신은 CarFin AI의 메인 상담사입니다. 따뜻하고 전문적인 톤으로 고객을 맞이하고 안내하세요.

사용자 입력: "${userInput}"

역할:
- 친근한 환영 인사
- 서비스 간단 소개
- 다음 단계 안내
- 고객 불안감 해소

응답 스타일:
- 전문적이지만 친근함
- 간결하고 명확한 안내
- 고객 중심적 사고

응답은 반드시 다음 JSON 형식으로만 답변하세요:
{
  "response": "상담사 응답 메시지",
  "agent": "consultant"
}`;

    try {
      const result = await this.callGeminiAPI(prompt, 'consultant');
      return JSON.parse(this.extractJSONFromResponse(result.message));
    } catch (error) {
      return {
        response: "안녕하세요! CarFin AI 상담사입니다. 최적의 중고차를 찾아드리겠습니다.",
        agent: 'consultant'
      };
    }
  }

  /**
   * 자연스러운 대화형 정보 수집
   */
  async conversationalDataCollection(userInput: string, currentData: Partial<UserProfile> = {}): Promise<{
    response: string;
    collected_data: Partial<UserProfile>;
    next_question: string;
    collection_progress: number;
    is_complete: boolean;
  }> {
    const prompt = `
당신은 친근한 자동차 상담사입니다. 사용자와 자연스러운 대화를 통해 차량 추천에 필요한 정보를 점진적으로 수집하세요.

현재 수집된 정보:
${JSON.stringify(currentData, null, 2)}

사용자 입력: "${userInput}"

다음 규칙을 따르세요:
1. 한 번에 하나의 정보만 자연스럽게 물어보세요
2. 딱딱한 설문조사가 아닌 친근한 대화 형태로 진행하세요
3. 사용자의 답변에서 숨겨진 정보도 추출하세요
4. 수집 순서: 예산 → 용도 → 연료타입 → 기타 선호사항

필요한 정보 목록:
- budget (예산)
- purpose (용도: 출퇴근, 가족용, 레저 등)
- fuelType (연료: 가솔린, 하이브리드, 전기)
- preferences (선호사항: 연비, 안전성, 디자인 등)

응답은 반드시 다음 JSON 형식으로만 답변하세요. 다른 텍스트는 포함하지 마세요:
{
  "response": "친근한 대화 응답",
  "collected_data": { 추출된 정보 },
  "next_question": "다음 질문",
  "collection_progress": 진행률(0-100),
  "is_complete": false/true
}`;

    try {
      const result = await this.callGeminiAPI(prompt, 'data_collector');
      const cleanedMessage = this.extractJSONFromResponse(result.message);
      return JSON.parse(cleanedMessage);
    } catch (error) {
      console.error('Conversational data collection failed:', error);
      return this.createFallbackResponse(userInput, currentData);
    }
  }

  private extractJSONFromResponse(response: string): string {
    try {
      // JSON 블록 패턴 찾기
      const jsonMatch = response.match(/```json\n([\s\S]*?)\n```/) ||
                       response.match(/```\n([\s\S]*?)\n```/) ||
                       response.match(/\{[\s\S]*\}/);

      if (jsonMatch) {
        return jsonMatch[1] || jsonMatch[0];
      }

      // 만약 이미 JSON 형태라면 그대로 반환
      const trimmed = response.trim();
      if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
        return trimmed;
      }

      throw new Error('No JSON found in response');
    } catch (error) {
      console.error('JSON extraction failed:', error);
      throw error;
    }
  }

  private createFallbackResponse(userInput: string, currentData: Partial<UserProfile>) {
    // 현재 데이터 기반으로 다음 질문 결정
    const dataKeys = Object.keys(currentData);

    if (!dataKeys.includes('budget')) {
      return {
        response: `${userInput}에 대해 말씀해주셔서 감사합니다! 차량 선택에 도움이 되도록 예산 범위를 알려주시겠어요?`,
        collected_data: this.extractDataFromInput(userInput, currentData),
        next_question: "예산은 어느 정도로 생각하고 계신가요?",
        collection_progress: 20,
        is_complete: false
      };
    }

    if (!dataKeys.includes('purpose')) {
      return {
        response: "예산을 알려주셔서 감사합니다! 차량을 주로 어떤 용도로 사용하실 예정인가요?",
        collected_data: this.extractDataFromInput(userInput, currentData),
        next_question: "출퇴근용인가요, 가족용인가요?",
        collection_progress: 40,
        is_complete: false
      };
    }

    if (!dataKeys.includes('fuelType')) {
      return {
        response: "사용 용도를 알겠습니다! 연료 타입은 어떤 것을 선호하시나요?",
        collected_data: this.extractDataFromInput(userInput, currentData),
        next_question: "가솔린, 하이브리드, 전기차 중에서요?",
        collection_progress: 60,
        is_complete: false
      };
    }

    if (!dataKeys.includes('preferences') || (currentData.preferences && currentData.preferences.length === 0)) {
      return {
        response: "연료 타입 선호도를 알겠습니다! 차량 선택 시 가장 중요하게 생각하시는 요소는 무엇인가요?",
        collected_data: this.extractDataFromInput(userInput, currentData),
        next_question: "연비, 안전성, 디자인, 브랜드 중에 우선순위가 있으신가요?",
        collection_progress: 80,
        is_complete: false
      };
    }

    // 모든 정보 수집 완료
    return {
      response: "감사합니다! 모든 정보를 수집했습니다. 이제 맞춤형 차량을 찾아드리겠습니다!",
      collected_data: this.extractDataFromInput(userInput, currentData),
      next_question: "",
      collection_progress: 100,
      is_complete: true
    };
  }

  private extractDataFromInput(userInput: string, currentData: Partial<UserProfile>): Partial<UserProfile> {
    const extractedData = { ...currentData };
    const input = userInput.toLowerCase();

    // 예산 추출
    if (input.includes('만원') || input.includes('천만') || input.includes('억')) {
      const budgetMatch = input.match(/(\d+)(?:천만|만원|억)/);
      if (budgetMatch) {
        const amount = parseInt(budgetMatch[1]);
        if (input.includes('천만')) {
          extractedData.budget = amount * 1000;
        } else if (input.includes('억')) {
          extractedData.budget = amount * 10000;
        } else {
          extractedData.budget = amount;
        }
      }
    }

    // 용도 추출
    if (input.includes('출퇴근') || input.includes('통근')) {
      extractedData.purpose = '출퇴근용';
    } else if (input.includes('가족') || input.includes('패밀리')) {
      extractedData.purpose = '가족용';
    } else if (input.includes('레저') || input.includes('여행')) {
      extractedData.purpose = '레저용';
    } else if (input.includes('사업') || input.includes('업무')) {
      extractedData.purpose = '사업용';
    }

    // 연료 타입 추출
    if (input.includes('가솔린')) {
      extractedData.fuelType = '가솔린';
    } else if (input.includes('하이브리드')) {
      extractedData.fuelType = '하이브리드';
    } else if (input.includes('전기') || input.includes('ev')) {
      extractedData.fuelType = '전기차';
    } else if (input.includes('디젤')) {
      extractedData.fuelType = '디젤';
    }

    // 선호사항 추출
    if (!extractedData.preferences) {
      extractedData.preferences = [];
    }

    if (input.includes('연비')) extractedData.preferences.push('연비 중시');
    if (input.includes('안전')) extractedData.preferences.push('안전성 중시');
    if (input.includes('디자인') || input.includes('외관')) extractedData.preferences.push('디자인 중시');
    if (input.includes('브랜드')) extractedData.preferences.push('브랜드 중시');

    return extractedData;
  }

  /**
   * 실시간 금융상품 검색 에이전트
   */
  async searchFinancialProducts(vehiclePrice: number, userProfile: Partial<UserProfile>): Promise<FinancialOption[]> {
    if (!this.googleSearchApiKey || !this.customSearchEngineId) {
      console.log('Google Search API not configured, using default financial products');
      return this.getMockFinancialOptions();
    }

    try {
      // 사용자 프로필 기반 맞춤 검색
      const searchQueries = this.buildFinanceSearchQueries(vehiclePrice, userProfile);
      const allResults: FinancialOption[] = [];

      for (const query of searchQueries) {
        try {
          const results = await this.performFinanceSearch(query);
          allResults.push(...results);
        } catch (error) {
          console.error(`Finance search failed for query: ${query}`, error);
        }
      }

      // 결과 정렬 및 중복 제거
      const uniqueResults = this.deduplicateFinancialOptions(allResults);
      return uniqueResults
        .sort((a, b) => a.monthly_payment - b.monthly_payment)
        .slice(0, 4);

    } catch (error) {
      console.error('Real-time financial search failed:', error);
      return this.getMockFinancialOptions();
    }
  }

  /**
   * 금융상품 검색 쿼리 생성
   */
  private buildFinanceSearchQueries(vehiclePrice: number, userProfile: Partial<UserProfile>): string[] {
    const queries = [
      `중고차 대출 ${vehiclePrice}만원 금리 비교 2024`,
      `자동차 할부 금리 순위 2024`,
      `자동차 리스 ${vehiclePrice}만원 조건 2024`
    ];

    // 예산 기반 추가 쿼리
    if (userProfile.budget) {
      queries.push(`예산 ${userProfile.budget}만원 자동차 대출 2024`);
    }

    // 소득 기반 추가 쿼리 (개인정보이므로 일반화)
    if (userProfile.income && userProfile.income > 3000) {
      queries.push(`고소득 자동차 대출 우대금리 2024`);
    }

    return queries;
  }

  /**
   * Google Search API 실행
   */
  private async performFinanceSearch(query: string): Promise<FinancialOption[]> {
    const url = new URL('https://www.googleapis.com/customsearch/v1');
    url.searchParams.append('key', this.googleSearchApiKey!);
    url.searchParams.append('cx', this.customSearchEngineId!);
    url.searchParams.append('q', query);
    url.searchParams.append('num', '5');
    url.searchParams.append('dateRestrict', 'm3'); // 최근 3개월

    const response = await fetch(url.toString());

    if (!response.ok) {
      throw new Error(`Google Search API error: ${response.status}`);
    }

    const data = await response.json();

    if (!data.items) {
      return [];
    }

    return this.parseFinanceSearchResults(data.items);
  }

  /**
   * 검색 결과 파싱 및 금융상품 정보 추출
   */
  private parseFinanceSearchResults(items: any[]): FinancialOption[] {
    const results: FinancialOption[] = [];

    for (const item of items) {
      try {
        const { title, link, snippet } = item;
        const domain = new URL(link).hostname;

        // 신뢰할 수 있는 금융기관 확인
        const trustedSources = ['kb.co.kr', 'shinhan.com', 'wooribank.com', 'hanabank.com', 'nhcapital.co.kr'];
        const isTrusted = trustedSources.some(source => domain.includes(source));

        if (!isTrusted) continue;

        // 금리 정보 추출
        const interestRates = this.extractInterestRates(snippet + ' ' + title);
        if (interestRates.length === 0) continue;

        // 제공업체 이름 추출
        const provider = this.extractProviderName(domain, title);

        // 금융상품 객체 생성
        const financialOption: FinancialOption = {
          type: this.determineFinanceType(title + snippet),
          monthly_payment: 0, // 계산 필요
          total_cost: 0, // 계산 필요
          down_payment: 0, // 기본값
          description: `${provider} - 실시간 검색 결과`,
          pros: [`신뢰할 수 있는 ${provider}`, '최신 금리 정보'],
          cons: ['상세 조건 확인 필요', '신용도에 따라 변동']
        };

        results.push(financialOption);

      } catch (error) {
        console.error('Failed to parse search result:', error);
      }
    }

    return results;
  }

  /**
   * 텍스트에서 금리 정보 추출
   */
  private extractInterestRates(text: string): number[] {
    const rates: number[] = [];

    // 금리 패턴 매칭
    const patterns = [
      /(\d+\.?\d*)%\s*~\s*(\d+\.?\d*)%/g, // "4.5% ~ 8.9%" 형태
      /연\s*(\d+\.?\d*)%/g, // "연 5.5%" 형태
      /금리\s*(\d+\.?\d*)%/g, // "금리 6.2%" 형태
      /(\d+\.?\d*)%\s*대/g // "5%대" 형태
    ];

    for (const pattern of patterns) {
      let match;
      while ((match = pattern.exec(text)) !== null) {
        for (let i = 1; i < match.length; i++) {
          const rate = parseFloat(match[i]);
          if (rate >= 1 && rate <= 30) { // 현실적인 금리 범위
            rates.push(rate);
          }
        }
      }
    }

    return rates;
  }

  /**
   * 제공업체 이름 추출
   */
  private extractProviderName(domain: string, title: string): string {
    const providerMap: { [key: string]: string } = {
      'kb.co.kr': 'KB국민은행',
      'shinhan.com': '신한은행',
      'wooribank.com': '우리은행',
      'hanabank.com': '하나은행',
      'nhcapital.co.kr': '농협캐피탈'
    };

    for (const [domainKey, provider] of Object.entries(providerMap)) {
      if (domain.includes(domainKey)) {
        return provider;
      }
    }

    return domain;
  }

  /**
   * 금융상품 타입 결정
   */
  private determineFinanceType(text: string): string {
    if (text.includes('리스')) return '자동차 리스';
    if (text.includes('할부')) return '자동차 할부';
    if (text.includes('대출')) return '자동차 대출';
    return '자동차 금융';
  }

  /**
   * 중복 제거
   */
  private deduplicateFinancialOptions(options: FinancialOption[]): FinancialOption[] {
    const seen = new Set<string>();
    return options.filter(option => {
      const key = `${option.type}_${option.description}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }

  /**
   * 실시간 협업 필터링
   */
  async realTimeCollaborativeFiltering(userProfile: Partial<UserProfile>): Promise<Vehicle[]> {
    // 가상의 차량 데이터베이스 (실제로는 DB에서 가져옴)
    const vehicleDatabase = this.getMockVehicleDatabase();

    const filteringPrompt = `
당신은 차량 추천 AI입니다. 사용자 프로필을 기반으로 실시간 협업 필터링을 수행하여 최적의 차량 3개를 선별하세요.

사용자 프로필:
${JSON.stringify(userProfile, null, 2)}

차량 데이터베이스:
${JSON.stringify(vehicleDatabase, null, 2)}

협업 필터링 알고리즘:
1. 예산 범위 내 차량 필터링
2. 용도에 따른 차체 타입 매칭
3. 연료 타입 선호도 반영
4. 사용자 선호사항 점수화
5. 종합 매치 스코어 계산

각 차량에 대해 match_score(0-100)를 계산하고 상위 3개를 선별하세요.

응답은 반드시 다음 JSON 배열 형식으로만 답변하세요:
[
  {
    "id": "1",
    "brand": "브랜드",
    "model": "모델",
    "year": 2023,
    "price": 3000,
    "mileage": 20000,
    "fuel_type": "가솔린",
    "body_type": "세단",
    "color": "화이트",
    "location": "서울",
    "images": [],
    "features": ["기능1", "기능2"],
    "fuel_efficiency": 15.0,
    "safety_rating": 5,
    "match_score": 90,
    "description": "설명"
  }
]`;

    try {
      const result = await this.callGeminiAPI(filteringPrompt, 'filtering_agent');
      const cleanedMessage = this.extractJSONFromResponse(result.message);
      const vehicles = JSON.parse(cleanedMessage);

      // 배열인지 확인하고 Vehicle 타입 검증
      if (Array.isArray(vehicles) && vehicles.length > 0) {
        return vehicles
          .filter(v => v && v.id && v.brand && v.model)
          .sort((a: Vehicle, b: Vehicle) => (b.match_score || 0) - (a.match_score || 0))
          .slice(0, 3);
      }

      throw new Error('Invalid vehicle array received');
    } catch (error) {
      console.error('Real-time collaborative filtering failed:', error);
      // 사용자 프로필 기반 매치 스코어 계산하여 반환
      return this.getMockVehicleDatabase()
        .map(vehicle => ({
          ...vehicle,
          match_score: this.calculateMatchScore(vehicle, userProfile)
        }))
        .sort((a, b) => b.match_score - a.match_score)
        .slice(0, 3);
    }
  }

  private calculateMatchScore(vehicle: Vehicle, userProfile: Partial<UserProfile>): number {
    let score = 70; // 기본 점수

    // 예산 매칭 (30점)
    if (userProfile.budget) {
      if (vehicle.price <= userProfile.budget * 0.9) {
        score += 30;
      } else if (vehicle.price <= userProfile.budget) {
        score += 20;
      } else if (vehicle.price <= userProfile.budget * 1.1) {
        score += 10;
      }
    }

    // 연료 타입 매칭 (20점)
    if (userProfile.fuelType && vehicle.fuel_type === userProfile.fuelType) {
      score += 20;
    }

    // 용도 매칭 (15점)
    if (userProfile.purpose) {
      if (userProfile.purpose === '출퇴근용' && vehicle.fuel_efficiency >= 12) {
        score += 15;
      } else if (userProfile.purpose === '가족용' && vehicle.body_type === '세단') {
        score += 15;
      } else if (userProfile.purpose === '레저용') {
        score += 10;
      }
    }

    // 선호사항 매칭 (10점)
    if (userProfile.preferences) {
      userProfile.preferences.forEach(pref => {
        if (pref.includes('연비') && vehicle.fuel_efficiency >= 14) {
          score += 5;
        }
        if (pref.includes('안전') && vehicle.safety_rating >= 5) {
          score += 5;
        }
      });
    }

    return Math.min(Math.max(score, 0), 100); // 0-100 범위로 제한
  }

  private createVehicleExpertPrompt(userInput: string, userProfile?: Partial<UserProfile>): string {
    return `
당신은 10년 경력의 차량 추천 전문가입니다.

사용자 입력: "${userInput}"
사용자 프로필: ${JSON.stringify(userProfile || {}, null, 2)}

역할:
- 사용자 요구사항 분석
- 최적의 차량 3개 추천
- 각 차량의 장단점과 선택 이유 설명
- 실용적인 관점에서 조언 제공

응답 형식:
- 친근하고 전문적인 톤
- 구체적인 차량 모델과 가격 제시
- 사용자 라이프스타일 고려한 맞춤 추천

금융 전문가와 협업하여 종합적인 솔루션을 제공할 예정이니, 차량 자체에 집중해서 추천해주세요.`;
  }

  private createFinanceExpertPrompt(userInput: string, userProfile?: Partial<UserProfile>): string {
    return `
당신은 8년 경력의 자동차 금융 상담 전문가입니다.

사용자 입력: "${userInput}"
사용자 프로필: ${JSON.stringify(userProfile || {}, null, 2)}

역할:
- 사용자 예산 분석 및 금융 상황 평가
- 다양한 구매 옵션 제시 (현금, 대출, 리스, 할부)
- 월 납부액과 총 비용 계산
- 사용자에게 최적인 금융 방안 추천

응답 형식:
- 복잡한 금융 용어를 쉽게 설명
- 실제 숫자 기반 계산 제시
- 사용자 상황에 맞는 현실적 조언

차량 전문가가 추천한 차량들에 대해 금융 솔루션을 제공할 예정이니, 금융 측면에 집중해주세요.`;
  }

  private createCollaborationPrompt(vehicleExpertResponse: string, financeExpertResponse: string, userInput: string, userProfile?: Partial<UserProfile>): string {
    return `
두 전문가의 의견을 종합하여 최종 추천안을 작성하세요.

차량 전문가 의견:
${vehicleExpertResponse}

금융 전문가 의견:
${financeExpertResponse}

사용자 원본 요청: "${userInput}"
사용자 프로필: ${JSON.stringify(userProfile || {}, null, 2)}

협업 지침:
1. 두 전문가의 의견을 통합하여 최적화된 솔루션 도출
2. 차량 추천과 금융 옵션을 매칭하여 구체적 제안
3. 사용자가 바로 결정할 수 있는 actionable한 추천
4. 예상되는 질문에 대한 선제적 답변 포함

최종 응답은 다음을 포함해야 합니다:
- 추천 차량 3개 (구체적 모델명, 가격)
- 각 차량별 최적 금융 옵션
- 종합 추천 순위와 이유
- 다음 단계 안내`;
  }

  private async callGeminiAPI(prompt: string, agent: string): Promise<AgentResponse> {
    try {
      const response = await fetch(this.apiUrl + '?key=' + this.apiKey, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          contents: [{
            parts: [{
              text: prompt
            }]
          }],
          generationConfig: {
            temperature: 0.7,
            topK: 40,
            topP: 0.95,
            maxOutputTokens: 2048,
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Gemini API error: ${response.status}`);
      }

      const data = await response.json();
      const content = data.candidates?.[0]?.content?.parts?.[0]?.text || '';

      return {
        agent: agent as 'vehicle_expert' | 'finance_expert',
        message: content,
        confidence: 0.85,
        next_questions: []
      };

    } catch (error) {
      console.error(`Gemini API call failed for ${agent}:`, error);
      throw error;
    }
  }

  private processCollaborativeResult(
    vehicleResponse: AgentResponse,
    financeResponse: AgentResponse,
    finalResult: AgentResponse,
    userProfile?: Partial<UserProfile>
  ): CollaborativeResult {
    // 실제로는 응답을 파싱하여 구조화된 데이터로 변환
    // 여기서는 mock 데이터 반환
    return {
      vehicles: this.getMockVehicleDatabase().slice(0, 3),
      financial_options: this.getMockFinancialOptions(),
      recommendations: [
        "현대 아반떼 - 합리적인 첫차로 최적",
        "기아 K5 - 가족용으로 적합한 중형차",
        "제네시스 G70 - 프리미엄 선택지"
      ],
      agent_discussion: [vehicleResponse, financeResponse, finalResult]
    };
  }

  private getMockVehicleDatabase(): Vehicle[] {
    return [
      {
        id: "1",
        brand: "현대",
        model: "아반떼",
        year: 2023,
        price: 2800,
        mileage: 15000,
        fuel_type: "가솔린",
        body_type: "세단",
        color: "화이트",
        location: "서울",
        images: [],
        features: ["스마트크루즈", "후방카메라", "블루투스"],
        fuel_efficiency: 14.5,
        safety_rating: 5,
        match_score: 92,
        description: "경제적이고 실용적인 첫차 추천"
      },
      {
        id: "2",
        brand: "기아",
        model: "K5",
        year: 2022,
        price: 3500,
        mileage: 25000,
        fuel_type: "하이브리드",
        body_type: "세단",
        color: "블랙",
        location: "인천",
        images: [],
        features: ["하이브리드", "선루프", "통풍시트"],
        fuel_efficiency: 16.8,
        safety_rating: 5,
        match_score: 88,
        description: "연비 좋은 하이브리드 중형차"
      },
      {
        id: "3",
        brand: "제네시스",
        model: "G70",
        year: 2023,
        price: 4200,
        mileage: 8000,
        fuel_type: "가솔린",
        body_type: "세단",
        color: "그레이",
        location: "경기",
        images: [],
        features: ["가죽시트", "프리미엄사운드", "어댑티브크루즈"],
        fuel_efficiency: 11.5,
        safety_rating: 5,
        match_score: 84,
        description: "프리미엄 브랜드의 럭셔리 세단"
      }
    ];
  }

  private getMockFinancialOptions(): FinancialOption[] {
    return [
      {
        type: "현금 일시불",
        monthly_payment: 0,
        total_cost: 2800,
        down_payment: 2800,
        description: "이자 부담 없는 일시불 구매",
        pros: ["이자 없음", "즉시 소유권"],
        cons: ["높은 초기 부담"]
      },
      {
        type: "은행 대출",
        monthly_payment: 52,
        total_cost: 3120,
        down_payment: 500,
        description: "5년 은행 대출 (연 4.5%)",
        pros: ["낮은 금리", "장기 분납"],
        cons: ["신용심사 필요"]
      },
      {
        type: "할부",
        monthly_payment: 58,
        total_cost: 3480,
        down_payment: 0,
        description: "5년 할부 (연 6.5%)",
        pros: ["초기 부담 적음", "심사 간편"],
        cons: ["높은 총 비용"]
      }
    ];
  }
  /**
   * 차량 전문가 AI - 개별 차량 분석 및 추천
   */
  async vehicleExpertAgent(userInput: string, userProfile?: Partial<UserProfile>): Promise<{
    response: string;
    agent: 'vehicle_expert';
    recommendations: string[];
  }> {
    const prompt = `
당신은 10년 경력의 차량 전문가입니다. 사용자의 요구사항을 분석하여 최적의 차량을 추천하세요.

사용자 입력: "${userInput}"
사용자 프로필: ${JSON.stringify(userProfile || {}, null, 2)}

전문 분야:
- 차량 성능 분석
- 브랜드별 특징 설명
- 실용성 평가
- 시장가 분석

응답 스타일:
- 기술적 전문성
- 구체적인 모델명과 수치
- 장단점 명확 제시
- 실용적 조언

응답은 반드시 다음 JSON 형식으로만 답변하세요:
{
  "response": "전문가 분석 메시지",
  "agent": "vehicle_expert",
  "recommendations": ["추천1", "추천2", "추천3"]
}`;

    try {
      const result = await this.callGeminiAPI(prompt, 'vehicle_expert');
      return JSON.parse(this.extractJSONFromResponse(result.message));
    } catch (error) {
      return {
        response: "차량 전문가입니다. 사용자님의 조건을 분석하여 최적의 차량을 추천드리겠습니다.",
        agent: 'vehicle_expert',
        recommendations: ["현대 아반떼", "기아 K5", "제네시스 G70"]
      };
    }
  }

  /**
   * 금융 전문가 AI - 금융상품 분석 및 상담
   */
  async financeExpertAgent(userInput: string, userProfile?: Partial<UserProfile>): Promise<{
    response: string;
    agent: 'finance_expert';
    financial_options: string[];
  }> {
    const prompt = `
당신은 8년 경력의 자동차 금융 전문가입니다. 사용자의 경제 상황을 고려한 최적의 금융 솔루션을 제공하세요.

사용자 입력: "${userInput}"
사용자 프로필: ${JSON.stringify(userProfile || {}, null, 2)}

전문 분야:
- 대출/할부/리스 분석
- 금리 비교 및 설명
- 월 납입금 계산
- 총 비용 분석

응답 스타일:
- 숫자 기반 명확한 설명
- 복잡한 금융 용어 쉽게 설명
- 사용자 상황 맞춤 조언
- 위험요소 사전 고지

${this.googleSearchApiKey ? `
Google Search API를 활용하여 실시간 금융상품 정보도 검색하여 최신 금리와 조건을 반영하세요.
` : ''}

응답은 반드시 다음 JSON 형식으로만 답변하세요:
{
  "response": "금융 전문가 상담 메시지",
  "agent": "finance_expert",
  "financial_options": ["옵션1", "옵션2", "옵션3"]
}`;

    try {
      // Google Search API 사용 시 실시간 금융상품 검색
      if (this.googleSearchApiKey && this.customSearchEngineId && userProfile?.budget) {
        const searchResults = await this.searchFinancialProducts({
          type: 'loan',
          vehicle_price: userProfile.budget,
          credit_score: 'good'
        });

        // 검색 결과를 프롬프트에 추가
        const enhancedPrompt = prompt + `

최신 실시간 금융상품 정보:
${JSON.stringify(searchResults, null, 2)}

위 실시간 정보를 활용하여 더욱 정확하고 최신의 금융 상담을 제공하세요.`;

        const result = await this.callGeminiAPI(enhancedPrompt, 'finance_expert');
        return JSON.parse(this.extractJSONFromResponse(result.message));
      }

      const result = await this.callGeminiAPI(prompt, 'finance_expert');
      return JSON.parse(this.extractJSONFromResponse(result.message));
    } catch (error) {
      return {
        response: "금융 전문가입니다. 사용자님의 예산과 상황에 맞는 최적의 금융 솔루션을 찾아드리겠습니다.",
        agent: 'finance_expert',
        financial_options: ["현금구매", "은행대출", "할부금융"]
      };
    }
  }
}

export { GeminiMultiAgent, type UserProfile, type Vehicle, type CollaborativeResult };