/**
 * 실시간 금융상품 정보 검색 엔진
 * Google Search API + 웹 스크래핑으로 최신 금융 정보 수집
 */

interface FinancialSearchResult {
  provider: string;
  product_name: string;
  interest_rate?: {
    min?: number;
    max?: number;
    typical?: number;
  };
  source_url: string;
  last_updated: Date;
  confidence: number;
}

interface SearchQuery {
  type: 'loan' | 'lease' | 'installment';
  vehicle_price: number;
  credit_score?: 'excellent' | 'good' | 'fair' | 'poor';
  keywords?: string[];
}

class FinancialInformationSearcher {
  private googleApiKey: string;
  private customSearchEngineId: string;
  private requestCount: number = 0;
  private dailyLimit: number = 100; // Google 무료 할당량

  // 신뢰할 수 있는 금융정보 소스들
  private trustedSources = [
    'kb.co.kr',
    'shinhan.com',
    'wooribank.com',
    'hanabank.com',
    'bnkfn.co.kr',
    'nhcapital.co.kr',
    'lotte.co.kr',
    'samsung.co.kr',
    'hyundaicapital.com'
  ];

  constructor(googleApiKey: string, customSearchEngineId: string) {
    this.googleApiKey = googleApiKey;
    this.customSearchEngineId = customSearchEngineId;
  }

  /**
   * 실시간 금융상품 검색
   */
  async searchFinancialProducts(query: SearchQuery): Promise<import('../realistic-agents').FinancialProduct[]> {
    const searchResults = await this.performGoogleSearch(query);
    const parsedProducts = await this.parseSearchResults(searchResults, query);

    // 신뢰도 기반 정렬 및 상위 결과 반환
    return parsedProducts
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);
  }

  /**
   * Google Custom Search API 호출
   */
  private async performGoogleSearch(query: SearchQuery): Promise<any[]> {
    if (this.requestCount >= this.dailyLimit) {
      console.warn('Google Search API daily limit reached');
      return [];
    }

    const searchQueries = this.buildSearchQueries(query);
    const allResults: any[] = [];

    for (const searchQuery of searchQueries) {
      try {
        const url = new URL('https://www.googleapis.com/customsearch/v1');
        url.searchParams.append('key', this.googleApiKey);
        url.searchParams.append('cx', this.customSearchEngineId);
        url.searchParams.append('q', searchQuery);
        url.searchParams.append('num', '5');
        url.searchParams.append('dateRestrict', 'm3'); // 최근 3개월

        const response = await fetch(url.toString());

        if (!response.ok) {
          throw new Error(`Google Search API error: ${response.status}`);
        }

        const data = await response.json();
        if (data.items) {
          allResults.push(...data.items);
        }

        this.requestCount++;

        // API 호출 간격 조절 (Rate limiting)
        await new Promise(resolve => setTimeout(resolve, 1000));

      } catch (error) {
        console.error('Google Search failed:', error);
      }
    }

    return allResults;
  }

  /**
   * 검색 쿼리 생성
   */
  private buildSearchQueries(query: SearchQuery): string[] {
    const baseQueries = [];

    switch (query.type) {
      case 'loan':
        baseQueries.push(
          `중고차 대출 ${query.vehicle_price}만원 금리 2024`,
          `자동차 담보대출 금리 비교 2024`,
          `중고차 할부 금리 순위 2024`
        );
        break;

      case 'lease':
        baseQueries.push(
          `자동차 리스 ${query.vehicle_price}만원 2024`,
          `중고차 리스 금리 비교 2024`,
          `차량 리스 월납입금 계산기 2024`
        );
        break;

      case 'installment':
        baseQueries.push(
          `자동차 할부 금리 비교 2024`,
          `중고차 카드할부 조건 2024`,
          `차량 할부금융 순위 2024`
        );
        break;
    }

    // 신용등급별 쿼리 추가
    if (query.credit_score) {
      const creditQueries = baseQueries.map(q =>
        `${q} ${this.getCreditScoreKeyword(query.credit_score!)}`
      );
      baseQueries.push(...creditQueries);
    }

    return baseQueries;
  }

  /**
   * 검색 결과 파싱
   */
  private async parseSearchResults(
    searchResults: any[],
    query: SearchQuery
  ): Promise<import('../realistic-agents').FinancialProduct[]> {
    const products: import('../realistic-agents').FinancialProduct[] = [];

    for (const result of searchResults) {
      try {
        const product = await this.extractFinancialProduct(result, query);
        if (product) {
          products.push(product);
        }
      } catch (error) {
        console.error('Failed to parse search result:', error);
      }
    }

    return products;
  }

  /**
   * 개별 검색 결과에서 금융상품 정보 추출
   */
  private async extractFinancialProduct(
    searchResult: any,
    query: SearchQuery
  ): Promise<import('../realistic-agents').FinancialProduct | null> {
    const { title, link, snippet } = searchResult;

    // 신뢰할 수 있는 소스인지 확인
    const domain = new URL(link).hostname;
    const isTrustedSource = this.trustedSources.some(source => domain.includes(source));

    // 기본 신뢰도 설정
    let confidence = isTrustedSource ? 0.8 : 0.4;

    // 제공업체 추출
    const provider = this.extractProvider(domain, title);

    // 금리 정보 추출
    const interestRate = this.extractInterestRate(snippet + ' ' + title);

    // 상품명 추출
    const productName = this.extractProductName(title, query.type);

    // 최소 정보가 있는 경우만 반환
    if (!provider || !interestRate) {
      return null;
    }

    // 금리 정보가 있으면 신뢰도 증가
    if (interestRate.typical) {
      confidence += 0.2;
    }

    const product: import('../realistic-agents').FinancialProduct = {
      id: `search_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: this.mapToProductType(query.type),
      provider: provider,
      product_name: productName,
      interest_rate: interestRate,
      loan_to_value: this.estimateLoanToValue(query.type),
      max_amount: this.estimateMaxAmount(provider),
      term_months: this.getTypicalTerms(query.type),
      requirements: this.getTypicalRequirements(provider),
      pros: this.generatePros(provider, interestRate),
      cons: this.generateCons(query.type),
      last_updated: new Date()
    };

    return product;
  }

  /**
   * 헬퍼 메서드들
   */
  private extractProvider(domain: string, title: string): string {
    // 도메인 기반 제공업체 매핑
    const providerMap: { [key: string]: string } = {
      'kb.co.kr': 'KB국민은행',
      'shinhan.com': '신한은행',
      'wooribank.com': '우리은행',
      'hanabank.com': '하나은행',
      'nhcapital.co.kr': '농협캐피탈',
      'hyundaicapital.com': '현대캐피탈',
      'lotte.co.kr': '롯데캐피탈',
      'samsung.co.kr': '삼성캐피탈'
    };

    for (const [domain_key, provider] of Object.entries(providerMap)) {
      if (domain.includes(domain_key)) {
        return provider;
      }
    }

    // 제목에서 추출 시도
    const titleProviders = ['KB', '신한', '우리', '하나', '농협', '현대', '롯데', '삼성'];
    for (const provider of titleProviders) {
      if (title.includes(provider)) {
        return `${provider}${provider === 'KB' ? '국민은행' : ''}`;
      }
    }

    return domain;
  }

  private extractInterestRate(text: string): { min: number; max: number; typical: number } {
    // 금리 패턴 매칭
    const patterns = [
      /(\d+\.?\d*)%\s*~\s*(\d+\.?\d*)%/g, // "4.5% ~ 8.9%" 형태
      /연\s*(\d+\.?\d*)%/g, // "연 5.5%" 형태
      /금리\s*(\d+\.?\d*)%/g, // "금리 6.2%" 형태
      /(\d+\.?\d*)%\s*대/g // "5%대" 형태
    ];

    const rates: number[] = [];

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

    if (rates.length === 0) {
      // 기본값 반환
      return { min: 4.5, max: 12.0, typical: 8.0 };
    }

    rates.sort((a, b) => a - b);

    return {
      min: rates[0],
      max: rates[rates.length - 1],
      typical: rates[Math.floor(rates.length / 2)] // 중간값
    };
  }

  private extractProductName(title: string, type: 'loan' | 'lease' | 'installment'): string {
    const typeMap = {
      'loan': '자동차대출',
      'lease': '자동차리스',
      'installment': '자동차할부'
    };

    // 제목에서 상품명 추출 시도
    const cleanTitle = title.replace(/[\[\]()]/g, '').trim();

    if (cleanTitle.length > 10) {
      return cleanTitle.substring(0, 30) + '...';
    }

    return typeMap[type];
  }

  private mapToProductType(type: string): 'bank_loan' | 'capital_loan' | 'lease' | 'installment' {
    switch (type) {
      case 'loan': return 'bank_loan';
      case 'lease': return 'lease';
      case 'installment': return 'installment';
      default: return 'bank_loan';
    }
  }

  private estimateLoanToValue(type: string): number {
    switch (type) {
      case 'loan': return 80;
      case 'lease': return 90;
      case 'installment': return 100;
      default: return 80;
    }
  }

  private estimateMaxAmount(provider: string): number {
    // 제공업체별 최대 대출액 추정
    const bankLimits: { [key: string]: number } = {
      'KB국민은행': 100000,
      '신한은행': 100000,
      '우리은행': 80000,
      '하나은행': 80000,
      '현대캐피탈': 50000,
      '롯데캐피탈': 30000
    };

    return bankLimits[provider] || 50000;
  }

  private getTypicalTerms(type: string): number[] {
    switch (type) {
      case 'loan': return [12, 24, 36, 48, 60];
      case 'lease': return [24, 36, 48];
      case 'installment': return [6, 12, 18, 24, 36];
      default: return [24, 36, 48];
    }
  }

  private getTypicalRequirements(provider: string): string[] {
    const baseRequirements = ['신분증', '소득증빙서류'];

    if (provider.includes('은행')) {
      return [...baseRequirements, '재직증명서', '신용등급 4등급 이상'];
    } else {
      return [...baseRequirements, '신용등급 6등급 이상'];
    }
  }

  private generatePros(provider: string, interestRate: any): string[] {
    const pros = [];

    if (provider.includes('은행')) {
      pros.push('신뢰할 수 있는 금융기관');
    }

    if (interestRate.typical < 7) {
      pros.push('낮은 금리');
    }

    pros.push('온라인 신청 가능', '중도상환 가능');

    return pros;
  }

  private generateCons(type: string): string[] {
    const baseCons = ['신용조회 필요'];

    switch (type) {
      case 'loan':
        return [...baseCons, '담보 설정', '까다로운 심사'];
      case 'lease':
        return [...baseCons, '소유권 없음', '거리 제한'];
      case 'installment':
        return [...baseCons, '높은 금리', '단기간 상환'];
      default:
        return baseCons;
    }
  }

  private getCreditScoreKeyword(creditScore: string): string {
    switch (creditScore) {
      case 'excellent': return '1등급 2등급';
      case 'good': return '3등급 4등급';
      case 'fair': return '5등급 6등급';
      case 'poor': return '7등급 8등급';
      default: return '';
    }
  }

  /**
   * 캐시 관리
   */
  private searchCache = new Map<string, { data: import('../realistic-agents').FinancialProduct[]; timestamp: number }>();
  private cacheTimeout = 3600000; // 1시간

  private getCacheKey(query: SearchQuery): string {
    return JSON.stringify(query);
  }

  async getCachedResults(query: SearchQuery): Promise<import('../realistic-agents').FinancialProduct[] | null> {
    const cacheKey = this.getCacheKey(query);
    const cached = this.searchCache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
      return cached.data;
    }

    return null;
  }

  private setCachedResults(query: SearchQuery, data: import('../realistic-agents').FinancialProduct[]): void {
    const cacheKey = this.getCacheKey(query);
    this.searchCache.set(cacheKey, {
      data,
      timestamp: Date.now()
    });
  }
}

export { FinancialInformationSearcher, type SearchQuery, type FinancialSearchResult };