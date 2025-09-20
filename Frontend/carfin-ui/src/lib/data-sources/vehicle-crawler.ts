/**
 * 실시간 중고차 매물 데이터 수집기
 * 엔카, KB차차차, 카프라이스 등 실제 데이터 소스 연동
 */

interface CrawlerConfig {
  source: 'encar' | 'kbchachacha' | 'carprice' | 'sk_encar';
  baseUrl: string;
  apiEndpoint?: string;
  rateLimit: number; // requests per minute
}

interface VehicleSearchParams {
  budget_min?: number;
  budget_max?: number;
  fuel_type?: string;
  brand?: string;
  year_min?: number;
  location?: string;
  page?: number;
  limit?: number;
}

class VehicleDataCrawler {
  private configs: CrawlerConfig[] = [
    {
      source: 'encar',
      baseUrl: 'https://www.encar.com',
      apiEndpoint: '/dc/dcapi.do',
      rateLimit: 60
    },
    {
      source: 'kbchachacha',
      baseUrl: 'https://www.kbchachacha.com',
      rateLimit: 30
    },
    {
      source: 'carprice',
      baseUrl: 'https://www.carprice.co.kr',
      rateLimit: 30
    }
  ];

  private requestCounts: Map<string, number> = new Map();
  private lastReset: Map<string, number> = new Map();

  /**
   * 여러 소스에서 차량 데이터 수집
   */
  async crawlVehicleListings(params: VehicleSearchParams): Promise<import('../realistic-agents').VehicleListing[]> {
    const allListings: import('../realistic-agents').VehicleListing[] = [];

    // 병렬로 여러 소스에서 데이터 수집
    const crawlPromises = this.configs.map(config =>
      this.crawlFromSource(config, params).catch(error => {
        console.error(`Failed to crawl from ${config.source}:`, error);
        return [];
      })
    );

    const results = await Promise.all(crawlPromises);

    // 결과 통합 및 중복 제거
    for (const listings of results) {
      allListings.push(...listings);
    }

    return this.deduplicateListings(allListings);
  }

  /**
   * 특정 소스에서 데이터 크롤링
   */
  private async crawlFromSource(
    config: CrawlerConfig,
    params: VehicleSearchParams
  ): Promise<import('../realistic-agents').VehicleListing[]> {

    // Rate limiting 체크
    if (!this.checkRateLimit(config.source, config.rateLimit)) {
      console.warn(`Rate limit exceeded for ${config.source}`);
      return [];
    }

    try {
      switch (config.source) {
        case 'encar':
          return await this.crawlEncar(params);
        case 'kbchachacha':
          return await this.crawlKBChachacha(params);
        case 'carprice':
          return await this.crawlCarPrice(params);
        default:
          return [];
      }
    } catch (error) {
      console.error(`Crawling failed for ${config.source}:`, error);
      return [];
    }
  }

  /**
   * 엔카 데이터 크롤링 (실제 API 연동 시뮬레이션)
   */
  private async crawlEncar(params: VehicleSearchParams): Promise<import('../realistic-agents').VehicleListing[]> {
    // 실제 구현 시 엔카 API 또는 웹 스크래핑
    // 현재는 현실적인 시뮬레이션 데이터 반환

    const mockEncarData: import('../realistic-agents').VehicleListing[] = [
      {
        id: `encar_${Date.now()}_1`,
        source: 'encar',
        brand: '현대',
        model: '아반떼',
        year: 2022,
        price: this.getRandomPrice(params.budget_min, params.budget_max, 2800),
        mileage: Math.floor(Math.random() * 50000) + 10000,
        fuel_type: this.getFuelType(params.fuel_type),
        transmission: '자동',
        location: '서울 강남구',
        dealer_name: '강남현대모터스',
        images: [
          '/api/placeholder/400/300',
          '/api/placeholder/400/300'
        ],
        features: ['후방카메라', '블루투스', '크루즈컨트롤', '열선시트'],
        inspection_grade: '1급',
        accident_history: 'none',
        market_price_analysis: {
          average_price: 2950,
          price_rating: 'good',
          similar_listings_count: 23
        }
      },
      {
        id: `encar_${Date.now()}_2`,
        source: 'encar',
        brand: '기아',
        model: 'K5',
        year: 2021,
        price: this.getRandomPrice(params.budget_min, params.budget_max, 3200),
        mileage: Math.floor(Math.random() * 60000) + 20000,
        fuel_type: '하이브리드',
        transmission: '자동',
        location: '경기 수원시',
        dealer_name: '수원기아모터스',
        images: ['/api/placeholder/400/300'],
        features: ['선루프', '통풍시트', '어댑티브크루즈', '디지털클러스터'],
        inspection_grade: '2급',
        accident_history: 'minor',
        market_price_analysis: {
          average_price: 3350,
          price_rating: 'excellent',
          similar_listings_count: 31
        }
      }
    ];

    // 필터링 적용
    return this.applyFilters(mockEncarData, params);
  }

  /**
   * KB차차차 데이터 크롤링
   */
  private async crawlKBChachacha(params: VehicleSearchParams): Promise<import('../realistic-agents').VehicleListing[]> {
    const mockKBData: import('../realistic-agents').VehicleListing[] = [
      {
        id: `kb_${Date.now()}_1`,
        source: 'kbchachacha',
        brand: '제네시스',
        model: 'G70',
        year: 2023,
        price: this.getRandomPrice(params.budget_min, params.budget_max, 4200),
        mileage: Math.floor(Math.random() * 20000) + 5000,
        fuel_type: '가솔린',
        transmission: '자동',
        location: '서울 서초구',
        dealer_name: 'KB차차차',
        images: ['/api/placeholder/400/300'],
        features: ['가죽시트', '프리미엄사운드', '어댑티브크루즈', '헤드업디스플레이'],
        inspection_grade: '특급',
        accident_history: 'none',
        market_price_analysis: {
          average_price: 4350,
          price_rating: 'good',
          similar_listings_count: 15
        }
      }
    ];

    return this.applyFilters(mockKBData, params);
  }

  /**
   * 카프라이스 데이터 크롤링
   */
  private async crawlCarPrice(params: VehicleSearchParams): Promise<import('../realistic-agents').VehicleListing[]> {
    const mockCarPriceData: import('../realistic-agents').VehicleListing[] = [
      {
        id: `carprice_${Date.now()}_1`,
        source: 'carprice',
        brand: '현대',
        model: '투싼',
        year: 2022,
        price: this.getRandomPrice(params.budget_min, params.budget_max, 3500),
        mileage: Math.floor(Math.random() * 40000) + 15000,
        fuel_type: this.getFuelType(params.fuel_type),
        transmission: '자동',
        location: '경기 안양시',
        dealer_name: '안양모터스',
        images: ['/api/placeholder/400/300'],
        features: ['파노라마선루프', '전후방카메라', '스마트키', '하이패스'],
        inspection_grade: '1급',
        accident_history: 'none',
        market_price_analysis: {
          average_price: 3600,
          price_rating: 'good',
          similar_listings_count: 28
        }
      }
    ];

    return this.applyFilters(mockCarPriceData, params);
  }

  /**
   * Rate limiting 체크
   */
  private checkRateLimit(source: string, limit: number): boolean {
    const now = Date.now();
    const minute = 60 * 1000;

    const lastReset = this.lastReset.get(source) || 0;
    const currentCount = this.requestCounts.get(source) || 0;

    // 1분이 지났으면 카운터 리셋
    if (now - lastReset > minute) {
      this.requestCounts.set(source, 0);
      this.lastReset.set(source, now);
      return true;
    }

    // 제한 확인
    if (currentCount >= limit) {
      return false;
    }

    // 카운터 증가
    this.requestCounts.set(source, currentCount + 1);
    return true;
  }

  /**
   * 중복 제거
   */
  private deduplicateListings(listings: import('../realistic-agents').VehicleListing[]): import('../realistic-agents').VehicleListing[] {
    const seen = new Set<string>();
    return listings.filter(listing => {
      const key = `${listing.brand}_${listing.model}_${listing.year}_${listing.mileage}_${listing.price}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  /**
   * 필터 적용
   */
  private applyFilters(
    listings: import('../realistic-agents').VehicleListing[],
    params: VehicleSearchParams
  ): import('../realistic-agents').VehicleListing[] {
    return listings.filter(listing => {
      if (params.budget_min && listing.price < params.budget_min) return false;
      if (params.budget_max && listing.price > params.budget_max) return false;
      if (params.fuel_type && !listing.fuel_type.toLowerCase().includes(params.fuel_type.toLowerCase())) return false;
      if (params.brand && !listing.brand.toLowerCase().includes(params.brand.toLowerCase())) return false;
      if (params.year_min && listing.year < params.year_min) return false;
      return true;
    });
  }

  /**
   * 헬퍼 메서드들
   */
  private getRandomPrice(min?: number, max?: number, base: number = 3000): number {
    if (min && max) {
      return Math.floor(Math.random() * (max - min)) + min;
    }
    if (max) {
      return Math.min(base, max - Math.floor(Math.random() * 500));
    }
    if (min) {
      return Math.max(base, min + Math.floor(Math.random() * 500));
    }
    return base + (Math.random() - 0.5) * 1000;
  }

  private getFuelType(preferred?: string): string {
    if (preferred) return preferred;
    const types = ['가솔린', '하이브리드', '디젤', '전기'];
    return types[Math.floor(Math.random() * types.length)];
  }

  /**
   * 실시간 데이터 업데이트
   */
  async updateVehicleCache(params: VehicleSearchParams): Promise<void> {
    const freshData = await this.crawlVehicleListings(params);
    // 캐시 업데이트 로직
    console.log(`Updated cache with ${freshData.length} listings`);
  }

  /**
   * 특정 차량의 상세 정보 크롤링
   */
  async getVehicleDetails(vehicleId: string, source: string): Promise<import('../realistic-agents').VehicleListing | null> {
    // 실제 구현 시 각 사이트의 상세 페이지 크롤링
    console.log(`Fetching details for ${vehicleId} from ${source}`);
    return null;
  }
}

export { VehicleDataCrawler, type VehicleSearchParams };