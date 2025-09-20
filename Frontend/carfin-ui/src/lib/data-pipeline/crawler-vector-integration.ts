// 크롤러 데이터와 벡터 임베딩 통합 파이프라인
import { ProcessedCarData, GeminiRequest } from '@/types';
import { VectorDatabaseManager, CarVectorData } from '@/lib/data-store/vector-database';
import { OptimizedGeminiClient } from '@/lib/gemini/gemini-client';

export interface CrawlerDataInput {
  // 크롤러팀에서 제공할 원시 데이터 구조
  raw_data: {
    source: 'encar' | 'kbcha' | 'autoplaza' | 'others';
    url: string;
    scraped_at: number;
    data: {
      brand: string;
      model: string;
      year: number;
      price: number;
      mileage: number;
      fuel_type: string;
      transmission: string;
      body_type: string;
      color?: string;
      images?: string[];
      description?: string;
      dealer_info?: {
        name: string;
        rating?: number;
        location?: string;
        contact?: string;
      };
      features?: string[];
      additional_specs?: Record<string, any>;
    };
  };

  // 데이터 품질 메타데이터
  quality_metrics: {
    completeness_score: number; // 0-1, 필수 필드 완성도
    reliability_score: number;  // 0-1, 소스 신뢰도
    freshness_score: number;    // 0-1, 데이터 신선도
    duplicate_probability: number; // 0-1, 중복 가능성
  };
}

export class CrawlerVectorIntegrationPipeline {
  private vectorDB: VectorDatabaseManager;
  private geminiClient: OptimizedGeminiClient;
  private processingQueue: CrawlerDataInput[] = [];
  private batchSize = 20; // 배치 처리 크기
  private processingInterval = 30000; // 30초마다 배치 처리

  constructor(vectorDB: VectorDatabaseManager, geminiClient: OptimizedGeminiClient) {
    this.vectorDB = vectorDB;
    this.geminiClient = geminiClient;
    this.startBatchProcessing();
  }

  // 크롤러 데이터 수신 및 큐에 추가
  async ingestCrawlerData(data: CrawlerDataInput): Promise<void> {
    // 데이터 품질 검증
    if (!this.validateDataQuality(data)) {
      console.warn('Data quality check failed, skipping:', data.raw_data.url);
      return;
    }

    this.processingQueue.push(data);

    // 긴급 처리가 필요한 고품질 데이터는 즉시 처리
    if (data.quality_metrics.reliability_score > 0.9 &&
        data.quality_metrics.freshness_score > 0.8) {
      await this.processDataImmediate(data);
    }
  }

  // 배치 처리 시작
  private startBatchProcessing(): void {
    setInterval(async () => {
      if (this.processingQueue.length > 0) {
        await this.processBatch();
      }
    }, this.processingInterval);
  }

  // 배치 처리 실행
  private async processBatch(): Promise<void> {
    const batch = this.processingQueue.splice(0, this.batchSize);

    console.log(`Processing batch of ${batch.length} items`);

    // 병렬 처리로 성능 최적화
    const processedResults = await Promise.allSettled(
      batch.map(data => this.processDataItem(data))
    );

    // 실패한 항목 재시도 큐에 추가
    processedResults.forEach((result, index) => {
      if (result.status === 'rejected') {
        console.error('Failed to process item:', result.reason);
        // 재시도 로직 추가 가능
      }
    });
  }

  // 개별 데이터 아이템 처리
  private async processDataItem(crawlerData: CrawlerDataInput): Promise<ProcessedCarData> {
    // 1. 원시 데이터를 표준화된 형태로 변환
    const normalizedData = await this.normalizeRawData(crawlerData);

    // 2. Gemini AI를 통한 의미적 분석
    const semanticFeatures = await this.extractSemanticFeatures(normalizedData, crawlerData);

    // 3. 시장 데이터와 결합
    const marketFeatures = await this.enrichWithMarketData(normalizedData);

    // 4. 최종 ProcessedCarData 생성
    const processedData: ProcessedCarData = {
      ...normalizedData,
      semantic_features: semanticFeatures,
      normalized_features: {
        age: new Date().getFullYear() - normalizedData.year,
        mileage_per_year: normalizedData.mileage / (new Date().getFullYear() - normalizedData.year),
        price_per_year: normalizedData.price / (new Date().getFullYear() - normalizedData.year),
        fuel_efficiency: await this.estimateFuelEfficiency(normalizedData),
        depreciation_rate: await this.calculateDepreciationRate(normalizedData),
        market_position: this.determineMarketPosition(normalizedData.price)
      },
      similarity_vector: await this.generateSimilarityVector(normalizedData, semanticFeatures)
    };

    // 5. 벡터 DB에 저장
    await this.vectorDB.storeCarVector(processedData);

    return processedData;
  }

  // 즉시 처리 (고품질 데이터용)
  private async processDataImmediate(data: CrawlerDataInput): Promise<void> {
    try {
      await this.processDataItem(data);
      // 큐에서 제거
      const index = this.processingQueue.indexOf(data);
      if (index > -1) {
        this.processingQueue.splice(index, 1);
      }
    } catch (error) {
      console.error('Immediate processing failed:', error);
    }
  }

  // 원시 데이터 정규화
  private async normalizeRawData(crawlerData: CrawlerDataInput): Promise<ProcessedCarData> {
    const raw = crawlerData.raw_data.data;

    return {
      id: this.generateCarId(crawlerData),
      brand: this.normalizeBrand(raw.brand),
      model: this.normalizeModel(raw.model),
      year: raw.year,
      price: raw.price,
      mileage: raw.mileage,
      fuelType: this.normalizeFuelType(raw.fuel_type),
      transmission: raw.transmission,
      bodyType: this.normalizeBodyType(raw.body_type),
      images: raw.images || [],
      features: raw.features || [],
      location: raw.dealer_info?.location,
      dealer: raw.dealer_info?.name,
      sourceUrl: crawlerData.raw_data.url,
      lastUpdated: crawlerData.raw_data.scraped_at
    };
  }

  // Gemini AI를 통한 의미적 특성 추출
  private async extractSemanticFeatures(
    carData: ProcessedCarData,
    crawlerData: CrawlerDataInput
  ): Promise<any> {

    const prompt = this.buildSemanticAnalysisPrompt(carData, crawlerData);

    try {
      const response = await this.geminiClient.generateContent({
        agentType: 'DATA_EXTRACTOR' as any,
        prompt,
        temperature: 0.1,
        maxTokens: 800,
        model: 'gemini-1.5-flash'
      });

      return this.parseSemanticResponse(response.content);

    } catch (error) {
      console.error('Semantic analysis failed:', error);
      return this.getDefaultSemanticFeatures();
    }
  }

  // Gemini 프롬프트 생성
  private buildSemanticAnalysisPrompt(
    carData: ProcessedCarData,
    crawlerData: CrawlerDataInput
  ): string {
    const raw = crawlerData.raw_data.data;

    return `
차량 데이터를 분석하여 의미적 특성을 추출해주세요:

차량 정보:
- 브랜드/모델: ${carData.brand} ${carData.model}
- 연식: ${carData.year}년
- 가격: ${carData.price.toLocaleString()}만원
- 주행거리: ${carData.mileage.toLocaleString()}km
- 연료: ${carData.fuelType}
- 변속기: ${carData.transmission}
- 차종: ${carData.bodyType}
- 특징: ${raw.features?.join(', ') || '없음'}
- 설명: ${raw.description || '없음'}

다음 JSON 형태로 분석 결과를 제공해주세요:
{
  "target_demographic": "young_professional|family|luxury_seeker|economy_focused|senior",
  "usage_pattern": "city_driving|highway_cruising|family_transport|business|recreation",
  "style_category": "sporty|elegant|practical|rugged|minimalist",
  "reliability_score": 1-10,
  "maintenance_cost_level": "low|medium|high",
  "resale_value_potential": "excellent|good|fair|poor",
  "lifestyle_match": ["commuting", "family_trips", "weekend_adventures", "business_travel"],
  "emotional_appeal": {
    "prestige": 1-10,
    "sportiness": 1-10,
    "practicality": 1-10,
    "eco_friendliness": 1-10
  }
}`;
  }

  // Gemini 응답 파싱
  private parseSemanticResponse(responseContent: string): any {
    try {
      // JSON 추출 정규식
      const jsonMatch = responseContent.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        return JSON.parse(jsonMatch[0]);
      }

      return this.getDefaultSemanticFeatures();
    } catch (error) {
      console.error('Failed to parse semantic response:', error);
      return this.getDefaultSemanticFeatures();
    }
  }

  // 기본 의미적 특성
  private getDefaultSemanticFeatures(): any {
    return {
      target_demographic: "general",
      usage_pattern: "general",
      style_category: "practical",
      reliability_score: 5,
      maintenance_cost_level: "medium",
      resale_value_potential: "fair",
      lifestyle_match: ["commuting"],
      emotional_appeal: {
        prestige: 5,
        sportiness: 5,
        practicality: 7,
        eco_friendliness: 5
      }
    };
  }

  // 시장 데이터 통합
  private async enrichWithMarketData(carData: ProcessedCarData): Promise<any> {
    // 시장 데이터 API 호출 또는 데이터베이스 조회
    return {
      market_popularity: await this.getMarketPopularity(carData),
      price_trend: await this.getPriceTrend(carData),
      availability_index: await this.getAvailabilityIndex(carData),
      competitor_analysis: await this.getCompetitorAnalysis(carData)
    };
  }

  // 유사도 벡터 생성
  private async generateSimilarityVector(
    carData: ProcessedCarData,
    semanticFeatures: any
  ): Promise<number[]> {
    const vector: number[] = [];

    // 기본 속성 벡터 (0-20)
    vector.push(
      this.normalizeYear(carData.year),
      this.normalizePrice(carData.price),
      this.normalizeMileage(carData.mileage),
      this.encodeBrand(carData.brand),
      this.encodeBodyType(carData.bodyType),
      this.encodeFuelType(carData.fuelType)
    );

    // 의미적 특성 벡터 (20-40)
    if (semanticFeatures) {
      vector.push(
        this.encodeTargetDemographic(semanticFeatures.target_demographic),
        this.encodeUsagePattern(semanticFeatures.usage_pattern),
        this.encodeStyleCategory(semanticFeatures.style_category),
        semanticFeatures.reliability_score / 10,
        this.encodeMaintenanceCost(semanticFeatures.maintenance_cost_level),
        semanticFeatures.emotional_appeal.prestige / 10,
        semanticFeatures.emotional_appeal.sportiness / 10,
        semanticFeatures.emotional_appeal.practicality / 10,
        semanticFeatures.emotional_appeal.eco_friendliness / 10
      );
    }

    // 벡터를 512차원으로 패딩
    while (vector.length < 512) {
      vector.push(0);
    }

    return vector;
  }

  // 데이터 품질 검증
  private validateDataQuality(data: CrawlerDataInput): boolean {
    const raw = data.raw_data.data;
    const quality = data.quality_metrics;

    // 필수 필드 검증
    if (!raw.brand || !raw.model || !raw.year || !raw.price) {
      return false;
    }

    // 품질 임계값 검증
    if (quality.completeness_score < 0.6 || quality.reliability_score < 0.4) {
      return false;
    }

    // 데이터 범위 검증
    if (raw.year < 1990 || raw.year > new Date().getFullYear() + 1) {
      return false;
    }

    if (raw.price < 100 || raw.price > 100000) {
      return false;
    }

    return true;
  }

  // 헬퍼 메서드들
  private generateCarId(crawlerData: CrawlerDataInput): string {
    const raw = crawlerData.raw_data.data;
    const source = crawlerData.raw_data.source;

    return `${source}_${raw.brand}_${raw.model}_${raw.year}_${Date.now()}`;
  }

  private normalizeBrand(brand: string): string {
    const brandMap: Record<string, string> = {
      '현대': '현대',
      'hyundai': '현대',
      'HYUNDAI': '현대',
      '기아': '기아',
      'kia': '기아',
      'KIA': '기아'
      // 더 많은 브랜드 매핑 추가
    };

    return brandMap[brand] || brand;
  }

  private normalizeModel(model: string): string {
    return model.trim().toLowerCase();
  }

  private normalizeFuelType(fuelType: string): string {
    const fuelMap: Record<string, string> = {
      '가솔린': 'gasoline',
      '휘발유': 'gasoline',
      '디젤': 'diesel',
      '경유': 'diesel',
      '하이브리드': 'hybrid',
      '전기': 'electric',
      'LPG': 'lpg'
    };

    return fuelMap[fuelType] || fuelType;
  }

  private normalizeBodyType(bodyType: string): string {
    const typeMap: Record<string, string> = {
      '세단': 'sedan',
      'SUV': 'suv',
      '해치백': 'hatchback',
      '쿠페': 'coupe',
      '왜건': 'wagon'
    };

    return typeMap[bodyType] || bodyType;
  }

  private determineMarketPosition(price: number): 'budget' | 'mid-range' | 'premium' {
    if (price < 2000) return 'budget';
    if (price < 5000) return 'mid-range';
    return 'premium';
  }

  // 정규화 및 인코딩 메서드들
  private normalizeYear(year: number): number {
    const currentYear = new Date().getFullYear();
    return (year - 1990) / (currentYear - 1990);
  }

  private normalizePrice(price: number): number {
    return Math.min(1, price / 10000); // 1억원을 최대값으로 정규화
  }

  private normalizeMileage(mileage: number): number {
    return Math.min(1, mileage / 300000); // 30만km를 최대값으로 정규화
  }

  private encodeBrand(brand: string): number {
    const brands = ['현대', '기아', 'BMW', '벤츠', '아우디', '토요타'];
    const index = brands.indexOf(brand);
    return index === -1 ? 0.5 : index / brands.length;
  }

  private encodeBodyType(bodyType: string): number {
    const types = ['sedan', 'suv', 'hatchback', 'coupe', 'wagon'];
    const index = types.indexOf(bodyType);
    return index === -1 ? 0.5 : index / types.length;
  }

  private encodeFuelType(fuelType: string): number {
    const types = ['gasoline', 'diesel', 'hybrid', 'electric', 'lpg'];
    const index = types.indexOf(fuelType);
    return index === -1 ? 0.5 : index / types.length;
  }

  private encodeTargetDemographic(demographic: string): number {
    const types = ['young_professional', 'family', 'luxury_seeker', 'economy_focused', 'senior'];
    const index = types.indexOf(demographic);
    return index === -1 ? 0.5 : index / types.length;
  }

  private encodeUsagePattern(pattern: string): number {
    const patterns = ['city_driving', 'highway_cruising', 'family_transport', 'business', 'recreation'];
    const index = patterns.indexOf(pattern);
    return index === -1 ? 0.5 : index / patterns.length;
  }

  private encodeStyleCategory(style: string): number {
    const styles = ['sporty', 'elegant', 'practical', 'rugged', 'minimalist'];
    const index = styles.indexOf(style);
    return index === -1 ? 0.5 : index / styles.length;
  }

  private encodeMaintenanceCost(cost: string): number {
    const costs = ['low', 'medium', 'high'];
    const index = costs.indexOf(cost);
    return index === -1 ? 0.5 : index / costs.length;
  }

  // 시장 데이터 조회 메서드들 (추후 구현)
  private async getMarketPopularity(carData: ProcessedCarData): Promise<number> {
    return 0.5; // 임시값
  }

  private async getPriceTrend(carData: ProcessedCarData): Promise<number[]> {
    return [0.5, 0.5, 0.5]; // 임시값
  }

  private async getAvailabilityIndex(carData: ProcessedCarData): Promise<number> {
    return 0.5; // 임시값
  }

  private async getCompetitorAnalysis(carData: ProcessedCarData): Promise<any> {
    return {}; // 임시값
  }

  private async estimateFuelEfficiency(carData: ProcessedCarData): Promise<number> {
    return 10; // 임시값 (km/l)
  }

  private async calculateDepreciationRate(carData: ProcessedCarData): Promise<number> {
    return 0.15; // 임시값 (연 15% 감가상각)
  }
}