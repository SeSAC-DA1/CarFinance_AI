// 더미 데이터 생성기 - 협업 필터링 시뮬레이션용
import { ProcessedCarData, UserProfile, UserInteraction } from '@/types';

export class MockDataGenerator {
  private brands = ['현대', '기아', 'BMW', '벤츠', '아우디', '토요타', '혼다', '닛산', '포드', '폭스바겐'];
  private models = {
    '현대': ['아반떼', '소나타', '그랜저', '산타페', '투싼', '코나', '벨로스터'],
    '기아': ['K3', 'K5', 'K7', '쏘렌토', '스포티지', '니로', '셀토스'],
    'BMW': ['320i', '520i', '730i', 'X3', 'X5', 'X7', 'i4'],
    '벤츠': ['C-Class', 'E-Class', 'S-Class', 'GLC', 'GLE', 'GLS', 'EQC'],
    '아우디': ['A3', 'A4', 'A6', 'Q5', 'Q7', 'Q8', 'e-tron'],
    '토요타': ['캠리', '아발론', '카롤라', 'RAV4', '하이랜더', '프라도', '프리우스'],
    '혼다': ['어코드', '시빅', 'CR-V', '파일럿', '오디세이', '인사이트'],
    '닛산': ['알티마', '맥시마', '로그', '무라노', 'GT-R'],
    '포드': ['몬데오', '머스탱', '익스플로러', 'F-150'],
    '폭스바겐': ['골프', '파사트', '티구안', '투아렉']
  };

  private bodyTypes = ['sedan', 'suv', 'hatchback', 'coupe', 'wagon'];
  private fuelTypes = ['gasoline', 'diesel', 'hybrid', 'electric'];
  private transmissions = ['manual', 'automatic'];
  private colors = ['흰색', '검은색', '은색', '빨간색', '파란색', '회색', '갈색'];

  // 더미 차량 데이터 생성
  generateCarData(count: number = 100): ProcessedCarData[] {
    const cars: ProcessedCarData[] = [];

    for (let i = 0; i < count; i++) {
      const brand = this.randomChoice(this.brands);
      const model = this.randomChoice(this.models[brand]);
      const year = this.randomInt(2015, 2024);
      const currentYear = new Date().getFullYear();
      const age = currentYear - year;

      const car: ProcessedCarData = {
        id: `car_${i + 1}`,
        brand,
        model,
        year,
        price: this.generatePrice(brand, model, age),
        mileage: this.generateMileage(age),
        fuelType: this.randomChoice(this.fuelTypes),
        transmission: this.randomChoice(this.transmissions),
        bodyType: this.randomChoice(this.bodyTypes),
        images: this.generateImageUrls(brand, model),
        features: this.generateFeatures(),
        location: this.randomChoice(['서울', '경기', '부산', '대구', '인천', '광주', '대전']),
        dealer: `${brand} 공식딜러 ${this.randomInt(1, 50)}호점`,
        sourceUrl: `https://example.com/cars/${i + 1}`,
        lastUpdated: Date.now() - this.randomInt(0, 30) * 24 * 60 * 60 * 1000,

        normalized_features: {
          age: age,
          mileage_per_year: age > 0 ? this.generateMileage(age) / age : 0,
          price_per_year: age > 0 ? this.generatePrice(brand, model, age) / age : 0,
          fuel_efficiency: this.generateFuelEfficiency(this.randomChoice(this.fuelTypes)),
          depreciation_rate: this.calculateDepreciationRate(brand, age),
          market_position: this.determineMarketPosition(brand)
        },

        semantic_features: {
          target_demographic: this.randomChoice(['young_professional', 'family', 'luxury_seeker', 'economy_focused', 'senior']),
          usage_pattern: this.randomChoice(['city_driving', 'highway_cruising', 'family_transport', 'business', 'recreation']),
          style_category: this.randomChoice(['sporty', 'elegant', 'practical', 'rugged', 'minimalist']),
          reliability_score: this.randomFloat(6, 10),
          maintenance_cost_level: this.randomChoice(['low', 'medium', 'high'])
        },

        similarity_vector: this.generateSimilarityVector(brand, model, year, this.generatePrice(brand, model, age))
      };

      cars.push(car);
    }

    return cars;
  }

  // 더미 사용자 프로필 생성
  generateUserProfiles(count: number = 20): UserProfile[] {
    const users: UserProfile[] = [];

    for (let i = 0; i < count; i++) {
      const user: UserProfile = {
        id: `user_${i + 1}`,
        preferences: this.generateUserPreferences(),
        interaction_history: [],
        learning_metadata: {
          session_count: this.randomInt(1, 50),
          last_active: Date.now() - this.randomInt(0, 30) * 24 * 60 * 60 * 1000,
          preference_confidence: this.randomFloat(0.3, 0.9),
          behavioral_patterns: this.generateBehavioralPatterns()
        }
      };

      users.push(user);
    }

    return users;
  }

  // 더미 사용자 상호작용 데이터 생성
  generateUserInteractions(users: UserProfile[], cars: ProcessedCarData[], interactionsPerUser: number = 10): UserInteraction[] {
    const interactions: UserInteraction[] = [];

    users.forEach(user => {
      for (let i = 0; i < interactionsPerUser; i++) {
        const car = this.randomChoice(cars);
        const action = this.randomChoice(['click', 'long_hover', 'like', 'save', 'detail_view', 'compare_add', 'skip']);

        const interaction: UserInteraction = {
          id: `interaction_${user.id}_${car.id}_${i}`,
          userId: user.id,
          sessionId: `session_${user.id}_${Math.floor(i / 3)}`,
          type: 'implicit',
          action,
          target: {
            type: 'car_card',
            carId: car.id,
            position: this.randomInt(1, 20)
          },
          context: {
            duration: this.generateInteractionDuration(action),
            scrollPosition: this.randomInt(0, 5000),
            viewport: { width: 1920, height: 1080 },
            device: this.randomChoice(['desktop', 'mobile', 'tablet']),
            timeOfDay: this.generateTimeOfDay()
          },
          timestamp: Date.now() - this.randomInt(0, 30) * 24 * 60 * 60 * 1000
        };

        interactions.push(interaction);
      }
    });

    return interactions;
  }

  // 가격 생성 (브랜드별 차등)
  private generatePrice(brand: string, model: string, age: number): number {
    const basePrice = this.getBrandBasePrice(brand);
    const modelMultiplier = this.getModelMultiplier(model);
    const ageDepreciation = Math.max(0.3, 1 - (age * 0.1)); // 연간 10% 감가상각

    const price = basePrice * modelMultiplier * ageDepreciation;
    return Math.round(price / 10) * 10; // 10만원 단위로 반올림
  }

  private getBrandBasePrice(brand: string): number {
    const basePrices: Record<string, number> = {
      '현대': 2500,
      '기아': 2400,
      'BMW': 5000,
      '벤츠': 6000,
      '아우디': 5500,
      '토요타': 3000,
      '혼다': 2800,
      '닛산': 2600,
      '포드': 3200,
      '폭스바겐': 3500
    };
    return basePrices[brand] || 2500;
  }

  private getModelMultiplier(model: string): number {
    const luxuryModels = ['그랜저', 'K7', '730i', 'S-Class', 'A6', 'GLS'];
    const sportModels = ['벨로스터', 'GT-R', '머스탱'];
    const suvModels = ['산타페', '쏘렌토', 'X5', 'GLE', 'Q7'];

    if (luxuryModels.some(luxury => model.includes(luxury))) return 1.5;
    if (sportModels.some(sport => model.includes(sport))) return 1.4;
    if (suvModels.some(suv => model.includes(suv))) return 1.3;
    return 1.0;
  }

  // 주행거리 생성 (연식별)
  private generateMileage(age: number): number {
    const baseAnnualMileage = this.randomInt(8000, 20000);
    return Math.round(baseAnnualMileage * age * this.randomFloat(0.8, 1.2));
  }

  // 연비 생성
  private generateFuelEfficiency(fuelType: string): number {
    const efficiency: Record<string, [number, number]> = {
      'gasoline': [8, 15],
      'diesel': [12, 18],
      'hybrid': [15, 25],
      'electric': [4, 6] // kWh/100km
    };

    const [min, max] = efficiency[fuelType] || [8, 15];
    return this.randomFloat(min, max);
  }

  // 사용자 선호도 생성
  private generateUserPreferences(): Record<string, number> {
    return {
      price_sensitivity: this.randomFloat(0.2, 0.9),
      age_preference: this.randomFloat(0.3, 0.8),
      mileage_importance: this.randomFloat(0.4, 0.9),
      brand_loyalty: this.randomFloat(0.1, 0.7),
      fuel_efficiency_priority: this.randomFloat(0.3, 0.9),
      performance_preference: this.randomFloat(0.2, 0.8),
      luxury_inclination: this.randomFloat(0.1, 0.6),
      practicality_focus: this.randomFloat(0.5, 0.9),
      safety_priority: this.randomFloat(0.6, 1.0),
      technology_interest: this.randomFloat(0.3, 0.8)
    };
  }

  // 행동 패턴 생성
  private generateBehavioralPatterns(): string[] {
    const patterns = ['price_conscious', 'feature_focused', 'brand_loyal', 'impulsive', 'research_heavy', 'quick_decision'];
    return this.randomSample(patterns, this.randomInt(1, 3));
  }

  // 상호작용 지속시간 생성
  private generateInteractionDuration(action: string): number {
    const durations: Record<string, [number, number]> = {
      'click': [100, 500],
      'long_hover': [2000, 10000],
      'like': [200, 800],
      'save': [300, 1000],
      'detail_view': [5000, 30000],
      'compare_add': [800, 2000],
      'skip': [50, 200]
    };

    const [min, max] = durations[action] || [100, 1000];
    return this.randomInt(min, max);
  }

  // 시간대 생성
  private generateTimeOfDay(): string {
    const hour = this.randomInt(0, 23);
    if (hour >= 6 && hour < 12) return 'morning';
    if (hour >= 12 && hour < 18) return 'afternoon';
    if (hour >= 18 && hour < 22) return 'evening';
    return 'night';
  }

  // 차량 특성 생성
  private generateFeatures(): string[] {
    const allFeatures = [
      '네비게이션', '후방카메라', '블루투스', '크루즈컨트롤', '자동주차',
      '스마트키', '썬루프', '가죽시트', '열선시트', '통풍시트',
      '자동에어컨', 'LED헤드라이트', '어라운드뷰', '차선이탈경보',
      '자동긴급제동', '스마트폰연동', '무선충전', '프리미엄사운드'
    ];

    return this.randomSample(allFeatures, this.randomInt(3, 8));
  }

  // 이미지 URL 생성
  private generateImageUrls(brand: string, model: string): string[] {
    const baseUrl = 'https://picsum.photos';
    const imageCount = this.randomInt(3, 8);
    const urls: string[] = [];

    for (let i = 0; i < imageCount; i++) {
      urls.push(`${baseUrl}/800/600?random=${brand}-${model}-${i}`);
    }

    return urls;
  }

  // 유사도 벡터 생성 (512차원)
  private generateSimilarityVector(brand: string, model: string, year: number, price: number): number[] {
    const vector = new Array(512).fill(0);

    // 기본 특성 (인덱스 0-10)
    vector[0] = this.normalizeBrand(brand);
    vector[1] = this.normalizeYear(year);
    vector[2] = this.normalizePrice(price);
    vector[3] = this.randomFloat(0, 1); // 연비 점수
    vector[4] = this.randomFloat(0, 1); // 안전 점수

    // 의미적 특성 (인덱스 10-100)
    for (let i = 10; i < 100; i++) {
      vector[i] = this.randomFloat(0, 1);
    }

    // 나머지는 노이즈 또는 0
    for (let i = 100; i < 512; i++) {
      vector[i] = this.randomFloat(-0.1, 0.1);
    }

    return vector;
  }

  // 헬퍼 함수들
  private randomChoice<T>(array: T[]): T {
    return array[Math.floor(Math.random() * array.length)];
  }

  private randomSample<T>(array: T[], count: number): T[] {
    const shuffled = [...array].sort(() => 0.5 - Math.random());
    return shuffled.slice(0, count);
  }

  private randomInt(min: number, max: number): number {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  private randomFloat(min: number, max: number): number {
    return Math.random() * (max - min) + min;
  }

  private normalizeBrand(brand: string): number {
    const brands = ['현대', '기아', 'BMW', '벤츠', '아우디', '토요타', '혼다', '닛산', '포드', '폭스바겐'];
    const index = brands.indexOf(brand);
    return index === -1 ? 0.5 : index / brands.length;
  }

  private normalizeYear(year: number): number {
    return (year - 2015) / (2024 - 2015);
  }

  private normalizePrice(price: number): number {
    return Math.min(1, price / 10000); // 1억원을 최대값으로 정규화
  }

  private calculateDepreciationRate(brand: string, age: number): number {
    const luxuryBrands = ['BMW', '벤츠', '아우디'];
    const baseDep = luxuryBrands.includes(brand) ? 0.12 : 0.15;
    return baseDep + (age * 0.01);
  }

  private determineMarketPosition(brand: string): 'budget' | 'mid-range' | 'premium' {
    const luxuryBrands = ['BMW', '벤츠', '아우디'];
    const midBrands = ['토요타', '혼다', '폭스바겐'];

    if (luxuryBrands.includes(brand)) return 'premium';
    if (midBrands.includes(brand)) return 'mid-range';
    return 'budget';
  }
}

// 미리 생성된 더미 데이터 인스턴스
export const mockDataGenerator = new MockDataGenerator();

// 전역 더미 데이터
export const MOCK_CARS = mockDataGenerator.generateCarData(150);
export const MOCK_USERS = mockDataGenerator.generateUserProfiles(30);
export const MOCK_INTERACTIONS = mockDataGenerator.generateUserInteractions(MOCK_USERS, MOCK_CARS, 15);