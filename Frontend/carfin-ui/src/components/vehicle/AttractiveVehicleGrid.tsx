'use client';

import { useState, useEffect } from 'react';
import { Container } from '@/components/design-system/layout/Container';
import { Button } from '@/components/ui/button';
import {
  Heart,
  X,
  DollarSign,
  Star,
  Flame,
  MessageCircle,
  Car,
  Fuel,
  Calendar,
  MapPin,
  TrendingUp,
  Zap,
  ArrowRight,
  Sparkles,
  Eye,
  Shield,
  Award
} from 'lucide-react';

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
  description: string;
  highlight?: string;
}

interface VehicleFeedback {
  vehicleId: string;
  feedbackType: 'love' | 'like' | 'dislike' | 'expensive' | 'maybe' | 'question';
  timestamp: Date;
}

interface AttractiveVehicleGridProps {
  userProfile?: any;
  onSelectionComplete?: (vehicles: Vehicle[], feedback: VehicleFeedback[]) => void;
}

export function AttractiveVehicleGrid({ userProfile, onSelectionComplete }: AttractiveVehicleGridProps) {
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [feedback, setFeedback] = useState<VehicleFeedback[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [showIntro, setShowIntro] = useState(true);
  const [likedCount, setLikedCount] = useState(0);

  useEffect(() => {
    loadVehicles();
  }, []);

  const loadVehicles = async () => {
    setIsLoading(true);

    // 백그라운드에서 AI 분석이 진행되는 동안 매력적인 차량들을 먼저 보여줌
    const attractiveVehicles = getAttractiveVehicles();
    setVehicles(attractiveVehicles);
    setIsLoading(false);
  };

  const getAttractiveVehicles = (): Vehicle[] => [
    {
      id: "1",
      brand: "테슬라",
      model: "모델 3",
      year: 2022,
      price: 4200,
      mileage: 15000,
      fuel_type: "전기",
      body_type: "세단",
      color: "화이트",
      location: "서울",
      images: ["/api/placeholder/600/400"],
      features: ["오토파일럿", "슈퍼차징", "OTA업데이트"],
      fuel_efficiency: 0,
      safety_rating: 5,
      match_score: 95,
      description: "미래를 경험하는 프리미엄 전기차",
      highlight: "🔥 인기급상승"
    },
    {
      id: "2",
      brand: "BMW",
      model: "320i",
      year: 2021,
      price: 3800,
      mileage: 28000,
      fuel_type: "가솔린",
      body_type: "세단",
      color: "블랙",
      location: "서울",
      images: ["/api/placeholder/600/400"],
      features: ["런플랫타이어", "하만카돈", "무선충전"],
      fuel_efficiency: 13.1,
      safety_rating: 5,
      match_score: 88,
      description: "독일 엔지니어링의 정수",
      highlight: "👑 프리미엄"
    },
    {
      id: "3",
      brand: "현대",
      model: "그랜저",
      year: 2023,
      price: 3500,
      mileage: 8000,
      fuel_type: "하이브리드",
      body_type: "세단",
      color: "화이트",
      location: "경기",
      images: ["/api/placeholder/600/400"],
      features: ["마사지시트", "디스플레이오디오", "V2L"],
      fuel_efficiency: 15.9,
      safety_rating: 5,
      match_score: 92,
      description: "한국 최고급 하이브리드",
      highlight: "✨ 신차급"
    },
    {
      id: "4",
      brand: "제네시스",
      model: "G90",
      year: 2022,
      price: 6200,
      mileage: 12000,
      fuel_type: "가솔린",
      body_type: "세단",
      color: "그레이",
      location: "서울",
      images: ["/api/placeholder/600/400"],
      features: ["나파가죽", "3D서라운드", "에어서스펜션"],
      fuel_efficiency: 9.8,
      safety_rating: 5,
      match_score: 85,
      description: "럭셔리의 새로운 정의",
      highlight: "💎 최고급"
    },
    {
      id: "5",
      brand: "기아",
      model: "EV6",
      year: 2023,
      price: 4800,
      mileage: 5000,
      fuel_type: "전기",
      body_type: "SUV",
      color: "화이트",
      location: "인천",
      images: ["/api/placeholder/600/400"],
      features: ["초고속충전", "V2L", "얼굴인식"],
      fuel_efficiency: 0,
      safety_rating: 5,
      match_score: 90,
      description: "전기차의 새로운 기준",
      highlight: "⚡ 신기술"
    },
    {
      id: "6",
      brand: "아우디",
      model: "A6",
      year: 2021,
      price: 4500,
      mileage: 22000,
      fuel_type: "가솔린",
      body_type: "세단",
      color: "실버",
      location: "경기",
      images: ["/api/placeholder/600/400"],
      features: ["콰트로", "버추얼콕핏", "매트릭스LED"],
      fuel_efficiency: 11.5,
      safety_rating: 5,
      match_score: 87,
      description: "독일 프리미엄의 완성",
      highlight: "🏆 베스트셀러"
    },
    {
      id: "7",
      brand: "렉서스",
      model: "ES300h",
      year: 2022,
      price: 5200,
      mileage: 18000,
      fuel_type: "하이브리드",
      body_type: "세단",
      color: "블랙",
      location: "서울",
      images: ["/api/placeholder/600/400"],
      features: ["마크레빈슨", "세미아닐린가죽", "LSS+"],
      fuel_efficiency: 17.2,
      safety_rating: 5,
      match_score: 89,
      description: "일본 장인정신의 걸작",
      highlight: "🎯 추천"
    },
    {
      id: "8",
      brand: "포르쉐",
      model: "마칸",
      year: 2021,
      price: 7800,
      mileage: 25000,
      fuel_type: "가솔린",
      body_type: "SUV",
      color: "레드",
      location: "서울",
      images: ["/api/placeholder/600/400"],
      features: ["PASM", "보스사운드", "파노라마루프"],
      fuel_efficiency: 9.2,
      safety_rating: 5,
      match_score: 82,
      description: "스포츠카의 DNA를 가진 SUV",
      highlight: "🏎️ 스포츠"
    }
  ];

  const handleFeedback = (feedbackType: VehicleFeedback['feedbackType']) => {
    const currentVehicle = vehicles[currentIndex];
    if (!currentVehicle) return;

    const newFeedback: VehicleFeedback = {
      vehicleId: currentVehicle.id,
      feedbackType,
      timestamp: new Date()
    };

    const updatedFeedback = [...feedback, newFeedback];
    setFeedback(updatedFeedback);

    if (feedbackType === 'love' || feedbackType === 'like') {
      setLikedCount(prev => prev + 1);
    }

    // 다음 차량으로 이동
    if (currentIndex < vehicles.length - 1) {
      setCurrentIndex(currentIndex + 1);
    } else {
      // 완료 처리
      const likedVehicles = updatedFeedback
        .filter(f => f.feedbackType === 'love' || f.feedbackType === 'like')
        .map(f => vehicles.find(v => v.id === f.vehicleId))
        .filter(Boolean) as Vehicle[];

      setTimeout(() => {
        onSelectionComplete?.(likedVehicles.slice(0, 3), updatedFeedback);
      }, 1000);
    }
  };

  const currentVehicle = vehicles[currentIndex];

  if (showIntro) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 flex items-center justify-center">
        <Container size="md">
          <div className="text-center space-y-8">
            <div className="space-y-4">
              <div className="w-20 h-20 bg-gradient-to-r from-pink-500 to-purple-600 rounded-3xl flex items-center justify-center mx-auto">
                <Heart className="w-10 h-10 text-white" />
              </div>
              <h1 className="text-4xl md:text-5xl font-bold text-gray-900">
                당신의 완벽한 차량을 찾아보세요
              </h1>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                마음에 드는 차량에는 ❤️, 별로인 차량에는 ✕를 눌러주세요.<br/>
                AI가 당신의 취향을 학습해서 최적의 매칭을 해드립니다.
              </p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto">
              <div className="bg-white rounded-2xl p-4 shadow-sm border">
                <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Heart className="w-6 h-6 text-red-500" />
                </div>
                <div className="text-sm font-medium text-gray-900">완전 좋아요</div>
                <div className="text-xs text-gray-600">이런 차량 더 보고 싶어요</div>
              </div>

              <div className="bg-white rounded-2xl p-4 shadow-sm border">
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Star className="w-6 h-6 text-blue-500" />
                </div>
                <div className="text-sm font-medium text-gray-900">괜찮아요</div>
                <div className="text-xs text-gray-600">나쁘지 않네요</div>
              </div>

              <div className="bg-white rounded-2xl p-4 shadow-sm border">
                <div className="w-12 h-12 bg-orange-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <DollarSign className="w-6 h-6 text-orange-500" />
                </div>
                <div className="text-sm font-medium text-gray-900">비싸요</div>
                <div className="text-xs text-gray-600">가격이 부담스러워요</div>
              </div>

              <div className="bg-white rounded-2xl p-4 shadow-sm border">
                <div className="w-12 h-12 bg-gray-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <X className="w-6 h-6 text-gray-500" />
                </div>
                <div className="text-sm font-medium text-gray-900">별로예요</div>
                <div className="text-xs text-gray-600">취향이 아니에요</div>
              </div>
            </div>

            <Button
              variant="default"
              size="lg"
              onClick={() => setShowIntro(false)}
              icon={<Sparkles className="w-6 h-6" aria-hidden="true" />}
              className="shadow-xl shadow-purple-200"
              aria-label="차량 선호도 테스트 시작하기"
            >
              차량 테스트 시작하기
            </Button>
          </div>
        </Container>
      </div>
    );
  }

  if (isLoading || !currentVehicle) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Car className="w-8 h-8 text-blue-600 animate-pulse" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">AI가 차량을 준비하고 있어요</h2>
          <p className="text-gray-600">당신에게 맞는 차량들을 선별중입니다...</p>
        </div>
      </div>
    );
  }

  if (currentIndex >= vehicles.length) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 flex items-center justify-center">
        <Container size="md">
          <div className="text-center space-y-8">
            <div className="w-20 h-20 bg-gradient-to-r from-green-500 to-blue-600 rounded-full flex items-center justify-center mx-auto">
              <TrendingUp className="w-10 h-10 text-white" />
            </div>
            <h2 className="text-4xl font-bold text-gray-900">
              완료! 🎉
            </h2>
            <p className="text-xl text-gray-600">
              {likedCount}개의 차량을 선택하셨어요.<br/>
              AI가 분석 결과를 준비하고 있습니다.
            </p>
            <div className="flex items-center justify-center gap-2">
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
              <div className="w-2 h-2 bg-blue-600 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
            </div>
          </div>
        </Container>
      </div>
    );
  }

  const progress = ((currentIndex + 1) / vehicles.length) * 100;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* 헤더 */}
      <div className="bg-white/90 backdrop-blur-md border-b border-gray-200 sticky top-0 z-10">
        <Container>
          <div className="py-4">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                  <Car className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-gray-900">취향 테스트</h1>
                  <p className="text-sm text-gray-600">{currentIndex + 1} / {vehicles.length}</p>
                </div>
              </div>

              <div className="text-center">
                <div className="text-sm text-gray-500 mb-1">선택한 차량</div>
                <div className="flex items-center gap-1">
                  <Heart className="w-4 h-4 text-red-500" />
                  <span className="font-bold text-red-500">{likedCount}</span>
                </div>
              </div>
            </div>

            {/* 진행바 */}
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-blue-600 to-purple-600 h-2 rounded-full transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </Container>
      </div>

      {/* 메인 콘텐츠 */}
      <Container size="md">
        <div className="py-8">
          <div className="max-w-md mx-auto">

            {/* 차량 카드 - Instagram 스타일 */}
            <div className="bg-white rounded-3xl shadow-xl overflow-hidden mb-8 relative">
              {/* 하이라이트 배지 */}
              {currentVehicle.highlight && (
                <div className="absolute top-4 left-4 z-10">
                  <div className="bg-gradient-to-r from-orange-500 to-red-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                    {currentVehicle.highlight}
                  </div>
                </div>
              )}

              {/* 매치 점수 */}
              <div className="absolute top-4 right-4 z-10">
                <div className="bg-white/90 backdrop-blur-sm rounded-full px-3 py-1 flex items-center gap-1">
                  <TrendingUp className="w-4 h-4 text-green-600" />
                  <span className="text-sm font-bold text-green-600">{currentVehicle.match_score}%</span>
                </div>
              </div>

              {/* 차량 이미지 */}
              <div className="aspect-[4/3] bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center relative">
                <Car className="w-20 h-20 text-gray-400" />

                {/* 뷰 카운터 (가상) */}
                <div className="absolute bottom-4 left-4">
                  <div className="bg-black/70 backdrop-blur-sm rounded-full px-3 py-1 flex items-center gap-1">
                    <Eye className="w-4 h-4 text-white" />
                    <span className="text-sm font-medium text-white">{Math.floor(Math.random() * 500 + 100)}</span>
                  </div>
                </div>
              </div>

              {/* 차량 정보 */}
              <div className="p-6">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900 mb-1">
                      {currentVehicle.brand} {currentVehicle.model}
                    </h2>
                    <p className="text-gray-600">{currentVehicle.year}년 • {currentVehicle.location}</p>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-blue-600">
                      {currentVehicle.price.toLocaleString()}만원
                    </div>
                  </div>
                </div>

                {/* 주요 스펙 */}
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Fuel className="w-4 h-4" />
                    <span>{currentVehicle.fuel_type}</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <MapPin className="w-4 h-4" />
                    <span>{currentVehicle.mileage.toLocaleString()}km</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Shield className="w-4 h-4" />
                    <span>안전도 {currentVehicle.safety_rating}점</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <Zap className="w-4 h-4" />
                    <span>{currentVehicle.fuel_efficiency > 0 ? `${currentVehicle.fuel_efficiency}km/L` : '전기차'}</span>
                  </div>
                </div>

                {/* 설명 */}
                <p className="text-gray-700 mb-6 text-center font-medium">
                  {currentVehicle.description}
                </p>

                {/* 특징 태그 */}
                <div className="flex flex-wrap gap-2 mb-6">
                  {currentVehicle.features.slice(0, 3).map((feature, index) => (
                    <span
                      key={index}
                      className="px-3 py-1 bg-blue-50 text-blue-700 text-sm rounded-full font-medium"
                    >
                      {feature}
                    </span>
                  ))}
                </div>
              </div>
            </div>

            {/* 액션 버튼들 - Tinder 스타일 */}
            <div className="flex items-center justify-center gap-4">
              <button
                onClick={() => handleFeedback('dislike')}
                className="w-16 h-16 bg-white rounded-full shadow-lg border-2 border-gray-200 flex items-center justify-center hover:border-red-300 hover:bg-red-50 transition-all duration-200 active:scale-95"
              >
                <X className="w-8 h-8 text-gray-600 hover:text-red-500" />
              </button>

              <button
                onClick={() => handleFeedback('expensive')}
                className="w-14 h-14 bg-white rounded-full shadow-lg border-2 border-gray-200 flex items-center justify-center hover:border-orange-300 hover:bg-orange-50 transition-all duration-200 active:scale-95"
              >
                <DollarSign className="w-6 h-6 text-gray-600 hover:text-orange-500" />
              </button>

              <button
                onClick={() => handleFeedback('like')}
                className="w-14 h-14 bg-white rounded-full shadow-lg border-2 border-gray-200 flex items-center justify-center hover:border-blue-300 hover:bg-blue-50 transition-all duration-200 active:scale-95"
              >
                <Star className="w-6 h-6 text-gray-600 hover:text-blue-500" />
              </button>

              <button
                onClick={() => handleFeedback('love')}
                className="w-16 h-16 bg-white rounded-full shadow-lg border-2 border-gray-200 flex items-center justify-center hover:border-red-300 hover:bg-red-50 transition-all duration-200 active:scale-95"
              >
                <Heart className="w-8 h-8 text-gray-600 hover:text-red-500" />
              </button>
            </div>

            {/* 도움말 */}
            <div className="text-center mt-6">
              <p className="text-sm text-gray-500">
                왼쪽 스와이프 ← 별로 | 오른쪽 스와이프 → 좋아요
              </p>
            </div>
          </div>
        </div>
      </Container>
    </div>
  );
}