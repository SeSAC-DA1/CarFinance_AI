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
  Car,
  Fuel,
  Calendar,
  MapPin,
  TrendingUp,
  Zap,
  ArrowRight,
  Sparkles,
  Shield,
  Award,
  CheckCircle,
  Eye,
  ThumbsUp,
  ThumbsDown
} from 'lucide-react';
import {
  Vehicle,
  VehicleFeedback,
  ModernVehicleGridProps
} from '@/types';

export function ModernVehicleGrid({ userProfile, onSelectionComplete }: ModernVehicleGridProps) {
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [vehicleFeedbacks, setVehicleFeedbacks] = useState<Record<string, VehicleFeedback['feedbackType']>>({});
  const [isLoading, setIsLoading] = useState(true);
  const [showIntro, setShowIntro] = useState(true);
  const [currentPhase, setCurrentPhase] = useState<'selection' | 'completed'>('selection');

  useEffect(() => {
    loadVehicles();
  }, []);

  const loadVehicles = async () => {
    setIsLoading(true);

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
      images: ["/api/placeholder/400/300"],
      features: ["오토파일럿", "슈퍼차징", "OTA업데이트"],
      fuel_efficiency: 0,
      safety_rating: 5,
      match_score: 95,
      description: "완전 자율주행 기능과 무선 업데이트로 계속 진화하는 전기차. 슈퍼차저 네트워크로 전국 어디든 빠른 충전 가능. 연비 걱정 없이 월 전기료 10만원대로 운행 가능.",
      highlight: "🔥 인기급상승"
    },
    {
      id: "2",
      brand: "현대",
      model: "그랜저",
      year: 2023,
      price: 3500,
      mileage: 8000,
      fuel_type: "하이브리드",
      body_type: "세단",
      color: "화이트",
      location: "경기",
      images: ["/api/placeholder/400/300"],
      features: ["마사지시트", "디스플레이오디오", "V2L"],
      fuel_efficiency: 15.9,
      safety_rating: 5,
      match_score: 92,
      description: "국산 최고급 대형 세단으로 뒷좌석 마사지 시트와 냉온장고 구비. 하이브리드로 시내 연비 17km/L 달성. 캠핑용 V2L 기능으로 외부 전원 공급도 가능한 실용적 럭셔리.",
      highlight: "✨ 신차급"
    },
    {
      id: "3",
      brand: "기아",
      model: "EV6",
      year: 2023,
      price: 4800,
      mileage: 5000,
      fuel_type: "전기",
      body_type: "SUV",
      color: "화이트",
      location: "인천",
      images: ["/api/placeholder/400/300"],
      features: ["초고속충전", "V2L", "얼굴인식"],
      fuel_efficiency: 0,
      safety_rating: 5,
      match_score: 90,
      description: "18분만에 80% 충전 가능한 초고속 충전 기술과 최대 3.5kW V2L로 가전제품 사용 가능. 얼굴인식 시트 자동 조절과 AR 내비게이션으로 미래형 드라이빙 경험 제공.",
      highlight: "⚡ 신기술"
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
      images: ["/api/placeholder/400/300"],
      features: ["나파가죽", "3D서라운드", "에어서스펜션"],
      fuel_efficiency: 9.8,
      safety_rating: 5,
      match_score: 85,
      description: "수제 나파가죽 시트와 렉시콘 3D 서라운드 오디오로 최고급 안락함 제공. 에어 서스펜션으로 노면 상태 관계없이 매끄러운 승차감. 대통령차 급의 품격과 안전성.",
      highlight: "💎 최고급"
    },
    {
      id: "5",
      brand: "BMW",
      model: "320i",
      year: 2021,
      price: 3800,
      mileage: 28000,
      fuel_type: "가솔린",
      body_type: "세단",
      color: "블랙",
      location: "서울",
      images: ["/api/placeholder/400/300"],
      features: ["런플랫타이어", "하만카돈", "무선충전"],
      fuel_efficiency: 13.1,
      safety_rating: 5,
      match_score: 88,
      description: "독일산 프리미엄 스포츠 세단으로 50:50 완벽한 무게배분과 후륜구동의 짜릿한 주행감 제공. 하만카돈 오디오와 런플랫 타이어로 안전하고 품격있는 드라이빙. 유지비는 국산차 수준.",
      highlight: "👑 프리미엄"
    },
    {
      id: "6",
      brand: "렉서스",
      model: "ES300h",
      year: 2022,
      price: 5200,
      mileage: 18000,
      fuel_type: "하이브리드",
      body_type: "세단",
      color: "블랙",
      location: "서울",
      images: ["/api/placeholder/400/300"],
      features: ["마크레빈슨", "세미아닐린가죽", "LSS+"],
      fuel_efficiency: 17.2,
      safety_rating: 5,
      match_score: 89,
      description: "일본 최고급 하이브리드로 마크레빈슨 프리미엄 오디오와 세미아닐린 가죽으로 감싸진 실내. 연비 18km/L과 15년 무상 A/S로 경제성과 신뢰성 모두 확보. 조용하고 부드러운 승차감.",
      highlight: "🎯 추천"
    }
  ];

  const handleVehicleFeedback = (vehicleId: string, feedbackType: VehicleFeedback['feedbackType']) => {
    setVehicleFeedbacks(prev => ({
      ...prev,
      [vehicleId]: feedbackType
    }));
  };

  const getSelectedCount = () => {
    return Object.values(vehicleFeedbacks).filter(feedback =>
      feedback === 'love' || feedback === 'like'
    ).length;
  };

  const getFeedbackIcon = (feedbackType?: VehicleFeedback['feedbackType']) => {
    switch (feedbackType) {
      case 'love': return <Heart className="w-5 h-5 text-red-500" />;
      case 'like': return <ThumbsUp className="w-5 h-5 text-blue-500" />;
      case 'dislike': return <ThumbsDown className="w-5 h-5 text-gray-500" />;
      case 'expensive': return <DollarSign className="w-5 h-5 text-orange-500" />;
      default: return null;
    }
  };

  const getFeedbackColor = (feedbackType?: VehicleFeedback['feedbackType']) => {
    switch (feedbackType) {
      case 'love': return 'border-red-500 bg-red-50';
      case 'like': return 'border-blue-500 bg-blue-50';
      case 'dislike': return 'border-gray-400 bg-gray-50 opacity-60';
      case 'expensive': return 'border-orange-500 bg-orange-50';
      default: return 'border-gray-200 bg-white hover:border-gray-300';
    }
  };

  const handleComplete = () => {
    const feedbackArray: VehicleFeedback[] = Object.entries(vehicleFeedbacks).map(([vehicleId, feedbackType]) => ({
      vehicleId,
      feedbackType,
      timestamp: new Date()
    }));

    const likedVehicles = feedbackArray
      .filter(f => f.feedbackType === 'love' || f.feedbackType === 'like')
      .map(f => vehicles.find(v => v.id === f.vehicleId))
      .filter(Boolean) as Vehicle[];

    onSelectionComplete?.(likedVehicles.slice(0, 3), feedbackArray);
  };

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
                마음에 드는 차량을 골라보세요
              </h1>
              <p className="text-xl text-gray-600 max-w-2xl mx-auto">
                여러 차량을 한눈에 보면서 취향에 맞는 차량들을 선택해주세요.<br/>
                각 차량 카드 아래 버튼으로 선호도를 표시할 수 있어요.
              </p>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-2xl mx-auto">
              <div className="bg-white rounded-2xl p-4 shadow-sm border">
                <div className="w-12 h-12 bg-red-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <Heart className="w-6 h-6 text-red-500" />
                </div>
                <div className="text-sm font-medium text-gray-900">정말 좋아요</div>
                <div className="text-xs text-gray-600">이런 차량 더 보고 싶어요</div>
              </div>

              <div className="bg-white rounded-2xl p-4 shadow-sm border">
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-3">
                  <ThumbsUp className="w-6 h-6 text-blue-500" />
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
                  <ThumbsDown className="w-6 h-6 text-gray-500" />
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
              aria-label="차량 선택 과정 시작하기"
            >
              차량 선택 시작하기
            </Button>
          </div>
        </Container>
      </div>
    );
  }

  if (isLoading) {
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* 헤더 */}
      <div className="bg-white/90 backdrop-blur-md border-b border-gray-200 sticky top-0 z-10">
        <Container>
          <div className="py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl flex items-center justify-center">
                  <Car className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-lg font-bold text-gray-900">차량 선호도 테스트</h1>
                  <p className="text-sm text-gray-600">마음에 드는 차량들을 선택해주세요</p>
                </div>
              </div>

              <div className="flex items-center gap-4">
                <div className="text-center">
                  <div className="text-sm text-gray-500 mb-1">선택한 차량</div>
                  <div className="flex items-center gap-1">
                    <Heart className="w-4 h-4 text-red-500" />
                    <span className="font-bold text-red-500">{getSelectedCount()}</span>
                  </div>
                </div>

                {getSelectedCount() >= 3 && (
                  <Button
                    variant="default"
                    size="sm"
                    onClick={handleComplete}
                    icon={<ArrowRight className="w-4 h-4" aria-hidden="true" />}
                    aria-label="차량 선택 완료하고 다음 단계로 진행"
                  >
                    완료
                  </Button>
                )}
              </div>
            </div>
          </div>
        </Container>
      </div>

      {/* 메인 콘텐츠 - 개선된 그리드 레이아웃 */}
      <Container>
        <div className="py-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
            {vehicles.map((vehicle) => {
              const feedback = vehicleFeedbacks[vehicle.id];

              return (
                <div
                  key={vehicle.id}
                  className={`bg-white rounded-3xl shadow-lg overflow-hidden transition-all duration-300 border-2 ${getFeedbackColor(feedback)} hover:shadow-xl hover:scale-105`}
                >
                  {/* 하이라이트 배지 */}
                  {vehicle.highlight && (
                    <div className="absolute top-4 left-4 z-10">
                      <div className="bg-gradient-to-r from-orange-500 to-red-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                        {vehicle.highlight}
                      </div>
                    </div>
                  )}

                  {/* 매치 점수 */}
                  <div className="absolute top-4 right-4 z-10">
                    <div className="bg-white/90 backdrop-blur-sm rounded-full px-3 py-1 flex items-center gap-1">
                      <TrendingUp className="w-4 h-4 text-green-600" />
                      <span className="text-sm font-bold text-green-600">{vehicle.match_score}%</span>
                    </div>
                  </div>

                  {/* 선택 상태 표시 */}
                  {feedback && (
                    <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 z-20">
                      <div className="w-16 h-16 bg-white rounded-full shadow-xl flex items-center justify-center">
                        {getFeedbackIcon(feedback)}
                      </div>
                    </div>
                  )}

                  {/* 차량 이미지 */}
                  <div className="aspect-[4/3] bg-gradient-to-br from-gray-100 to-gray-200 flex items-center justify-center relative">
                    <Car className="w-16 h-16 text-gray-400" />
                  </div>

                  {/* 차량 정보 */}
                  <div className="p-5">
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <h3 className="text-lg font-bold text-gray-900 mb-1">
                          {vehicle.brand} {vehicle.model}
                        </h3>
                        <p className="text-sm text-gray-600">{vehicle.year}년 • {vehicle.location}</p>
                      </div>
                      <div className="text-right">
                        <div className="text-xl font-bold text-blue-600 mb-1">
                          {vehicle.price.toLocaleString()}만원
                        </div>
                        <div className="flex items-center gap-1 text-xs text-gray-500">
                          <Star className="w-3 h-3 text-yellow-500" />
                          <span>{vehicle.safety_rating}.0</span>
                        </div>
                      </div>
                    </div>

                    {/* 핵심 스펙 */}
                    <div className="grid grid-cols-3 gap-3 mb-4 text-xs">
                      <div className="flex items-center gap-1 text-gray-600">
                        <Fuel className="w-3 h-3" />
                        <span>{vehicle.fuel_type}</span>
                      </div>
                      <div className="flex items-center gap-1 text-gray-600">
                        <MapPin className="w-3 h-3" />
                        <span>{(vehicle.mileage / 10000).toFixed(1)}만km</span>
                      </div>
                      {vehicle.fuel_efficiency > 0 && (
                        <div className="flex items-center gap-1 text-green-600">
                          <Zap className="w-3 h-3" />
                          <span>{vehicle.fuel_efficiency}km/L</span>
                        </div>
                      )}
                    </div>

                    {/* 주요 특징 태그 */}
                    <div className="flex flex-wrap gap-1 mb-4">
                      {vehicle.features.slice(0, 3).map((feature, idx) => (
                        <span
                          key={idx}
                          className="inline-block px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded-lg font-medium"
                        >
                          {feature}
                        </span>
                      ))}
                    </div>

                    {/* 상세 설명 */}
                    <p className="text-sm text-gray-700 mb-4 line-clamp-3 leading-relaxed">
                      {vehicle.description}
                    </p>

                    {/* 액션 버튼들 */}
                    <div className="grid grid-cols-4 gap-2">
                      <button
                        onClick={() => handleVehicleFeedback(vehicle.id, 'dislike')}
                        className={`p-3 rounded-xl border-2 transition-all duration-200 ${
                          feedback === 'dislike'
                            ? 'border-gray-400 bg-gray-100'
                            : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                        }`}
                        title="별로예요"
                      >
                        <ThumbsDown className="w-4 h-4 text-gray-500 mx-auto" />
                      </button>

                      <button
                        onClick={() => handleVehicleFeedback(vehicle.id, 'expensive')}
                        className={`p-3 rounded-xl border-2 transition-all duration-200 ${
                          feedback === 'expensive'
                            ? 'border-orange-400 bg-orange-100'
                            : 'border-gray-200 hover:border-orange-300 hover:bg-orange-50'
                        }`}
                        title="비싸요"
                      >
                        <DollarSign className="w-4 h-4 text-orange-500 mx-auto" />
                      </button>

                      <button
                        onClick={() => handleVehicleFeedback(vehicle.id, 'like')}
                        className={`p-3 rounded-xl border-2 transition-all duration-200 ${
                          feedback === 'like'
                            ? 'border-blue-400 bg-blue-100'
                            : 'border-gray-200 hover:border-blue-300 hover:bg-blue-50'
                        }`}
                        title="괜찮아요"
                      >
                        <ThumbsUp className="w-4 h-4 text-blue-500 mx-auto" />
                      </button>

                      <button
                        onClick={() => handleVehicleFeedback(vehicle.id, 'love')}
                        className={`p-3 rounded-xl border-2 transition-all duration-200 ${
                          feedback === 'love'
                            ? 'border-red-400 bg-red-100'
                            : 'border-gray-200 hover:border-red-300 hover:bg-red-50'
                        }`}
                        title="정말 좋아요"
                      >
                        <Heart className="w-4 h-4 text-red-500 mx-auto" />
                      </button>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* 하단 완료 버튼 */}
          {getSelectedCount() >= 3 && (
            <div className="fixed bottom-8 left-1/2 transform -translate-x-1/2 z-20">
              <Button
                variant="default"
                size="lg"
                onClick={handleComplete}
                icon={<ArrowRight className="w-6 h-6" aria-hidden="true" />}
                className="shadow-2xl shadow-blue-300"
                aria-label={`${getSelectedCount()}개 차량 선택을 완료하고 분석 단계로 진행하기`}
              >
                {getSelectedCount()}개 차량 선택 완료
              </Button>
            </div>
          )}

          {/* 안내 메시지 */}
          {getSelectedCount() < 3 && (
            <div className="text-center mt-12">
              <p className="text-gray-600 text-lg">
                최소 3개 이상의 차량을 선택해주세요 ({getSelectedCount()}/3)
              </p>
              <p className="text-gray-500 text-sm mt-2">
                더 정확한 분석을 위해 다양한 차량에 반응을 보여주세요
              </p>
            </div>
          )}
        </div>
      </Container>
    </div>
  );
}