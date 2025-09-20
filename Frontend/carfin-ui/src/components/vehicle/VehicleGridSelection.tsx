'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import {
  Car,
  Heart,
  X,
  Fuel,
  Gauge,
  Calendar,
  MapPin,
  ArrowRight,
  CheckCircle,
  TrendingUp,
  ThumbsUp,
  ThumbsDown,
  DollarSign,
  Star,
  Flame,
  MessageCircle,
  FilterX,
  Zap
} from 'lucide-react';

type FeedbackType = 'like' | 'dislike' | 'expensive' | 'maybe' | 'love' | 'question';

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
}

interface VehicleFeedback {
  vehicleId: string;
  feedbackType: FeedbackType;
  timestamp: Date;
}

type FilteringPhase = 'initial' | 'narrowing' | 'final';

interface VehicleGridSelectionProps {
  userProfile?: any;
  onSelectionComplete?: (selectedVehicles: Vehicle[], feedback: VehicleFeedback[]) => void;
}

export function VehicleGridSelection({ userProfile, onSelectionComplete }: VehicleGridSelectionProps) {
  const [allVehicles, setAllVehicles] = useState<Vehicle[]>([]);
  const [displayedVehicles, setDisplayedVehicles] = useState<Vehicle[]>([]);
  const [vehicleFeedbacks, setVehicleFeedbacks] = useState<VehicleFeedback[]>([]);
  const [currentPhase, setCurrentPhase] = useState<FilteringPhase>('initial');
  const [isLoading, setIsLoading] = useState(true);
  const [phaseMessage, setPhaseMessage] = useState('');

  useEffect(() => {
    loadInitialVehicles();
  }, []);

  const loadInitialVehicles = async () => {
    try {
      setIsLoading(true);
      setPhaseMessage('AI가 당신에게 맞는 차량들을 찾고 있어요...');

      const mockVehicles = getExpandedMockVehicles();
      setAllVehicles(mockVehicles);
      setDisplayedVehicles(mockVehicles.slice(0, 12)); // 처음에는 12개 표시
      setCurrentPhase('initial');
      setPhaseMessage('마음에 드는 차량들에 반응을 보여주세요! 👍👎💰');
    } catch (error) {
      console.error('Failed to load vehicles:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const getExpandedMockVehicles = (): Vehicle[] => [
    {
      id: "1", brand: "현대", model: "아반떼", year: 2022, price: 2800, mileage: 15000,
      fuel_type: "가솔린", body_type: "세단", color: "화이트", location: "서울",
      images: ["/api/placeholder/400/300"], features: ["스마트크루즈", "후방카메라", "블루투스"],
      fuel_efficiency: 14.5, safety_rating: 5, match_score: 92,
      description: "경제적이고 실용적인 첫차 추천"
    },
    {
      id: "2", brand: "기아", model: "K5", year: 2021, price: 3200, mileage: 25000,
      fuel_type: "하이브리드", body_type: "세단", color: "블랙", location: "인천",
      images: ["/api/placeholder/400/300"], features: ["하이브리드", "선루프", "통풍시트"],
      fuel_efficiency: 16.8, safety_rating: 5, match_score: 88,
      description: "연비 좋은 하이브리드 중형차"
    },
    {
      id: "3", brand: "제네시스", model: "G70", year: 2023, price: 4200, mileage: 8000,
      fuel_type: "가솔린", body_type: "세단", color: "그레이", location: "경기",
      images: ["/api/placeholder/400/300"], features: ["가죽시트", "프리미엄사운드", "어댑티브크루즈"],
      fuel_efficiency: 11.5, safety_rating: 5, match_score: 84,
      description: "프리미엄 브랜드의 럭셔리 세단"
    },
    {
      id: "4", brand: "현대", model: "투싼", year: 2022, price: 3500, mileage: 18000,
      fuel_type: "가솔린", body_type: "SUV", color: "화이트", location: "서울",
      images: ["/api/placeholder/400/300"], features: ["전동시트", "파노라마선루프", "스마트키"],
      fuel_efficiency: 12.8, safety_rating: 5, match_score: 89,
      description: "가족용으로 완벽한 중형 SUV"
    },
    {
      id: "5", brand: "기아", model: "쏘렌토", year: 2021, price: 3900, mileage: 22000,
      fuel_type: "디젤", body_type: "SUV", color: "블루", location: "경기",
      images: ["/api/placeholder/400/300"], features: ["7인승", "사이드에어백", "차선유지"],
      fuel_efficiency: 15.2, safety_rating: 5, match_score: 86,
      description: "넓은 공간의 대형 SUV"
    },
    {
      id: "6", brand: "BMW", model: "320i", year: 2020, price: 3800, mileage: 28000,
      fuel_type: "가솔린", body_type: "세단", color: "블랙", location: "서울",
      images: ["/api/placeholder/400/300"], features: ["런플랫타이어", "하만카돈", "무선충전"],
      fuel_efficiency: 13.1, safety_rating: 5, match_score: 81,
      description: "독일 프리미엄의 대표 세단"
    },
    {
      id: "7", brand: "메르세데스벤츠", model: "C200", year: 2021, price: 4500, mileage: 19000,
      fuel_type: "가솔린", body_type: "세단", color: "실버", location: "인천",
      images: ["/api/placeholder/400/300"], features: ["AMG라인", "부르메스터", "에어매틱"],
      fuel_efficiency: 12.5, safety_rating: 5, match_score: 79,
      description: "럭셔리의 정점을 보여주는 세단"
    },
    {
      id: "8", brand: "현대", model: "그랜저", year: 2022, price: 4100, mileage: 16000,
      fuel_type: "하이브리드", body_type: "세단", color: "화이트", location: "경기",
      images: ["/api/placeholder/400/300"], features: ["마사지시트", "디스플레이오디오", "V2L"],
      fuel_efficiency: 15.9, safety_rating: 5, match_score: 87,
      description: "한국 최고급 하이브리드 세단"
    },
    {
      id: "9", brand: "기아", model: "스포티지", year: 2023, price: 3300, mileage: 12000,
      fuel_type: "가솔린", body_type: "SUV", color: "레드", location: "서울",
      images: ["/api/placeholder/400/300"], features: ["HUD", "무선애플카플레이", "후석모니터"],
      fuel_efficiency: 13.2, safety_rating: 5, match_score: 90,
      description: "스타일리시한 중형 SUV"
    },
    {
      id: "10", brand: "현대", model: "코나", year: 2022, price: 2900, mileage: 21000,
      fuel_type: "가솔린", body_type: "SUV", color: "옐로우", location: "인천",
      images: ["/api/placeholder/400/300"], features: ["헤드업디스플레이", "빌트인캠", "스마트파킹"],
      fuel_efficiency: 14.1, safety_rating: 4, match_score: 85,
      description: "개성있는 디자인의 소형 SUV"
    },
    {
      id: "11", brand: "테슬라", model: "모델3", year: 2021, price: 4800, mileage: 15000,
      fuel_type: "전기", body_type: "세단", color: "화이트", location: "서울",
      images: ["/api/placeholder/400/300"], features: ["오토파일럿", "슈퍼차징", "OTA업데이트"],
      fuel_efficiency: 0, safety_rating: 5, match_score: 78,
      description: "미래를 앞서가는 전기차"
    },
    {
      id: "12", brand: "아우디", model: "A4", year: 2020, price: 4000, mileage: 24000,
      fuel_type: "가솔린", body_type: "세단", color: "그레이", location: "경기",
      images: ["/api/placeholder/400/300"], features: ["콰트로", "버추얼콕핏", "뱅앤올룹슨"],
      fuel_efficiency: 12.9, safety_rating: 5, match_score: 82,
      description: "독일 기술력의 중형 세단"
    }
  ];

  const handleVehicleFeedback = (vehicleId: string, feedbackType: FeedbackType) => {
    const newFeedback: VehicleFeedback = {
      vehicleId,
      feedbackType,
      timestamp: new Date()
    };

    const updatedFeedbacks = [...vehicleFeedbacks, newFeedback];
    setVehicleFeedbacks(updatedFeedbacks);

    // 진행 단계 체크
    checkPhaseProgression(updatedFeedbacks);
  };

  const checkPhaseProgression = (feedbacks: VehicleFeedback[]) => {
    const totalFeedbacks = feedbacks.length;
    const likedVehicles = feedbacks.filter(f => f.feedbackType === 'like' || f.feedbackType === 'love').length;

    if (currentPhase === 'initial' && totalFeedbacks >= 6) {
      // Phase 1 → 2: 좁히기
      progressToNarrowing();
    } else if (currentPhase === 'narrowing' && likedVehicles >= 3) {
      // Phase 2 → 3: 최종 선택
      progressToFinal();
    }
  };

  const progressToNarrowing = () => {
    setCurrentPhase('narrowing');
    setPhaseMessage('좋은 선택이에요! 이제 더 정확한 추천을 위해 범위를 좁혀볼게요');

    // 피드백 기반으로 7개로 줄이기
    const likedIds = vehicleFeedbacks
      .filter(f => f.feedbackType === 'like' || f.feedbackType === 'love')
      .map(f => f.vehicleId);

    const dislikedIds = vehicleFeedbacks
      .filter(f => f.feedbackType === 'dislike' || f.feedbackType === 'expensive')
      .map(f => f.vehicleId);

    let filteredVehicles = allVehicles.filter(v => !dislikedIds.includes(v.id));

    // 좋아한 차량 우선, 나머지는 매치스코어 순
    const prioritizedVehicles = [
      ...filteredVehicles.filter(v => likedIds.includes(v.id)),
      ...filteredVehicles
        .filter(v => !likedIds.includes(v.id))
        .sort((a, b) => b.match_score - a.match_score)
    ].slice(0, 7);

    setDisplayedVehicles(prioritizedVehicles);
  };

  const progressToFinal = () => {
    setCurrentPhase('final');
    setPhaseMessage('완벽해요! 최종 추천 3개를 준비했습니다');

    const likedIds = vehicleFeedbacks
      .filter(f => f.feedbackType === 'like' || f.feedbackType === 'love')
      .map(f => f.vehicleId);

    const finalVehicles = allVehicles
      .filter(v => likedIds.includes(v.id))
      .sort((a, b) => b.match_score - a.match_score)
      .slice(0, 3);

    setDisplayedVehicles(finalVehicles);

    // 1.5초 후 자동으로 다음 단계로
    setTimeout(() => {
      onSelectionComplete?.(finalVehicles, vehicleFeedbacks);
    }, 1500);
  };

  const getFeedbackIcon = (type: FeedbackType) => {
    switch (type) {
      case 'like': return <ThumbsUp className="w-4 h-4" />;
      case 'dislike': return <ThumbsDown className="w-4 h-4" />;
      case 'expensive': return <DollarSign className="w-4 h-4" />;
      case 'maybe': return <Star className="w-4 h-4" />;
      case 'love': return <Flame className="w-4 h-4" />;
      case 'question': return <MessageCircle className="w-4 h-4" />;
      default: return <Heart className="w-4 h-4" />;
    }
  };

  const getFeedbackColor = (type: FeedbackType) => {
    switch (type) {
      case 'like': return 'bg-blue-500 hover:bg-blue-600';
      case 'dislike': return 'bg-red-500 hover:bg-red-600';
      case 'expensive': return 'bg-orange-500 hover:bg-orange-600';
      case 'maybe': return 'bg-yellow-500 hover:bg-yellow-600';
      case 'love': return 'bg-pink-500 hover:bg-pink-600';
      case 'question': return 'bg-purple-500 hover:bg-purple-600';
      default: return 'bg-gray-500 hover:bg-gray-600';
    }
  };

  const getVehicleFeedback = (vehicleId: string) => {
    return vehicleFeedbacks.find(f => f.vehicleId === vehicleId);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Car className="w-8 h-8 text-blue-600 animate-pulse" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">차량을 선별하고 있어요</h2>
          <p className="text-gray-600">{phaseMessage}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 헤더 */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                {currentPhase === 'initial' && '차량 탐색'}
                {currentPhase === 'narrowing' && '범위 좁히기'}
                {currentPhase === 'final' && '최종 추천'}
              </h1>
              <p className="text-blue-600 font-medium">{phaseMessage}</p>
            </div>
            <div className="flex items-center gap-4">
              <div className="text-sm text-gray-500">
                단계 {currentPhase === 'initial' ? '1' : currentPhase === 'narrowing' ? '2' : '3'}/3
              </div>
              <div className="flex gap-1">
                <div className={`w-2 h-2 rounded-full ${currentPhase !== 'initial' ? 'bg-blue-600' : 'bg-gray-300'}`} />
                <div className={`w-2 h-2 rounded-full ${currentPhase === 'final' ? 'bg-blue-600' : currentPhase === 'narrowing' ? 'bg-blue-600' : 'bg-gray-300'}`} />
                <div className={`w-2 h-2 rounded-full ${currentPhase === 'final' ? 'bg-blue-600' : 'bg-gray-300'}`} />
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 차량 그리드 */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className={`grid gap-6 ${
          currentPhase === 'final'
            ? 'grid-cols-1 md:grid-cols-3'
            : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4'
        }`}>
          {displayedVehicles.map((vehicle) => {
            const feedback = getVehicleFeedback(vehicle.id);
            const hasFeedback = !!feedback;

            return (
              <div
                key={vehicle.id}
                className={`bg-white rounded-xl shadow-sm border-2 overflow-hidden transition-all duration-300 ${
                  hasFeedback
                    ? 'border-blue-300 shadow-lg transform scale-105'
                    : 'border-gray-200 hover:shadow-md'
                } ${currentPhase === 'final' ? 'lg:aspect-auto' : ''}`}
              >
                {/* 피드백 표시 */}
                {hasFeedback && (
                  <div className="bg-blue-50 px-4 py-2 flex items-center gap-2">
                    {getFeedbackIcon(feedback.feedbackType)}
                    <span className="text-sm font-medium text-blue-700">
                      {feedback.feedbackType === 'like' && '관심있어요'}
                      {feedback.feedbackType === 'love' && '정말 좋아요'}
                      {feedback.feedbackType === 'dislike' && '별로에요'}
                      {feedback.feedbackType === 'expensive' && '비싸요'}
                      {feedback.feedbackType === 'maybe' && '나중에 봐요'}
                      {feedback.feedbackType === 'question' && '질문있어요'}
                    </span>
                  </div>
                )}

                {/* 차량 이미지 */}
                <div className="aspect-video bg-gray-100 flex items-center justify-center relative">
                  <Car className="w-12 h-12 text-gray-400" />
                  <div className="absolute top-2 right-2 bg-green-500 text-white px-2 py-1 rounded-full text-xs font-medium">
                    매치 {vehicle.match_score}%
                  </div>
                </div>

                {/* 차량 정보 */}
                <div className="p-4">
                  <div className="mb-3">
                    <h3 className="font-bold text-lg text-gray-900">
                      {vehicle.brand} {vehicle.model}
                    </h3>
                    <p className="text-gray-600 text-sm">{vehicle.year}년 • {vehicle.color}</p>
                  </div>

                  <div className="mb-3">
                    <div className="text-xl font-bold text-blue-600 mb-1">
                      {vehicle.price.toLocaleString()}만원
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-2 mb-3 text-sm text-gray-600">
                    <div className="flex items-center gap-1">
                      <Gauge className="w-3 h-3" />
                      <span>{vehicle.mileage.toLocaleString()}km</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Fuel className="w-3 h-3" />
                      <span>{vehicle.fuel_type}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <MapPin className="w-3 h-3" />
                      <span>{vehicle.location}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <Zap className="w-3 h-3" />
                      <span>{vehicle.fuel_efficiency}km/L</span>
                    </div>
                  </div>

                  {/* 특징 태그 */}
                  <div className="flex flex-wrap gap-1 mb-4">
                    {vehicle.features.slice(0, 2).map((feature, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded"
                      >
                        {feature}
                      </span>
                    ))}
                  </div>

                  {/* 피드백 버튼들 */}
                  {!hasFeedback && (
                    <div className="grid grid-cols-3 gap-1">
                      <button
                        onClick={() => handleVehicleFeedback(vehicle.id, 'like')}
                        className="p-2 bg-blue-500 hover:bg-blue-600 text-white rounded text-xs transition-colors"
                        title="관심있어요"
                      >
                        👍
                      </button>
                      <button
                        onClick={() => handleVehicleFeedback(vehicle.id, 'expensive')}
                        className="p-2 bg-orange-500 hover:bg-orange-600 text-white rounded text-xs transition-colors"
                        title="비싸요"
                      >
                        💰
                      </button>
                      <button
                        onClick={() => handleVehicleFeedback(vehicle.id, 'dislike')}
                        className="p-2 bg-red-500 hover:bg-red-600 text-white rounded text-xs transition-colors"
                        title="별로에요"
                      >
                        👎
                      </button>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* 진행 상황 */}
        {currentPhase !== 'final' && (
          <div className="mt-8 text-center">
            <p className="text-gray-600">
              {vehicleFeedbacks.length}개 차량에 반응했어요 •
              {currentPhase === 'initial'
                ? ` ${Math.max(0, 6 - vehicleFeedbacks.length)}개 더 선택해보세요!`
                : ` 곧 최종 추천을 받게 됩니다!`
              }
            </p>
          </div>
        )}

        {currentPhase === 'final' && (
          <div className="mt-8 text-center">
            <div className="inline-flex items-center gap-2 bg-green-100 text-green-700 px-4 py-2 rounded-full">
              <CheckCircle className="w-5 h-5" />
              <span className="font-medium">최종 추천이 완료되었어요!</span>
            </div>
            <p className="text-gray-600 mt-2">상세 분석 결과를 준비하고 있습니다...</p>
          </div>
        )}
      </div>
    </div>
  );
}