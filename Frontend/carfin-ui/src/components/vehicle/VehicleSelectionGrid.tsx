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
  FilterX
} from 'lucide-react';
import { GeminiMultiAgent, type Vehicle, type UserProfile } from '@/lib/gemini-agents';

type FeedbackType = 'like' | 'dislike' | 'expensive' | 'maybe' | 'love' | 'question';

interface VehicleFeedback {
  vehicleId: string;
  feedbackType: FeedbackType;
  timestamp: Date;
  reasons?: string[];
}

interface UserFeedback {
  vehicleId: string;
  action: 'like' | 'dislike' | 'maybe';
  reasons?: string[];
}

type FilteringPhase = 'initial' | 'narrowing' | 'final';

interface VehicleSelectionGridProps {
  userProfile?: Partial<UserProfile>;
  onSelectionComplete?: (selectedVehicle: Vehicle, feedback: UserFeedback[]) => void;
}

export function VehicleSelectionGrid({ userProfile, onSelectionComplete }: VehicleSelectionGridProps) {
  const [allVehicles, setAllVehicles] = useState<Vehicle[]>([]);
  const [displayedVehicles, setDisplayedVehicles] = useState<Vehicle[]>([]);
  const [vehicleFeedbacks, setVehicleFeedbacks] = useState<VehicleFeedback[]>([]);
  const [currentPhase, setCurrentPhase] = useState<FilteringPhase>('initial');
  const [isLoading, setIsLoading] = useState(true);
  const [phaseMessage, setPhaseMessage] = useState('');
  const geminiAgent = new GeminiMultiAgent(process.env.NEXT_PUBLIC_GEMINI_API_KEY || '');

  useEffect(() => {
    loadInitialVehicles();
  }, []);

  const loadInitialVehicles = async () => {
    try {
      setIsLoading(true);
      setPhaseMessage('AI가 당신에게 맞는 차량들을 찾고 있어요...');

      const initialVehicles = await geminiAgent.realTimeCollaborativeFiltering(userProfile || {});
      const mockVehicles = getExpandedMockVehicles();

      setAllVehicles(mockVehicles);
      setDisplayedVehicles(mockVehicles.slice(0, 10)); // 처음에는 10개만 표시
      setCurrentPhase('initial');
      setPhaseMessage('마음에 드는 차량들에 반응을 보여주세요! 👍👎💰');
    } catch (error) {
      console.error('Failed to load vehicles:', error);
      const mockVehicles = getExpandedMockVehicles();
      setAllVehicles(mockVehicles);
      setDisplayedVehicles(mockVehicles.slice(0, 10));
      setPhaseMessage('마음에 드는 차량들에 반응을 보여주세요! 👍👎💰');
    } finally {
      setIsLoading(false);
    }
  };

  const getExpandedMockVehicles = (): Vehicle[] => [
    {
      id: "1",
      brand: "현대",
      model: "아반떼",
      year: 2022,
      price: 2800,
      mileage: 15000,
      fuel_type: "가솔린",
      body_type: "세단",
      color: "화이트",
      location: "서울",
      images: ["/api/placeholder/400/300"],
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
      year: 2021,
      price: 3200,
      mileage: 25000,
      fuel_type: "하이브리드",
      body_type: "세단",
      color: "블랙",
      location: "인천",
      images: ["/api/placeholder/400/300"],
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
      images: ["/api/placeholder/400/300"],
      features: ["가죽시트", "프리미엄사운드", "어댑티브크루즈"],
      fuel_efficiency: 11.5,
      safety_rating: 5,
      match_score: 84,
      description: "프리미엄 브랜드의 럭셔리 세단"
    }
  ];

  // 이 컴포넌트는 사용되지 않음 - VehicleGridSelection이 대신 사용됨
  // 모든 기능이 VehicleGridSelection으로 이동되었음

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <Car className="w-8 h-8 text-blue-600 animate-pulse" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">차량을 선별하고 있어요</h2>
          <p className="text-gray-600">AI가 당신에게 맞는 차량들을 찾고 있습니다...</p>
        </div>
      </div>
    );
  }

  if (!currentVehicle) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="w-8 h-8 text-green-600" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">선택 완료!</h2>
          <p className="text-gray-600">분석 결과를 준비하고 있습니다...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 헤더 */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-4xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">차량 선택</h1>
              <p className="text-gray-600">마음에 드는 차량에 하트를, 그렇지 않은 차량에 X를 눌러주세요</p>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-500">진행률</div>
              <div className="text-lg font-bold text-blue-600">{currentIndex + 1}/{vehicles.length}</div>
            </div>
          </div>

          {/* 진행바 */}
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${learningProgress}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* 메인 콘텐츠 */}
      <div className="max-w-2xl mx-auto px-6 py-8">
        {/* 차량 카드 */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 overflow-hidden mb-6">
          {/* 차량 이미지 */}
          <div className="aspect-video bg-gray-100 flex items-center justify-center">
            <Car className="w-16 h-16 text-gray-400" />
          </div>

          {/* 차량 정보 */}
          <div className="p-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-1">
                  {currentVehicle.brand} {currentVehicle.model}
                </h2>
                <p className="text-gray-600">{currentVehicle.year}년식</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-bold text-blue-600 mb-1">
                  {currentVehicle.price.toLocaleString()}만원
                </div>
                <div className="flex items-center gap-1 text-sm text-green-600">
                  <TrendingUp className="w-4 h-4" />
                  매치율 {currentVehicle.match_score}%
                </div>
              </div>
            </div>

            {/* 차량 세부 정보 */}
            <div className="grid grid-cols-2 gap-4 mb-6">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Gauge className="w-4 h-4" />
                <span>{currentVehicle.mileage.toLocaleString()}km</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Fuel className="w-4 h-4" />
                <span>{currentVehicle.fuel_type}</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Calendar className="w-4 h-4" />
                <span>{currentVehicle.year}년</span>
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <MapPin className="w-4 h-4" />
                <span>{currentVehicle.location}</span>
              </div>
            </div>

            {/* 특징 */}
            <div className="mb-6">
              <h3 className="font-semibold text-gray-900 mb-2">주요 특징</h3>
              <div className="flex flex-wrap gap-2">
                {currentVehicle.features.map((feature, index) => (
                  <span
                    key={index}
                    className="px-3 py-1 bg-blue-50 text-blue-700 text-sm rounded-full"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            </div>

            {/* 설명 */}
            <p className="text-gray-600 mb-6">{currentVehicle.description}</p>
          </div>
        </div>

        {/* 선택 버튼 */}
        <div className="flex gap-4">
          <Button
            onClick={() => handleFeedback('dislike')}
            variant="outline"
            size="lg"
            className="flex-1 h-14 border-red-200 text-red-600 hover:bg-red-50 hover:border-red-300"
          >
            <X className="w-6 h-6 mr-2" />
            관심 없음
          </Button>

          <Button
            onClick={() => handleFeedback('maybe')}
            variant="outline"
            size="lg"
            className="flex-1 h-14 border-gray-300 text-gray-700 hover:bg-gray-50"
          >
            나중에
          </Button>

          <Button
            onClick={() => handleFeedback('like')}
            size="lg"
            className="flex-1 h-14 bg-blue-600 hover:bg-blue-700 text-white"
          >
            <Heart className="w-6 h-6 mr-2" />
            마음에 들어요
          </Button>
        </div>

        {/* 진행 상태 */}
        <div className="mt-6 text-center text-sm text-gray-500">
          {vehicles.length - currentIndex - 1}개의 차량이 더 남았어요
        </div>
      </div>
    </div>
  );
}