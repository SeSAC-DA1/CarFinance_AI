'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Car,
  Fuel,
  Calendar,
  DollarSign,
  Heart,
  GitCompare,
  CreditCard,
  Star,
  MapPin,
  Settings,
  Shield,
  Sparkles,
  TrendingUp,
  Users,
  Award,
  Clock,
  Target
} from 'lucide-react';

// 추천 시스템 import
import { mockRecommendationEngine } from '@/lib/recommendation/mock-recommendation-engine';
import { getUserBehaviorTracker } from '@/lib/analytics/user-behavior-tracker';
import { RecommendationResponse, UserProfile as RecommendationUserProfile } from '@/types/recommendation';
import { MOCK_CARS } from '@/lib/data/mock-data';

// 기존 인터페이스들 (호환성 유지)
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
  // 추천 관련 추가 필드
  recommendation_reason?: string;
  recommendation_type?: 'collaborative' | 'content' | 'popular' | 'similar';
  confidence_score?: number;
}

interface UserProfile {
  user_id: string;
  name: string;
  email: string;
  age: number;
  income: number;
  preferences: string[];
  purpose: string;
}

interface RecommendationVehicleGridProps {
  userProfile: UserProfile;
  onVehicleSelect: (vehicleId: string) => void;
  onRequestFinancing: (vehicleId: string) => void;
  selectedVehicles: string[];
  recommendationType?: 'homepage' | 'personalized' | 'similar';
  currentVehicleId?: string; // 유사 차량 추천용
  limit?: number;
}

export function RecommendationVehicleGrid({
  userProfile,
  onVehicleSelect,
  onRequestFinancing,
  selectedVehicles,
  recommendationType = 'personalized',
  currentVehicleId,
  limit = 12
}: RecommendationVehicleGridProps) {
  const [vehicles, setVehicles] = useState<Vehicle[]>([]);
  const [favoriteVehicles, setFavoriteVehicles] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [recommendationData, setRecommendationData] = useState<RecommendationResponse | null>(null);
  const [behaviorTracker] = useState(() => getUserBehaviorTracker(userProfile.user_id));

  // 📚 학습 포인트: 컴포넌트 마운트 시 추천 데이터 로드
  useEffect(() => {
    loadRecommendations();
  }, [userProfile.user_id, recommendationType, currentVehicleId]);

  const loadRecommendations = async () => {
    setLoading(true);

    try {
      console.log('🔄 추천 데이터 로딩 중...', { recommendationType, currentVehicleId });

      // UserProfile을 RecommendationUserProfile로 변환
      const recommendationUserProfile: RecommendationUserProfile = {
        user_id: userProfile.user_id,
        name: userProfile.name,
        email: userProfile.email,
        age: userProfile.age,
        income: userProfile.income,
        purpose: userProfile.purpose as any,
        preferences: userProfile.preferences,
        budgetRange: {
          min: Math.max(1000, userProfile.income * 0.3 * 10), // 연소득의 30%를 최소 예산으로
          max: userProfile.income * 0.8 * 10 // 연소득의 80%를 최대 예산으로
        },
        createdAt: new Date(),
        lastActiveAt: new Date()
      };

      // 추천 엔진 호출
      const response = await mockRecommendationEngine.getRecommendations({
        userId: userProfile.user_id,
        userProfile: recommendationUserProfile,
        context: {
          type: recommendationType,
          currentVehicleId,
          limit
        },
        excludeVehicleIds: selectedVehicles
      });

      setRecommendationData(response);

      // 추천 결과를 Vehicle 형태로 변환
      const recommendedVehicles: Vehicle[] = response.recommendations.map(rec => {
        const mockCar = MOCK_CARS.find(car => car.id === rec.vehicleId);
        if (!mockCar) return null;

        return {
          id: mockCar.id,
          brand: mockCar.brand,
          model: mockCar.model,
          year: mockCar.year,
          price: mockCar.price,
          mileage: mockCar.mileage,
          fuel_type: mockCar.fuelType,
          body_type: mockCar.bodyType,
          color: '기본색상',
          location: mockCar.location,
          images: mockCar.images,
          features: mockCar.features,
          fuel_efficiency: mockCar.normalized_features.fuel_efficiency,
          safety_rating: Math.floor(mockCar.semantic_features.reliability_score / 2),
          match_score: Math.round(rec.score * 100),
          description: `${rec.reasons[0]?.description || '추천 차량'}`,
          // 추천 관련 메타데이터
          recommendation_reason: rec.reasons[0]?.description,
          recommendation_type: rec.scores.collaborative > rec.scores.contentBased ? 'collaborative' : 'content',
          confidence_score: rec.confidence
        };
      }).filter(Boolean) as Vehicle[];

      setVehicles(recommendedVehicles);

      console.log('✅ 추천 완료:', {
        count: recommendedVehicles.length,
        processingTime: `${response.metadata.processingTime}ms`,
        modelUsed: response.metadata.modelUsed
      });

    } catch (error) {
      console.error('❌ 추천 로딩 실패:', error);
      // 폴백: 기본 차량들 표시
      setVehicles(MOCK_CARS.slice(0, limit).map(car => ({
        id: car.id,
        brand: car.brand,
        model: car.model,
        year: car.year,
        price: car.price,
        mileage: car.mileage,
        fuel_type: car.fuelType,
        body_type: car.bodyType,
        color: '기본색상',
        location: car.location,
        images: car.images,
        features: car.features,
        fuel_efficiency: car.normalized_features.fuel_efficiency,
        safety_rating: Math.floor(car.semantic_features.reliability_score / 2),
        match_score: 50,
        description: '기본 추천 차량'
      })));
    } finally {
      setLoading(false);
    }
  };

  // 📚 학습 포인트: 사용자 행동 추적 통합
  const handleVehicleInteraction = (vehicleId: string, interactionType: 'click' | 'save' | 'view') => {
    console.log('👆 사용자 상호작용:', { vehicleId, interactionType });

    // 행동 추적기에 기록
    switch (interactionType) {
      case 'view':
        behaviorTracker.trackVehicleView(vehicleId, {
          source: 'recommendation',
          position: vehicles.findIndex(v => v.id === vehicleId) + 1
        });
        break;
      case 'save':
        behaviorTracker.trackLike(vehicleId);
        break;
      case 'click':
        // 일반 클릭도 추적
        break;
    }
  };

  const toggleFavorite = (vehicleId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setFavoriteVehicles(prev =>
      prev.includes(vehicleId)
        ? prev.filter(id => id !== vehicleId)
        : [...prev, vehicleId]
    );
    handleVehicleInteraction(vehicleId, 'save');
  };

  const handleVehicleClick = (vehicleId: string) => {
    handleVehicleInteraction(vehicleId, 'view');
    onVehicleSelect(vehicleId);
  };

  const getFuelTypeInfo = (fuelType: string) => {
    const fuelMap: Record<string, { label: string; color: string; emoji: string }> = {
      gasoline: { label: '가솔린', color: 'bg-blue-100 text-blue-800', emoji: '⛽' },
      diesel: { label: '디젤', color: 'bg-green-100 text-green-800', emoji: '🛢️' },
      hybrid: { label: '하이브리드', color: 'bg-emerald-100 text-emerald-800', emoji: '🔋' },
      electric: { label: '전기', color: 'bg-purple-100 text-purple-800', emoji: '⚡' },
    };
    return fuelMap[fuelType] || { label: fuelType, color: 'bg-gray-100 text-gray-800', emoji: '🚗' };
  };

  const getMatchScoreColor = (score: number) => {
    if (score >= 90) return 'from-emerald-500 to-green-500';
    if (score >= 80) return 'from-purple-500 to-pink-500';
    if (score >= 70) return 'from-yellow-500 to-orange-500';
    return 'from-gray-500 to-slate-500';
  };

  const getRecommendationTypeIcon = (type?: string) => {
    switch (type) {
      case 'collaborative':
        return <Users className="w-4 h-4" />;
      case 'content':
        return <Target className="w-4 h-4" />;
      case 'popular':
        return <TrendingUp className="w-4 h-4" />;
      case 'similar':
        return <GitCompare className="w-4 h-4" />;
      default:
        return <Sparkles className="w-4 h-4" />;
    }
  };

  const getRecommendationTypeLabel = (type?: string) => {
    switch (type) {
      case 'collaborative':
        return '유사한 사용자';
      case 'content':
        return '맞춤 선호도';
      case 'popular':
        return '인기 차량';
      case 'similar':
        return '유사 차량';
      default:
        return 'AI 추천';
    }
  };

  const getRecommendationTitle = () => {
    switch (recommendationType) {
      case 'homepage':
        return '🏠 홈페이지 추천';
      case 'similar':
        return '🔄 유사 차량';
      case 'personalized':
        return '🎯 맞춤 추천';
      default:
        return '🚗 AI 차량 추천';
    }
  };

  if (loading) {
    return (
      <div className="space-y-8">
        <div className="text-center">
          <div className="animate-spin w-8 h-8 border-4 border-purple-500 border-t-transparent rounded-full mx-auto mb-4"></div>
          <h2 className="text-2xl font-bold text-gray-800">AI가 당신에게 맞는 차량을 찾고 있습니다...</h2>
          <p className="text-gray-600">잠시만 기다려주세요</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* 추천 결과 헤더 */}
      <div className="text-center">
        <h2 className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
          {getRecommendationTitle()}
        </h2>
        <p className="text-gray-600 text-lg">
          총 {vehicles.length}대의 차량이 {userProfile.name}님과 매칭되었습니다
        </p>

        {/* 추천 메타데이터 표시 */}
        {recommendationData && (
          <div className="mt-4 flex justify-center gap-4 text-sm text-gray-500">
            <div className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              <span>{recommendationData.metadata.processingTime}ms</span>
            </div>
            <div className="flex items-center gap-1">
              <Award className="w-4 h-4" />
              <span>{recommendationData.metadata.modelUsed}</span>
            </div>
            {recommendationData.debug && (
              <div className="flex items-center gap-1">
                <Target className="w-4 h-4" />
                <span>{recommendationData.debug.userSegment} 세그먼트</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* 차량 그리드 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {vehicles.map((vehicle) => {
          const fuelInfo = getFuelTypeInfo(vehicle.fuel_type);
          const isSelected = selectedVehicles.includes(vehicle.id);
          const isFavorited = favoriteVehicles.includes(vehicle.id);

          return (
            <Card
              key={vehicle.id}
              className={`group relative bg-white/80 backdrop-blur-sm border-2 transition-all duration-300 cursor-pointer hover:shadow-2xl hover:shadow-purple-500/20 hover:-translate-y-2 ${
                isSelected
                  ? 'border-purple-500 shadow-lg shadow-purple-500/25 bg-gradient-to-br from-purple-50 to-pink-50'
                  : 'border-purple-200/50 hover:border-purple-300'
              }`}
              onClick={() => handleVehicleClick(vehicle.id)}
            >
              {/* 매칭 점수 배지 */}
              <div className="absolute -top-2 -right-2 z-10">
                <div className={`bg-gradient-to-r ${getMatchScoreColor(vehicle.match_score)} text-white px-3 py-1 rounded-full text-sm font-bold shadow-lg`}>
                  {vehicle.match_score}% 매칭
                </div>
              </div>

              {/* 추천 타입 배지 */}
              <div className="absolute -top-2 -left-2 z-10">
                <div className="bg-white/90 backdrop-blur-sm text-purple-600 px-2 py-1 rounded-full text-xs font-medium shadow-lg border border-purple-200 flex items-center gap-1">
                  {getRecommendationTypeIcon(vehicle.recommendation_type)}
                  <span>{getRecommendationTypeLabel(vehicle.recommendation_type)}</span>
                </div>
              </div>

              {/* 좋아요 버튼 */}
              <button
                onClick={(e) => toggleFavorite(vehicle.id, e)}
                className={`absolute top-4 right-4 z-10 p-2 rounded-full backdrop-blur-sm transition-all duration-200 ${
                  isFavorited
                    ? 'bg-red-500 text-white shadow-lg'
                    : 'bg-white/70 text-gray-600 hover:bg-red-50 hover:text-red-500'
                }`}
              >
                <Heart className={`w-4 h-4 ${isFavorited ? 'fill-current' : ''}`} />
              </button>

              {/* 차량 이미지 */}
              <div className="relative h-48 bg-gradient-to-br from-purple-100 to-pink-100 rounded-t-lg overflow-hidden">
                {vehicle.images && vehicle.images.length > 0 ? (
                  <img
                    src={vehicle.images[0]}
                    alt={`${vehicle.brand} ${vehicle.model}`}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500"
                  />
                ) : (
                  <div className="w-full h-full flex items-center justify-center">
                    <Car className="w-16 h-16 text-purple-400" />
                  </div>
                )}
                <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent"></div>
              </div>

              {/* 차량 정보 */}
              <CardHeader className="pb-4">
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <CardTitle className="text-xl font-bold text-gray-900 mb-1">
                      {vehicle.brand} {vehicle.model}
                    </CardTitle>
                    <div className="flex items-center gap-2 text-gray-600">
                      <Calendar className="w-4 h-4" />
                      <span>{vehicle.year}년</span>
                      <span className="text-gray-400">•</span>
                      <span>{vehicle.mileage?.toLocaleString()}km</span>
                    </div>
                  </div>
                </div>

                <div className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                  {vehicle.price?.toLocaleString()}만원
                </div>
              </CardHeader>

              <CardContent className="space-y-4">
                {/* 연료/연비 정보 */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center gap-2">
                    <div className={`p-1.5 rounded-lg ${fuelInfo.color}`}>
                      <span className="text-sm">{fuelInfo.emoji}</span>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">연료</div>
                      <div className="text-sm font-medium">{fuelInfo.label}</div>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <div className="p-1.5 rounded-lg bg-orange-100 text-orange-800">
                      <Fuel className="w-4 h-4" />
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">연비</div>
                      <div className="text-sm font-medium">{vehicle.fuel_efficiency.toFixed(1)}km/L</div>
                    </div>
                  </div>
                </div>

                {/* 위치/안전성 */}
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center gap-2">
                    <MapPin className="w-4 h-4 text-gray-400" />
                    <span className="text-sm text-gray-600">{vehicle.location}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-gray-400" />
                    <div className="flex items-center">
                      {[...Array(5)].map((_, i) => (
                        <Star
                          key={i}
                          className={`w-3 h-3 ${
                            i < vehicle.safety_rating
                              ? 'text-yellow-400 fill-current'
                              : 'text-gray-300'
                          }`}
                        />
                      ))}
                      <span className="text-xs text-gray-500 ml-1">안전</span>
                    </div>
                  </div>
                </div>

                {/* 주요 옵션 */}
                {vehicle.features && vehicle.features.length > 0 && (
                  <div className="space-y-2">
                    <div className="text-sm font-medium text-gray-700">주요 옵션</div>
                    <div className="flex flex-wrap gap-1">
                      {vehicle.features.slice(0, 3).map((feature, index) => (
                        <Badge
                          key={index}
                          variant="secondary"
                          className="text-xs bg-purple-50 text-purple-700 border-purple-200"
                        >
                          {feature}
                        </Badge>
                      ))}
                      {vehicle.features.length > 3 && (
                        <Badge variant="outline" className="text-xs text-gray-500">
                          +{vehicle.features.length - 3}개
                        </Badge>
                      )}
                    </div>
                  </div>
                )}

                {/* 추천 이유 */}
                {vehicle.recommendation_reason && (
                  <div className="bg-purple-50 rounded-lg p-3 border border-purple-100">
                    <div className="flex items-start gap-2">
                      <Sparkles className="w-4 h-4 text-purple-500 flex-shrink-0 mt-0.5" />
                      <p className="text-sm text-purple-700 line-clamp-2">
                        {vehicle.recommendation_reason}
                      </p>
                    </div>
                  </div>
                )}

                {/* 액션 버튼 */}
                <div className="grid grid-cols-2 gap-2 pt-4">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleVehicleClick(vehicle.id);
                    }}
                    className="border-purple-200 text-purple-600 hover:bg-purple-50 hover:border-purple-300"
                  >
                    <GitCompare className="w-4 h-4 mr-1" />
                    {isSelected ? '선택됨' : '선택'}
                  </Button>

                  <Button
                    size="sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRequestFinancing(vehicle.id);
                    }}
                    className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white"
                  >
                    <CreditCard className="w-4 h-4 mr-1" />
                    대출
                  </Button>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* 선택된 차량 상태 */}
      {selectedVehicles.length > 0 && (
        <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200/50 text-center">
          <h3 className="text-lg font-semibold text-purple-800 mb-2">
            {selectedVehicles.length}/3 차량 선택됨
          </h3>
          <p className="text-purple-600 text-sm">
            {selectedVehicles.length === 3
              ? '3대 모두 선택되었습니다! 상세 비교 분석을 확인해보세요.'
              : `${3 - selectedVehicles.length}대 더 선택하시면 상세 비교 분석을 받을 수 있습니다.`
            }
          </p>
        </div>
      )}

      {/* 빈 상태 */}
      {vehicles.length === 0 && !loading && (
        <div className="text-center py-12">
          <Car className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            추천할 차량이 없습니다
          </h3>
          <p className="text-gray-500">
            프로필 정보를 업데이트하거나 다른 조건을 시도해보세요.
          </p>
        </div>
      )}
    </div>
  );
}