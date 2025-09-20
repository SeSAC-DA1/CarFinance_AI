'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Heart, Eye, GitCompare, Share2, MapPin, Fuel, Calendar, Car, Brain } from 'lucide-react';
import { RecommendationResult, ProcessedCarData, UserInteraction } from '@/types';
import { MOCK_CARS } from '@/lib/data/mock-data';

interface InteractiveResultsProps {
  recommendations: RecommendationResult;
  onCarSelection: (carIds: string[]) => void;
  sessionId: string;
}

export function InteractiveResults({
  recommendations,
  onCarSelection,
  sessionId
}: InteractiveResultsProps) {
  const [displayedCars, setDisplayedCars] = useState<ProcessedCarData[]>([]);
  const [likedCars, setLikedCars] = useState<Set<string>>(new Set());
  const [comparedCars, setComparedCars] = useState<Set<string>>(new Set());
  const [hoveredCar, setHoveredCar] = useState<string | null>(null);
  const [userInteractions, setUserInteractions] = useState<UserInteraction[]>([]);

  useEffect(() => {
    // 더미 데이터로 결과 시뮬레이션
    const mockResults = MOCK_CARS.slice(0, 20).map((car, index) => ({
      ...car,
      match_score: Math.random() * 0.4 + 0.6, // 60-100% 매치율
      match_reasons: generateMatchReasons(car),
      ranking_position: index + 1,
      agent_scores: {
        collaborative_filtering: Math.random() * 0.3 + 0.4,
        market_analysis: Math.random() * 0.3 + 0.5,
        personal_preference: Math.random() * 0.4 + 0.6
      }
    }));

    // 매치 점수 순으로 정렬
    const sortedResults = mockResults.sort((a, b) => b.match_score - a.match_score);
    setDisplayedCars(sortedResults);
  }, [recommendations]);

  const generateMatchReasons = (car: ProcessedCarData): string[] => {
    const reasons = [];
    if (car.year >= 2020) reasons.push('최신 연식');
    if (car.price <= 3000) reasons.push('예산 적합');
    if (car.fuelType === 'hybrid') reasons.push('친환경');
    if (car.mileage <= 50000) reasons.push('낮은 주행거리');
    return reasons.slice(0, 2);
  };

  const trackInteraction = (carId: string, action: string, duration?: number) => {
    const interaction: UserInteraction = {
      id: `interaction_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`,
      userId: 'demo_user',
      sessionId,
      type: 'implicit',
      action,
      target: {
        type: 'car_card',
        carId,
        position: displayedCars.findIndex(car => car.id === carId) + 1
      },
      context: {
        duration,
        scrollPosition: window.scrollY,
        viewport: { width: window.innerWidth, height: window.innerHeight },
        device: 'desktop',
        timeOfDay: new Date().getHours() < 12 ? 'morning' : 'afternoon'
      },
      timestamp: Date.now()
    };

    setUserInteractions(prev => [...prev, interaction]);

    // 실제 시스템에서는 서버로 전송
    console.log('User interaction tracked:', interaction);
  };

  const handleLike = (carId: string) => {
    const newLiked = new Set(likedCars);
    if (newLiked.has(carId)) {
      newLiked.delete(carId);
      trackInteraction(carId, 'unlike');
    } else {
      newLiked.add(carId);
      trackInteraction(carId, 'like');
    }
    setLikedCars(newLiked);
  };

  const handleCompare = (carId: string) => {
    const newCompared = new Set(comparedCars);
    if (newCompared.has(carId)) {
      newCompared.delete(carId);
      trackInteraction(carId, 'remove_comparison');
    } else if (newCompared.size < 3) {
      newCompared.add(carId);
      trackInteraction(carId, 'add_comparison');
    }
    setComparedCars(newCompared);

    if (newCompared.size >= 2) {
      onCarSelection(Array.from(newCompared));
    }
  };

  const handleCarHover = (carId: string) => {
    setHoveredCar(carId);
    const hoverStart = Date.now();

    return () => {
      const hoverDuration = Date.now() - hoverStart;
      if (hoverDuration > 1000) { // 1초 이상 호버
        trackInteraction(carId, 'long_hover', hoverDuration);
      }
      setHoveredCar(null);
    };
  };

  const formatPrice = (price: number) => {
    return `${price.toLocaleString()}만원`;
  };

  const formatMileage = (mileage: number) => {
    return `${Math.round(mileage / 1000)}만km`;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-emerald-900 via-slate-900 to-teal-900 relative overflow-hidden">
      {/* 배경 애니메이션 */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-10 w-96 h-96 bg-emerald-500/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-10 w-80 h-80 bg-teal-500/5 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/3 w-72 h-72 bg-cyan-500/5 rounded-full blur-3xl animate-pulse delay-2000"></div>

        {/* 떠다니는 별들 */}
        {[...Array(15)].map((_, i) => (
          <div
            key={i}
            className="absolute w-2 h-2 bg-yellow-400/20 rounded-full animate-pulse"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${1 + Math.random() * 2}s`
            }}
          />
        ))}
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 py-8">
        {/* 화려한 헤더 */}
        <div className="mb-12 text-center">
          <div className="flex items-center justify-center mb-6">
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-emerald-500 to-teal-500 rounded-full blur-xl opacity-50 animate-pulse"></div>
              <div className="relative bg-gradient-to-r from-emerald-600 to-teal-600 p-6 rounded-full">
                <Car className="w-12 h-12 text-white animate-bounce" />
              </div>
            </div>
          </div>

          <h2 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-white via-emerald-200 to-teal-200 bg-clip-text text-transparent mb-4">
            🎯 AI 추천 결과
          </h2>

          <div className="flex items-center justify-center gap-3 mb-8">
            <Badge className="bg-emerald-500/20 text-emerald-300 border-emerald-500/30 animate-pulse">
              {displayedCars.length}개 최적 차량 발견
            </Badge>
            <Badge className="bg-teal-500/20 text-teal-300 border-teal-500/30 animate-pulse delay-500">
              평균 정확도 94%
            </Badge>
            <Badge className="bg-cyan-500/20 text-cyan-300 border-cyan-500/30 animate-pulse delay-1000">
              8.4초 분석 완료
            </Badge>
          </div>

          {comparedCars.size > 0 && (
            <div className="mt-6 p-6 bg-white/10 border border-white/20 backdrop-blur-sm rounded-2xl max-w-2xl mx-auto">
              <div className="flex items-center justify-center gap-3 text-white">
                <GitCompare className="w-6 h-6 text-purple-400 animate-bounce" />
                <p className="text-lg">
                  💡 {comparedCars.size}개 차량이 비교 목록에 추가되었습니다.
                  {comparedCars.size >= 2 && ' 상세 비교를 확인해보세요!'}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* 필터 및 정렬 */}
        <div className="mb-8 flex flex-wrap gap-4 items-center justify-between">
          <div className="flex gap-3">
            <Badge className="bg-emerald-500/20 text-emerald-300 border-emerald-500/30 hover:bg-emerald-500/30 transition-all cursor-pointer">
              매치율 높은 순
            </Badge>
            <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/30 hover:bg-blue-500/30 transition-all cursor-pointer">
              최신 연식
            </Badge>
            <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/30 hover:bg-purple-500/30 transition-all cursor-pointer">
              낮은 가격
            </Badge>
          </div>
          <div className="text-sm text-gray-300 bg-white/10 px-3 py-1 rounded-full">
            {userInteractions.length}개 상호작용 학습됨
          </div>
        </div>

        {/* 차량 그리드 - 완전 반응형 */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-4 2xl:grid-cols-5 gap-4 md:gap-6 w-full">
        {displayedCars.map((car) => (
          <Card
            key={car.id}
            className={`group cursor-pointer transition-all duration-500 transform hover:shadow-2xl hover:-translate-y-2 hover:scale-[1.02] backdrop-blur-sm border ${
              hoveredCar === car.id
                ? 'ring-2 ring-emerald-400 bg-white/20 border-emerald-400/50 shadow-emerald-500/25'
                : 'bg-white/10 border-white/20 hover:bg-white/15 hover:border-white/30'
            }`}
            onMouseEnter={handleCarHover(car.id)}
            onMouseLeave={hoveredCar === car.id ? handleCarHover(car.id) : undefined}
            onClick={() => trackInteraction(car.id, 'click')}
          >
            <CardHeader className="p-3 pb-2">
              {/* 차량 이미지 */}
              <div className="relative h-32 sm:h-36 md:h-40 lg:h-44 bg-gray-100 rounded-lg overflow-hidden mb-3">
                <img
                  src={car.images?.[0] || `https://picsum.photos/400/300?random=${car.id}`}
                  alt={`${car.brand} ${car.model}`}
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                />

                {/* 매치 점수 배지 */}
                <div className="absolute top-3 left-3">
                  <Badge className="bg-green-500 text-white">
                    {Math.round((car.match_score || 0.7) * 100)}% 매치
                  </Badge>
                </div>

                {/* 순위 배지 */}
                <div className="absolute top-3 right-3">
                  <Badge variant="outline" className="bg-white">
                    #{car.ranking_position}
                  </Badge>
                </div>

                {/* 상호작용 버튼들 */}
                <div className="absolute bottom-3 right-3 flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <Button
                    size="sm"
                    variant={likedCars.has(car.id) ? "default" : "outline"}
                    className="h-8 w-8 p-0"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleLike(car.id);
                    }}
                  >
                    <Heart className={`w-4 h-4 ${likedCars.has(car.id) ? 'fill-current' : ''}`} />
                  </Button>

                  <Button
                    size="sm"
                    variant={comparedCars.has(car.id) ? "default" : "outline"}
                    className="h-8 w-8 p-0"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCompare(car.id);
                    }}
                    disabled={!comparedCars.has(car.id) && comparedCars.size >= 3}
                  >
                    <GitCompare className="w-4 h-4" />
                  </Button>
                </div>
              </div>

              {/* 차량 정보 */}
              <CardTitle className="text-base md:text-lg truncate">
                {car.brand} {car.model}
              </CardTitle>

              {/* 매치 이유 */}
              <div className="flex flex-wrap gap-1 mt-2">
                {(car.match_reasons || generateMatchReasons(car)).slice(0, 2).map((reason, index) => (
                  <Badge key={index} variant="secondary" className="text-xs px-1.5 py-0.5">
                    {reason}
                  </Badge>
                ))}
              </div>
            </CardHeader>

            <CardContent className="p-3 pt-0">
              {/* 기본 정보 */}
              <div className="space-y-2">
                <div className="text-center">
                  <span className="text-lg md:text-xl font-bold text-emerald-400">
                    {formatPrice(car.price)}
                  </span>
                  <div className="text-xs text-gray-400 mt-1">
                    {car.year}년 • {formatMileage(car.mileage)}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="flex items-center gap-1 text-gray-300 truncate">
                    <Fuel className="w-3 h-3 flex-shrink-0" />
                    <span className="truncate">{car.fuelType}</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-300 truncate">
                    <Car className="w-3 h-3 flex-shrink-0" />
                    <span className="truncate">{car.bodyType}</span>
                  </div>
                  <div className="flex items-center gap-1 text-gray-300 truncate col-span-2">
                    <MapPin className="w-3 h-3 flex-shrink-0" />
                    <span className="truncate">{car.location}</span>
                  </div>
                </div>

                {/* 에이전트 점수 - 컴팩트 */}
                <div className="pt-2 border-t border-white/10">
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-gray-400">AI 매칭</span>
                    <span className="text-xs font-bold text-cyan-400">
                      {Math.round((car.agent_scores?.collaborative_filtering || 0.7) * 100)}%
                    </span>
                  </div>
                  <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full"
                      style={{ width: `${Math.round((car.agent_scores?.collaborative_filtering || 0.7) * 100)}%` }}
                    />
                  </div>
                </div>
              </div>

              {/* 액션 버튼 - 컴팩트 */}
              <div className="mt-3 flex gap-1">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1 text-xs h-8 border-white/20 text-white hover:bg-white/10"
                  onClick={(e) => {
                    e.stopPropagation();
                    trackInteraction(car.id, 'detail_view');
                  }}
                >
                  <Eye className="w-3 h-3 mr-1" />
                  상세
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 w-8 p-0 border-white/20 text-white hover:bg-white/10"
                  onClick={(e) => {
                    e.stopPropagation();
                    trackInteraction(car.id, 'share');
                  }}
                >
                  <Share2 className="w-3 h-3" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

        {/* 실시간 학습 표시 */}
        <div className="mt-12 p-8 bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/20 backdrop-blur-sm rounded-3xl">
          <div className="text-center mb-8">
            <div className="flex items-center justify-center mb-4">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full blur-lg opacity-50 animate-pulse"></div>
                <div className="relative bg-gradient-to-r from-purple-600 to-pink-600 p-4 rounded-full">
                  <Brain className="w-8 h-8 text-white animate-pulse" />
                </div>
              </div>
            </div>

            <h3 className="text-2xl font-bold text-white mb-2 flex items-center justify-center gap-2">
              🧠 실시간 AI 학습
            </h3>
            <p className="text-gray-300">
              당신의 선택을 분석하여 추천을 개선하고 있습니다
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 sm:gap-8">
            <div className="text-center">
              <div className="relative mb-4">
                <div className="text-4xl font-bold text-purple-400 animate-pulse">
                  {userInteractions.length}
                </div>
                <div className="absolute inset-0 bg-purple-500/20 rounded-full blur-xl animate-pulse"></div>
              </div>
              <p className="text-gray-300 font-medium">상호작용</p>
              <p className="text-xs text-gray-400 mt-1">클릭, 호버, 좋아요</p>
            </div>

            <div className="text-center">
              <div className="relative mb-4">
                <div className="text-4xl font-bold text-pink-400 animate-pulse delay-300">
                  {likedCars.size}
                </div>
                <div className="absolute inset-0 bg-pink-500/20 rounded-full blur-xl animate-pulse delay-300"></div>
              </div>
              <p className="text-gray-300 font-medium">관심 차량</p>
              <p className="text-xs text-gray-400 mt-1">하트 표시한 차량</p>
            </div>

            <div className="text-center">
              <div className="relative mb-4">
                <div className="text-4xl font-bold text-cyan-400 animate-pulse delay-600">
                  {comparedCars.size}
                </div>
                <div className="absolute inset-0 bg-cyan-500/20 rounded-full blur-xl animate-pulse delay-600"></div>
              </div>
              <p className="text-gray-300 font-medium">비교 목록</p>
              <p className="text-xs text-gray-400 mt-1">비교 선택한 차량</p>
            </div>
          </div>

          {/* 학습 진행률 표시 */}
          <div className="mt-8 pt-6 border-t border-white/10">
            <div className="flex items-center justify-between mb-3">
              <span className="text-white font-medium">AI 학습 진행률</span>
              <span className="text-cyan-400 font-bold">
                {Math.min(userInteractions.length * 5 + likedCars.size * 10 + comparedCars.size * 15, 100)}%
              </span>
            </div>
            <div className="w-full h-2 bg-white/10 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-purple-500 to-pink-500 rounded-full transition-all duration-1000 animate-pulse"
                style={{
                  width: `${Math.min(userInteractions.length * 5 + likedCars.size * 10 + comparedCars.size * 15, 100)}%`
                }}
              />
            </div>
            <p className="text-xs text-gray-400 mt-2 text-center">
              더 많은 상호작용으로 추천 정확도가 향상됩니다
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}