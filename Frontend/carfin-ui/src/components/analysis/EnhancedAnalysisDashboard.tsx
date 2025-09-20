'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import {
  Car,
  BarChart3,
  TrendingUp,
  DollarSign,
  Star,
  CheckCircle,
  ArrowRight,
  MapPin,
  Calendar,
  Fuel,
  Gauge,
  Shield,
  Award,
  Clock,
  ThumbsUp,
  ThumbsDown,
  AlertTriangle,
  Target,
  Zap
} from 'lucide-react';
import { PentagonChart } from './PentagonChart';

interface Vehicle {
  id: string;
  brand: string;
  model: string;
  year: number;
  price: number;
  mileage: number;
  fuel_type: string;
  body_type: string;
  location: string;
  features: string[];
  fuel_efficiency: number;
  safety_rating: number;
  match_score: number;
  description: string;
}

interface VehicleFeedback {
  vehicleId: string;
  feedbackType: 'like' | 'dislike' | 'expensive' | 'maybe' | 'love' | 'question';
  timestamp: Date;
}

interface PersonalizedAnalysis {
  whyRecommended: string[];
  userPreferences: {
    pricesensitive: boolean;
    performance_focused: boolean;
    safety_focused: boolean;
    fuel_efficiency_focused: boolean;
  };
  pentagon_scores: {
    performance: number;
    price: number;
    safety: number;
    fuel_efficiency: number;
    design: number;
  };
  risk_factors: string[];
  pros: string[];
  cons: string[];
  confidence: number;
}

interface EnhancedAnalysisDashboardProps {
  selectedVehicles: Vehicle[];
  userFeedback: VehicleFeedback[];
  onProceedToFinance?: () => void;
  onSelectDifferentVehicle?: () => void;
}

export function EnhancedAnalysisDashboard({
  selectedVehicles,
  userFeedback,
  onProceedToFinance,
  onSelectDifferentVehicle
}: EnhancedAnalysisDashboardProps) {
  const [analysis, setAnalysis] = useState<PersonalizedAnalysis | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedVehicleIndex, setSelectedVehicleIndex] = useState(0);

  const currentVehicle = selectedVehicles && selectedVehicles.length > 0
    ? (selectedVehicles[selectedVehicleIndex] || selectedVehicles[0])
    : null;

  useEffect(() => {
    generatePersonalizedAnalysis();
  }, [selectedVehicles, userFeedback]);

  const generatePersonalizedAnalysis = async () => {
    if (!currentVehicle) {
      setIsLoading(false);
      return;
    }

    setIsLoading(true);

    // 사용자 피드백 패턴 분석
    const likedFeedbacks = userFeedback.filter(f => f.feedbackType === 'like' || f.feedbackType === 'love');
    const dislikedFeedbacks = userFeedback.filter(f => f.feedbackType === 'dislike');
    const expensiveFeedbacks = userFeedback.filter(f => f.feedbackType === 'expensive');

    // 사용자 선호도 분석
    const preferences = {
      pricesensitive: expensiveFeedbacks.length > 2,
      performance_focused: false, // 실제로는 좋아한 차량들의 패턴 분석
      safety_focused: currentVehicle.safety_rating >= 4,
      fuel_efficiency_focused: currentVehicle.fuel_efficiency > 13
    };

    // 시뮬레이션 대기
    await new Promise(resolve => setTimeout(resolve, 2000));

    const mockAnalysis: PersonalizedAnalysis = {
      whyRecommended: [
        `당신이 선택한 ${likedFeedbacks.length}개 차량과 비슷한 특성을 가졌어요`,
        `${preferences.pricesensitive ? '가격 대비' : ''} 높은 가치를 제공합니다`,
        `당신의 사용 패턴에 ${currentVehicle.match_score}% 적합해요`,
        '낮은 리스크로 안전한 선택입니다'
      ],
      userPreferences: preferences,
      pentagon_scores: {
        performance: Math.min(95, currentVehicle.year >= 2020 ? 85 : 70),
        price: preferences.pricesensitive ? 90 : 80,
        safety: currentVehicle.safety_rating * 20,
        fuel_efficiency: Math.min(95, currentVehicle.fuel_efficiency * 6),
        design: Math.min(95, currentVehicle.year >= 2021 ? 85 : 75)
      },
      risk_factors: [
        currentVehicle.mileage > 50000 ? '주행거리가 다소 높음' : '',
        currentVehicle.year < 2020 ? '연식이 다소 오래됨' : '',
      ].filter(Boolean),
      pros: [
        `${currentVehicle.brand}의 신뢰성`,
        `${currentVehicle.fuel_efficiency}km/L의 우수한 연비`,
        '풍부한 편의사양',
        '합리적인 가격대'
      ],
      cons: [
        currentVehicle.mileage > 30000 ? '상대적으로 높은 주행거리' : '',
        preferences.pricesensitive ? '추가 옵션 비용 고려 필요' : '',
        '일부 부품 교체 시기 도래 가능성'
      ].filter(Boolean),
      confidence: Math.min(95, 75 + likedFeedbacks.length * 5)
    };

    setAnalysis(mockAnalysis);
    setIsLoading(false);
  };

  const getUserFeedbackSummary = () => {
    const summary = userFeedback.reduce((acc, feedback) => {
      acc[feedback.feedbackType] = (acc[feedback.feedbackType] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return summary;
  };

  // 차량이 선택되지 않았을 경우
  if (!currentVehicle) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center max-w-md mx-auto px-6">
          <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <Car className="w-8 h-8 text-orange-600" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-4">
            선택된 차량이 없습니다
          </h2>
          <p className="text-gray-600 mb-6">
            분석을 위해 먼저 차량을 선택해주세요
          </p>
          <button
            onClick={onSelectDifferentVehicle}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg transition-colors"
          >
            차량 선택하러 가기
          </button>
        </div>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <BarChart3 className="w-8 h-8 text-blue-600 animate-pulse" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">당신만의 분석을 준비중...</h2>
          <p className="text-gray-600">선택한 차량들과 피드백을 바탕으로 개인화된 분석을 생성하고 있어요</p>
        </div>
      </div>
    );
  }

  if (!analysis) return null;

  const feedbackSummary = getUserFeedbackSummary();

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 헤더 */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="text-center">
            <h1 className="text-3xl font-bold text-gray-900 mb-2">🎯 당신만의 차량 분석</h1>
            <p className="text-gray-600">당신의 선택 패턴을 학습하여 개인화된 추천을 제공합니다</p>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        {/* 최종 추천 차량들 */}
        {selectedVehicles.length > 1 && (
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 mb-8">
            <h2 className="text-xl font-bold text-gray-900 mb-4">🏆 최종 추천 {selectedVehicles.length}개</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {selectedVehicles.map((vehicle, index) => (
                <div
                  key={vehicle.id}
                  className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                    index === selectedVehicleIndex
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                  onClick={() => setSelectedVehicleIndex(index)}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                      <Car className="w-6 h-6 text-gray-400" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900">
                        {vehicle.brand} {vehicle.model}
                      </h3>
                      <p className="text-sm text-gray-600">{vehicle.price.toLocaleString()}만원</p>
                    </div>
                  </div>
                  {index === selectedVehicleIndex && (
                    <div className="mt-2 text-xs text-blue-600 font-medium">← 현재 분석 중</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 메인 차량 카드 */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl border border-blue-200 p-6 mb-8">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-6">
              <div className="w-24 h-24 bg-white rounded-xl flex items-center justify-center shadow-sm">
                <Car className="w-12 h-12 text-blue-600" />
              </div>
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <h2 className="text-3xl font-bold text-gray-900">
                    {currentVehicle.brand} {currentVehicle.model}
                  </h2>
                  <div className="bg-green-100 text-green-700 px-3 py-1 rounded-full text-sm font-medium">
                    매치 {currentVehicle.match_score}%
                  </div>
                </div>
                <p className="text-gray-600 mb-4">{currentVehicle.year}년식 • {currentVehicle.body_type}</p>
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <div className="flex items-center gap-1">
                    <Gauge className="w-4 h-4" />
                    <span>{currentVehicle.mileage.toLocaleString()}km</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Fuel className="w-4 h-4" />
                    <span>{currentVehicle.fuel_type}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Zap className="w-4 h-4" />
                    <span>{currentVehicle.fuel_efficiency}km/L</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <MapPin className="w-4 h-4" />
                    <span>{currentVehicle.location}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-4xl font-bold text-blue-600 mb-1">
                {currentVehicle.price.toLocaleString()}만원
              </div>
              <div className="flex items-center gap-1 text-sm text-green-600">
                <TrendingUp className="w-4 h-4" />
                시장가 대비 합리적
              </div>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* 왼쪽: 오각형 차트 & 주요 분석 */}
          <div className="xl:col-span-2 space-y-6">
            {/* 종합 성능 분석 */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-xl font-bold text-gray-900 mb-6">📊 종합 성능 분석</h3>
              <div className="flex items-center justify-center">
                <PentagonChart
                  data={analysis.pentagon_scores}
                  size={300}
                  className="mx-auto"
                />
              </div>
            </div>

            {/* 왜 이 차량을 추천하나요? */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                  <Target className="w-5 h-5 text-green-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900">🎯 왜 이 차량을 추천하나요?</h3>
              </div>
              <div className="space-y-3">
                {analysis.whyRecommended.map((reason, index) => (
                  <div key={index} className="flex items-start gap-3">
                    <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center mt-0.5">
                      <span className="text-xs font-bold text-blue-600">{index + 1}</span>
                    </div>
                    <p className="text-gray-700 leading-relaxed">{reason}</p>
                  </div>
                ))}
              </div>
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <div className="flex items-center gap-2">
                  <Award className="w-5 h-5 text-blue-600" />
                  <span className="font-semibold text-blue-900">신뢰도: {analysis.confidence}%</span>
                </div>
              </div>
            </div>

            {/* 장단점 분석 */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-bold text-green-700 mb-4 flex items-center gap-2">
                  <ThumbsUp className="w-5 h-5" />
                  장점
                </h3>
                <div className="space-y-2">
                  {analysis.pros.map((pro, index) => (
                    <div key={index} className="flex items-center gap-2">
                      <CheckCircle className="w-4 h-4 text-green-500" />
                      <span className="text-sm text-gray-700">{pro}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-bold text-orange-700 mb-4 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5" />
                  고려사항
                </h3>
                <div className="space-y-2">
                  {analysis.cons.map((con, index) => (
                    <div key={index} className="flex items-center gap-2">
                      <div className="w-4 h-4 border-2 border-orange-400 rounded-full flex items-center justify-center">
                        <div className="w-1 h-1 bg-orange-400 rounded-full"></div>
                      </div>
                      <span className="text-sm text-gray-700">{con}</span>
                    </div>
                  ))}
                  {analysis.risk_factors.map((risk, index) => (
                    <div key={`risk-${index}`} className="flex items-center gap-2">
                      <AlertTriangle className="w-4 h-4 text-orange-500" />
                      <span className="text-sm text-gray-700">{risk}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* 오른쪽: 사용자 분석 & 피드백 */}
          <div className="space-y-6">
            {/* 당신의 선택 패턴 */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">👤 당신의 선택 패턴</h3>
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">선택한 차량</span>
                  <span className="font-semibold text-blue-600">{feedbackSummary.like || 0}개</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">관심 없는 차량</span>
                  <span className="font-semibold text-red-600">{feedbackSummary.dislike || 0}개</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-sm text-gray-600">가격 부담</span>
                  <span className="font-semibold text-orange-600">{feedbackSummary.expensive || 0}개</span>
                </div>
              </div>

              <div className="mt-6 pt-4 border-t border-gray-100">
                <h4 className="font-semibold text-gray-900 mb-3">분석된 선호도</h4>
                <div className="space-y-2">
                  {analysis.userPreferences.pricesensitive && (
                    <div className="bg-yellow-50 text-yellow-700 px-3 py-2 rounded-lg text-sm">
                      💰 가격 민감형
                    </div>
                  )}
                  {analysis.userPreferences.safety_focused && (
                    <div className="bg-blue-50 text-blue-700 px-3 py-2 rounded-lg text-sm">
                      🛡️ 안전 중시형
                    </div>
                  )}
                  {analysis.userPreferences.fuel_efficiency_focused && (
                    <div className="bg-green-50 text-green-700 px-3 py-2 rounded-lg text-sm">
                      ⚡ 연비 중시형
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* 추가 정보 */}
            <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-bold text-gray-900 mb-4">📋 추가 정보</h3>
              <div className="space-y-4">
                <div>
                  <h4 className="font-semibold text-gray-700 mb-2">주요 특징</h4>
                  <div className="flex flex-wrap gap-2">
                    {currentVehicle.features.slice(0, 4).map((feature, index) => (
                      <span
                        key={index}
                        className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded"
                      >
                        {feature}
                      </span>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold text-gray-700 mb-2">예상 유지비 (월)</h4>
                  <div className="text-2xl font-bold text-gray-900">
                    {Math.round(currentVehicle.price * 0.03).toLocaleString()}원
                  </div>
                  <p className="text-xs text-gray-500 mt-1">보험료, 유지보수비 포함 추정</p>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 액션 버튼 */}
        <div className="mt-12 flex flex-col sm:flex-row gap-4 justify-center">
          <Button
            onClick={onProceedToFinance}
            size="lg"
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-4 text-lg"
          >
            <CheckCircle className="w-5 h-5 mr-2" />
            이 차량으로 진행하기
            <ArrowRight className="w-5 h-5 ml-2" />
          </Button>

          <Button
            onClick={onSelectDifferentVehicle}
            variant="outline"
            size="lg"
            className="border-gray-300 text-gray-700 hover:bg-gray-50 px-8 py-4 text-lg"
          >
            다른 차량 보기
          </Button>
        </div>

        {/* 주의사항 */}
        <div className="mt-8 bg-blue-50 border border-blue-200 rounded-xl p-4">
          <div className="flex items-start gap-3">
            <Clock className="w-5 h-5 text-blue-600 mt-0.5" />
            <div>
              <h4 className="font-semibold text-blue-900 mb-1">개인화 분석 안내</h4>
              <p className="text-sm text-blue-700">
                이 분석은 당신의 선택 패턴을 학습하여 생성된 개인화 정보입니다.
                실제 차량 상태는 직접 확인하시고, 전문가의 검수를 받으시기 바랍니다.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}