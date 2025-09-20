'use client';

import { useState, useEffect, useCallback } from 'react';
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
  Fuel,
  Gauge,
  Shield,
  Award,
  Clock
} from 'lucide-react';
import { VehicleFeedback } from '@/types';
import { type Vehicle } from '@/lib/gemini-agents';

interface AnalysisData {
  selectedVehicle: Vehicle;
  matchScore: number;
  priceAnalysis: {
    marketPrice: number;
    dealPrice: number;
    savings: number;
    priceRating: 'excellent' | 'good' | 'fair';
  };
  riskAssessment: {
    overall: 'low' | 'medium' | 'high';
    factors: string[];
    score: number;
  };
  recommendation: {
    verdict: 'recommended' | 'conditional' | 'not_recommended';
    reasons: string[];
    confidence: number;
  };
}

interface AnalysisDashboardProps {
  selectedVehicle: Vehicle;
  userFeedback: VehicleFeedback[];
  onProceedToFinance?: () => void;
  onSelectDifferentVehicle?: () => void;
}

export function AnalysisDashboard({
  selectedVehicle,
  userFeedback: _userFeedback,
  onProceedToFinance,
  onSelectDifferentVehicle
}: AnalysisDashboardProps) {
  const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    generateAnalysis();
  }, [selectedVehicle, generateAnalysis]);

  const generateAnalysis = useCallback(async () => {
    setIsLoading(true);

    // 실제로는 Gemini API를 호출하여 분석을 생성
    await new Promise(resolve => setTimeout(resolve, 2000));

    const mockAnalysis: AnalysisData = {
      selectedVehicle,
      matchScore: selectedVehicle.match_score,
      priceAnalysis: {
        marketPrice: selectedVehicle.price + 200,
        dealPrice: selectedVehicle.price,
        savings: 200,
        priceRating: 'good'
      },
      riskAssessment: {
        overall: 'low',
        factors: ['정기 점검 이력 확인됨', '사고 이력 없음', '단일 소유주'],
        score: 85
      },
      recommendation: {
        verdict: 'recommended',
        reasons: [
          '시장 가격 대비 합리적인 가격',
          '사용자 취향과 92% 일치',
          '낮은 위험도로 안전한 선택',
          '연비와 유지비가 경제적'
        ],
        confidence: 92
      }
    };

    setAnalysis(mockAnalysis);
    setIsLoading(false);
  }, [selectedVehicle]);

  const getPriceRatingColor = (rating: string) => {
    switch (rating) {
      case 'excellent': return 'text-green-600 bg-green-50';
      case 'good': return 'text-blue-600 bg-blue-50';
      case 'fair': return 'text-yellow-600 bg-yellow-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'text-green-600 bg-green-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'high': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getRecommendationColor = (verdict: string) => {
    switch (verdict) {
      case 'recommended': return 'text-green-600 bg-green-50 border-green-200';
      case 'conditional': return 'text-yellow-600 bg-yellow-50 border-yellow-200';
      case 'not_recommended': return 'text-red-600 bg-red-50 border-red-200';
      default: return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <BarChart3 className="w-8 h-8 text-blue-600 animate-pulse" />
          </div>
          <h2 className="text-xl font-bold text-gray-900 mb-2">차량 분석 중...</h2>
          <p className="text-gray-600">AI가 선택하신 차량을 종합 분석하고 있습니다</p>
        </div>
      </div>
    );
  }

  if (!analysis) return null;

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 헤더 */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-6 py-6">
          <div className="text-center">
            <h1 className="text-2xl font-bold text-gray-900 mb-2">차량 분석 리포트</h1>
            <p className="text-gray-600">AI가 선택하신 차량을 종합 분석한 결과입니다</p>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-6 py-8">
        {/* 선택된 차량 요약 */}
        <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6 mb-8">
          <div className="flex items-start justify-between">
            <div className="flex items-start gap-6">
              <div className="w-24 h-24 bg-gray-100 rounded-xl flex items-center justify-center">
                <Car className="w-12 h-12 text-gray-400" />
              </div>
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-1">
                  {selectedVehicle.brand} {selectedVehicle.model}
                </h2>
                <p className="text-gray-600 mb-3">{selectedVehicle.year}년식</p>
                <div className="flex items-center gap-4 text-sm text-gray-600">
                  <div className="flex items-center gap-1">
                    <Gauge className="w-4 h-4" />
                    <span>{selectedVehicle.mileage.toLocaleString()}km</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Fuel className="w-4 h-4" />
                    <span>{selectedVehicle.fuel_type}</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <MapPin className="w-4 h-4" />
                    <span>{selectedVehicle.location}</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-3xl font-bold text-blue-600 mb-1">
                {selectedVehicle.price.toLocaleString()}만원
              </div>
              <div className="flex items-center gap-1 text-sm text-green-600">
                <TrendingUp className="w-4 h-4" />
                매치율 {analysis.matchScore}%
              </div>
            </div>
          </div>
        </div>

        {/* 분석 결과 그리드 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* 가격 분석 */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                <DollarSign className="w-5 h-5 text-blue-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">가격 분석</h3>
            </div>

            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">시장 평균가</span>
                <span className="font-medium">{analysis.priceAnalysis.marketPrice.toLocaleString()}만원</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600">제시 가격</span>
                <span className="font-medium text-blue-600">{analysis.priceAnalysis.dealPrice.toLocaleString()}만원</span>
              </div>
              <div className="flex justify-between items-center pt-2 border-t border-gray-100">
                <span className="text-sm text-gray-600">절약 금액</span>
                <span className="font-bold text-green-600">-{analysis.priceAnalysis.savings.toLocaleString()}만원</span>
              </div>
              <div className={`px-3 py-2 rounded-lg text-center text-sm font-medium ${getPriceRatingColor(analysis.priceAnalysis.priceRating)}`}>
                {analysis.priceAnalysis.priceRating === 'excellent' && '매우 좋은 가격'}
                {analysis.priceAnalysis.priceRating === 'good' && '합리적인 가격'}
                {analysis.priceAnalysis.priceRating === 'fair' && '적정 가격'}
              </div>
            </div>
          </div>

          {/* 위험도 평가 */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-green-100 rounded-lg flex items-center justify-center">
                <Shield className="w-5 h-5 text-green-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">위험도 평가</h3>
            </div>

            <div className="space-y-3">
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600 mb-1">{analysis.riskAssessment.score}점</div>
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${getRiskColor(analysis.riskAssessment.overall)}`}>
                  {analysis.riskAssessment.overall === 'low' && '낮은 위험도'}
                  {analysis.riskAssessment.overall === 'medium' && '보통 위험도'}
                  {analysis.riskAssessment.overall === 'high' && '높은 위험도'}
                </div>
              </div>
              <div className="space-y-2">
                {analysis.riskAssessment.factors.map((factor, index) => (
                  <div key={index} className="flex items-center gap-2 text-sm text-gray-600">
                    <CheckCircle className="w-4 h-4 text-green-500" />
                    <span>{factor}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* 매치 스코어 */}
          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-purple-100 rounded-lg flex items-center justify-center">
                <Star className="w-5 h-5 text-purple-600" />
              </div>
              <h3 className="text-lg font-semibold text-gray-900">적합도 분석</h3>
            </div>

            <div className="space-y-3">
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600 mb-2">{analysis.matchScore}%</div>
                <div className="w-full bg-gray-200 rounded-full h-2 mb-3">
                  <div
                    className="bg-purple-600 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${analysis.matchScore}%` }}
                  ></div>
                </div>
              </div>
              <div className="text-center text-sm text-gray-600">
                당신의 취향과 요구사항에<br />
                <span className="font-semibold text-purple-600">{analysis.matchScore}% 일치</span>합니다
              </div>
            </div>
          </div>
        </div>

        {/* AI 종합 추천 */}
        <div className={`rounded-2xl border-2 p-6 mb-8 ${getRecommendationColor(analysis.recommendation.verdict)}`}>
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-white rounded-xl flex items-center justify-center">
              <Award className="w-6 h-6 text-blue-600" />
            </div>
            <div className="flex-1">
              <h3 className="text-xl font-bold text-gray-900 mb-2">
                AI 종합 추천
                <span className="ml-2 text-sm font-normal">
                  (신뢰도 {analysis.recommendation.confidence}%)
                </span>
              </h3>
              <div className="mb-4">
                {analysis.recommendation.verdict === 'recommended' && (
                  <div className="text-lg font-semibold text-green-700">✅ 추천합니다</div>
                )}
                {analysis.recommendation.verdict === 'conditional' && (
                  <div className="text-lg font-semibold text-yellow-700">⚠️ 조건부 추천</div>
                )}
                {analysis.recommendation.verdict === 'not_recommended' && (
                  <div className="text-lg font-semibold text-red-700">❌ 추천하지 않음</div>
                )}
              </div>
              <div className="space-y-2">
                {analysis.recommendation.reasons.map((reason, index) => (
                  <div key={index} className="flex items-center gap-2 text-gray-700">
                    <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                    <span>{reason}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* 액션 버튼 */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
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
              <h4 className="font-semibold text-blue-900 mb-1">분석 정보 안내</h4>
              <p className="text-sm text-blue-700">
                이 분석은 AI가 제공한 참고용 정보입니다.
                실제 차량 상태는 직접 확인하시고, 전문가의 검수를 받으시기 바랍니다.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}