'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { CheckCircle, Star, TrendingUp, Zap } from 'lucide-react';
import { RecommendationResult, ProcessedCarData } from '@/types';
import { MOCK_CARS } from '@/lib/data/mock-data';

interface ComparisonAnalysis {
  advantages: Record<string, string[]>;
  disadvantages: Record<string, string[]>;
  bestChoice: string;
  reasoning: string;
}

interface AdvancedComparisonProps {
  selectedCarIds: string[];
  recommendations: RecommendationResult | null;
  onFinalDecision: () => void;
}

export function AdvancedComparison({
  selectedCarIds,
  recommendations: _recommendations,
  onFinalDecision
}: AdvancedComparisonProps) {
  const [comparisonCars, setComparisonCars] = useState<ProcessedCarData[]>([]);
  const [preferredCar, setPreferredCar] = useState<string | null>(null);
  const [comparisonAnalysis, setComparisonAnalysis] = useState<ComparisonAnalysis | null>(null);

  useEffect(() => {
    // 선택된 차량들 로드
    const selectedCars = MOCK_CARS.filter(car => selectedCarIds.includes(car.id));
    setComparisonCars(selectedCars.slice(0, 3)); // 최대 3개까지

    // AI 비교 분석 시뮬레이션
    generateComparisonAnalysis(selectedCars);
  }, [selectedCarIds]);

  const generateComparisonAnalysis = (cars: ProcessedCarData[]) => {
    // AI 비교 분석 시뮬레이션
    const analysis = {
      winner: cars[0]?.id,
      summary: '종합적으로 첫 번째 차량이 가성비와 실용성 면에서 우수합니다.',
      categories: {
        value_for_money: {
          winner: cars[0]?.id,
          scores: cars.map(car => ({
            carId: car.id,
            score: Math.random() * 30 + 70 // 70-100점
          }))
        },
        reliability: {
          winner: cars[1]?.id || cars[0]?.id,
          scores: cars.map(car => ({
            carId: car.id,
            score: Math.random() * 25 + 75 // 75-100점
          }))
        },
        fuel_efficiency: {
          winner: cars[Math.floor(Math.random() * cars.length)]?.id,
          scores: cars.map(car => ({
            carId: car.id,
            score: Math.random() * 30 + 60 // 60-90점
          }))
        },
        resale_value: {
          winner: cars[Math.floor(Math.random() * cars.length)]?.id,
          scores: cars.map(car => ({
            carId: car.id,
            score: Math.random() * 40 + 50 // 50-90점
          }))
        }
      },
      pros_cons: cars.map(car => ({
        carId: car.id,
        pros: generatePros(car),
        cons: generateCons(car)
      }))
    };

    setComparisonAnalysis(analysis);
  };

  const generatePros = (car: ProcessedCarData): string[] => {
    const allPros = [
      '우수한 연비',
      '낮은 유지비용',
      '높은 안전성',
      '뛰어난 디자인',
      '넓은 실내공간',
      '최신 기술',
      '좋은 브랜드 신뢰도',
      '저렴한 가격',
      '낮은 주행거리'
    ];

    return allPros.sort(() => 0.5 - Math.random()).slice(0, 3);
  };

  const generateCons = (car: ProcessedCarData): string[] => {
    const allCons = [
      '높은 가격',
      '높은 유지비',
      '낡은 연식',
      '높은 주행거리',
      '제한적인 옵션',
      '부족한 A/S',
      '재판매가 우려'
    ];

    return allCons.sort(() => 0.5 - Math.random()).slice(0, 2);
  };

  const handleCarSelection = (carId: string) => {
    setPreferredCar(carId);
  };

  const getScoreForCar = (category: { scores?: { carId: string; score: number }[] }, carId: string): number => {
    const score = category.scores?.find((s: { carId: string; score: number }) => s.carId === carId)?.score;
    return score || 75;
  };

  const formatPrice = (price: number) => {
    return `${price.toLocaleString()}만원`;
  };

  const formatMileage = (mileage: number) => {
    return `${Math.round(mileage / 1000)}만km`;
  };

  if (comparisonCars.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p>비교할 차량을 선택해주세요.</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* 헤더 */}
      <div className="mb-8 text-center">
        <h2 className="text-3xl font-bold text-gray-900 mb-2">
          ⚖️ AI 차량 비교 분석
        </h2>
        <p className="text-gray-600">
          선택하신 {comparisonCars.length}개 차량을 종합적으로 분석했습니다
        </p>
      </div>

      {/* AI 종합 분석 결과 */}
      {comparisonAnalysis && (
        <Card className="mb-8 bg-gradient-to-r from-blue-50 to-indigo-50">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="w-6 h-6 text-blue-600" />
              AI 종합 분석 결과
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-semibold mb-2">🏆 종합 추천</h3>
                <p className="text-gray-700 mb-4">{comparisonAnalysis.summary}</p>

                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">가성비</span>
                    <Badge className={comparisonAnalysis.categories.value_for_money.winner === comparisonCars[0]?.id ? 'bg-green-500' : ''}>
                      {comparisonCars.find(car => car.id === comparisonAnalysis.categories.value_for_money.winner)?.brand} {comparisonCars.find(car => car.id === comparisonAnalysis.categories.value_for_money.winner)?.model}
                    </Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">신뢰성</span>
                    <Badge className={comparisonAnalysis.categories.reliability.winner === comparisonCars[0]?.id ? 'bg-green-500' : ''}>
                      {comparisonCars.find(car => car.id === comparisonAnalysis.categories.reliability.winner)?.brand} {comparisonCars.find(car => car.id === comparisonAnalysis.categories.reliability.winner)?.model}
                    </Badge>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="font-semibold mb-4">📊 카테고리별 우승</h3>
                <div className="space-y-3">
                  {Object.entries(comparisonAnalysis.categories).map(([category, data]: [string, { scores?: { carId: string; score: number }[] }]) => (
                    <div key={category} className="flex items-center justify-between">
                      <span className="text-sm capitalize">{category.replace('_', ' ')}</span>
                      <div className="flex items-center gap-2">
                        <Star className="w-4 h-4 text-yellow-500" />
                        <span className="text-sm font-medium">
                          {comparisonCars.find(car => car.id === data.winner)?.brand} {comparisonCars.find(car => car.id === data.winner)?.model}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* 차량 비교 테이블 - 반응형 그리드 */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6 mb-8">
        {comparisonCars.map((car, index) => (
          <Card
            key={car.id}
            className={`cursor-pointer transition-all duration-300 ${
              preferredCar === car.id
                ? 'ring-2 ring-green-500 bg-green-50'
                : 'hover:shadow-lg'
            }`}
            onClick={() => handleCarSelection(car.id)}
          >
            <CardHeader>
              {/* 차량 이미지 */}
              <div className="relative h-40 bg-gray-100 rounded-lg overflow-hidden mb-4">
                <img
                  src={car.images?.[0] || `https://picsum.photos/400/300?random=${car.id}`}
                  alt={`${car.brand} ${car.model}`}
                  className="w-full h-full object-cover"
                />

                {preferredCar === car.id && (
                  <div className="absolute top-2 right-2">
                    <CheckCircle className="w-6 h-6 text-green-500 bg-white rounded-full" />
                  </div>
                )}

                {comparisonAnalysis?.winner === car.id && (
                  <div className="absolute top-2 left-2">
                    <Badge className="bg-yellow-500 text-white">
                      🏆 AI 추천
                    </Badge>
                  </div>
                )}
              </div>

              <CardTitle className="text-lg">
                {car.brand} {car.model}
              </CardTitle>
            </CardHeader>

            <CardContent>
              {/* 기본 정보 */}
              <div className="space-y-4">
                <div className="text-center">
                  <p className="text-2xl font-bold text-blue-600">
                    {formatPrice(car.price)}
                  </p>
                  <p className="text-sm text-gray-500">
                    {car.year}년 • {formatMileage(car.mileage)}
                  </p>
                </div>

                {/* 점수 */}
                {comparisonAnalysis && (
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm">카테고리별 점수</h4>
                    {Object.entries(comparisonAnalysis.categories).map(([category, data]: [string, { scores?: { carId: string; score: number }[] }]) => (
                      <div key={category} className="flex justify-between items-center">
                        <span className="text-xs text-gray-600 capitalize">
                          {category.replace('_', ' ')}
                        </span>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div
                              className={`h-full ${data.winner === car.id ? 'bg-green-500' : 'bg-blue-500'}`}
                              style={{ width: `${getScoreForCar(data, car.id)}%` }}
                            />
                          </div>
                          <span className="text-xs font-medium">
                            {Math.round(getScoreForCar(data, car.id))}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* 장단점 */}
                {comparisonAnalysis && (
                  <div className="space-y-2">
                    <h4 className="font-medium text-sm text-green-700">👍 장점</h4>
                    <ul className="text-xs space-y-1">
                      {comparisonAnalysis.pros_cons
                        .find((pc: { carId: string; pros?: string[] }) => pc.carId === car.id)?.pros
                        .map((pro: string, i: number) => (
                          <li key={i} className="text-green-600">• {pro}</li>
                        ))}
                    </ul>

                    <h4 className="font-medium text-sm text-red-700">👎 단점</h4>
                    <ul className="text-xs space-y-1">
                      {comparisonAnalysis.pros_cons
                        .find((pc: { carId: string; cons?: string[] }) => pc.carId === car.id)?.cons
                        .map((con: string, i: number) => (
                          <li key={i} className="text-red-600">• {con}</li>
                        ))}
                    </ul>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* 최종 결정 */}
      <Card className="bg-gradient-to-r from-green-50 to-emerald-50">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-6 h-6 text-green-600" />
            최종 결정 도움
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-center space-y-4">
            {preferredCar ? (
              <>
                <p className="text-lg">
                  <strong>
                    {comparisonCars.find(car => car.id === preferredCar)?.brand}{' '}
                    {comparisonCars.find(car => car.id === preferredCar)?.model}
                  </strong>
                  을(를) 선택하셨습니다!
                </p>
                <p className="text-gray-600">
                  이 차량은 당신의 요구사항에 매우 적합한 선택입니다.
                </p>
                <div className="flex gap-4 justify-center">
                  <Button onClick={onFinalDecision} className="bg-green-600 hover:bg-green-700">
                    <CheckCircle className="w-4 h-4 mr-2" />
                    최종 확정하기
                  </Button>
                  <Button variant="outline" onClick={() => setPreferredCar(null)}>
                    다시 선택하기
                  </Button>
                </div>
              </>
            ) : (
              <>
                <p className="text-lg">위 차량 중 하나를 선택해주세요</p>
                <p className="text-gray-600">
                  각 차량을 클릭하면 최종 결정을 도와드립니다.
                </p>
              </>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}