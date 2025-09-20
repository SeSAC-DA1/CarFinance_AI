'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Car, Bot, Sparkles, ArrowRight, Search, Zap, Brain, Target, Star, Rocket, Shield } from 'lucide-react';
import { UserInput } from '@/types';

interface CarFinderHomeProps {
  onSubmit: (input: UserInput) => void;
}

export function CarFinderHome({ onSubmit }: CarFinderHomeProps) {
  const [carModel, setCarModel] = useState('');
  const [yearRange, setYearRange] = useState<[number, number]>([2018, 2024]);
  const [budgetRange, setBudgetRange] = useState<[number, number]>([1500, 3000]);
  const [fuelType, setFuelType] = useState<string>('');
  const [bodyType, setBodyType] = useState<string>('');

  const handleSubmit = () => {
    const input: UserInput = {
      sessionId: `session_${Date.now()}`,
      carModel: carModel || undefined,
      yearRange,
      budgetRange,
      preferences: {
        fuelType: (fuelType as 'gasoline' | 'diesel' | 'hybrid' | 'electric') || undefined,
        bodyType: (bodyType as 'sedan' | 'suv' | 'hatchback' | 'coupe') || undefined,
        priorities: {
          price: 0.3,
          fuel_efficiency: 0.2,
          safety: 0.2,
          performance: 0.15,
          design: 0.15
        }
      }
    };

    onSubmit(input);
  };

  const isFormValid = budgetRange[0] > 0 && yearRange[0] > 2010;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* 배경 애니메이션 요소들 */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-10 w-72 h-72 bg-purple-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute top-40 right-10 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute bottom-20 left-1/3 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl animate-pulse delay-2000"></div>

        {/* 떠다니는 자동차 아이콘들 */}
        <div className="absolute top-32 left-1/4 text-white/5 animate-bounce delay-500">
          <Car size={40} />
        </div>
        <div className="absolute top-20 right-1/4 text-white/5 animate-bounce delay-1000">
          <Car size={30} />
        </div>
        <div className="absolute bottom-40 left-1/5 text-white/5 animate-bounce delay-1500">
          <Car size={35} />
        </div>
      </div>

      <div className="relative z-10 min-h-screen flex items-center justify-center px-4">
        <div className="max-w-6xl w-full">

          {/* HERO 섹션 */}
          <div className="text-center mb-16">
            <div className="flex items-center justify-center mb-8">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full blur-xl opacity-50 animate-pulse"></div>
                <div className="relative bg-gradient-to-r from-purple-600 to-pink-600 p-6 rounded-full">
                  <Bot className="w-12 h-12 text-white animate-spin-slow" />
                </div>
              </div>
            </div>

            <h1 className="text-6xl md:text-7xl font-bold bg-gradient-to-r from-white via-purple-200 to-pink-200 bg-clip-text text-transparent mb-6 animate-fade-in">
              AI 카파인더
            </h1>

            <div className="flex items-center justify-center gap-2 mb-8">
              <Sparkles className="w-6 h-6 text-yellow-400 animate-pulse" />
              <p className="text-xl text-gray-300 animate-fade-in-delay">
                멀티 에이전트 AI가 당신의 완벽한 차량을 찾아드립니다
              </p>
              <Sparkles className="w-6 h-6 text-yellow-400 animate-pulse" />
            </div>

            {/* 특징 카드들 */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6 mb-12 max-w-5xl mx-auto">
              <Card className="bg-white/5 border-white/10 backdrop-blur-sm hover:bg-white/10 transition-all duration-500 transform hover:scale-105 hover:rotate-1">
                <CardContent className="p-6 text-center">
                  <Brain className="w-12 h-12 text-purple-400 mx-auto mb-4 animate-pulse" />
                  <h3 className="text-lg font-semibold text-white mb-2">협업 필터링</h3>
                  <p className="text-gray-400 text-sm">실시간 학습으로 개인 맞춤 추천</p>
                </CardContent>
              </Card>

              <Card className="bg-white/5 border-white/10 backdrop-blur-sm hover:bg-white/10 transition-all duration-500 transform hover:scale-105 hover:-rotate-1">
                <CardContent className="p-6 text-center">
                  <Target className="w-12 h-12 text-blue-400 mx-auto mb-4 animate-pulse delay-500" />
                  <h3 className="text-lg font-semibold text-white mb-2">정밀 매칭</h3>
                  <p className="text-gray-400 text-sm">AI가 분석한 최적의 선택</p>
                </CardContent>
              </Card>

              <Card className="bg-white/5 border-white/10 backdrop-blur-sm hover:bg-white/10 transition-all duration-500 transform hover:scale-105 hover:rotate-1">
                <CardContent className="p-6 text-center">
                  <Zap className="w-12 h-12 text-yellow-400 mx-auto mb-4 animate-pulse delay-1000" />
                  <h3 className="text-lg font-semibold text-white mb-2">즉시 추천</h3>
                  <p className="text-gray-400 text-sm">3초 만에 완벽한 매칭</p>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* 입력 폼 */}
          <Card className="bg-white/10 border-white/20 backdrop-blur-xl shadow-2xl max-w-3xl mx-auto">
            <CardHeader>
              <CardTitle className="text-center text-2xl text-white flex items-center justify-center gap-3">
                <Search className="w-8 h-8 text-purple-400 animate-bounce" />
                당신의 드림카를 찾아보세요
                <Rocket className="w-8 h-8 text-pink-400 animate-bounce delay-500" />
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-8 p-8">

              {/* 예산 설정 */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-gradient-to-r from-green-400 to-emerald-500 rounded-full animate-pulse"></div>
                  <label className="text-lg font-medium text-white">
                    예산 범위: {budgetRange[0].toLocaleString()}만원 ~ {budgetRange[1].toLocaleString()}만원
                  </label>
                </div>
                <div className="relative">
                  <Slider
                    value={budgetRange}
                    onValueChange={(value) => setBudgetRange(value as [number, number])}
                    max={8000}
                    min={500}
                    step={100}
                    className="w-full slider-animated"
                  />
                  <div className="flex justify-between text-sm text-gray-400 mt-2">
                    <span>500만원</span>
                    <span>8,000만원</span>
                  </div>
                </div>
              </div>

              {/* 연식 설정 */}
              <div className="space-y-4">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-gradient-to-r from-blue-400 to-cyan-500 rounded-full animate-pulse delay-300"></div>
                  <label className="text-lg font-medium text-white">
                    연식 범위: {yearRange[0]}년 ~ {yearRange[1]}년
                  </label>
                </div>
                <div className="relative">
                  <Slider
                    value={yearRange}
                    onValueChange={(value) => setYearRange(value as [number, number])}
                    max={2024}
                    min={2010}
                    step={1}
                    className="w-full slider-animated"
                  />
                  <div className="flex justify-between text-sm text-gray-400 mt-2">
                    <span>2010년</span>
                    <span>2024년</span>
                  </div>
                </div>
              </div>

              {/* 선택사항들 */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Shield className="w-5 h-5 text-purple-400" />
                    <label className="text-white font-medium">연료 타입</label>
                  </div>
                  <Select value={fuelType} onValueChange={setFuelType}>
                    <SelectTrigger className="bg-white/10 border-white/20 text-white hover:bg-white/20 transition-all">
                      <SelectValue placeholder="선택하세요" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-600">
                      <SelectItem value="gasoline" className="text-white hover:bg-slate-700">가솔린</SelectItem>
                      <SelectItem value="diesel" className="text-white hover:bg-slate-700">디젤</SelectItem>
                      <SelectItem value="hybrid" className="text-white hover:bg-slate-700">하이브리드</SelectItem>
                      <SelectItem value="electric" className="text-white hover:bg-slate-700">전기</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <Car className="w-5 h-5 text-pink-400" />
                    <label className="text-white font-medium">차종</label>
                  </div>
                  <Select value={bodyType} onValueChange={setBodyType}>
                    <SelectTrigger className="bg-white/10 border-white/20 text-white hover:bg-white/20 transition-all">
                      <SelectValue placeholder="선택하세요" />
                    </SelectTrigger>
                    <SelectContent className="bg-slate-800 border-slate-600">
                      <SelectItem value="sedan" className="text-white hover:bg-slate-700">세단</SelectItem>
                      <SelectItem value="suv" className="text-white hover:bg-slate-700">SUV</SelectItem>
                      <SelectItem value="hatchback" className="text-white hover:bg-slate-700">해치백</SelectItem>
                      <SelectItem value="coupe" className="text-white hover:bg-slate-700">쿠페</SelectItem>
                      <SelectItem value="wagon" className="text-white hover:bg-slate-700">왜건</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* 선호 차량 모델 */}
              <div className="space-y-3">
                <div className="flex items-center gap-2">
                  <Star className="w-5 h-5 text-yellow-400" />
                  <label className="text-white font-medium">선호 브랜드/모델 (선택사항)</label>
                </div>
                <Input
                  value={carModel}
                  onChange={(e) => setCarModel(e.target.value)}
                  placeholder="예: 현대 아반떼, BMW 3시리즈..."
                  className="bg-white/10 border-white/20 text-white placeholder:text-gray-400 focus:bg-white/20 transition-all"
                />
              </div>

              {/* 제출 버튼 */}
              <div className="pt-6">
                <Button
                  onClick={handleSubmit}
                  disabled={!isFormValid}
                  className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold py-4 px-8 rounded-xl text-lg shadow-2xl transform transition-all duration-300 hover:scale-105 hover:shadow-purple-500/25 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none"
                >
                  <div className="flex items-center justify-center gap-3">
                    <Bot className="w-6 h-6 animate-spin-slow" />
                    AI로 완벽한 차량 찾기
                    <ArrowRight className="w-6 h-6 animate-bounce" />
                  </div>
                </Button>
              </div>

              {/* 통계 정보 */}
              <div className="grid grid-cols-3 gap-4 pt-6 border-t border-white/10">
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-400">15만+</div>
                  <div className="text-sm text-gray-400">분석 완료</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-pink-400">98%</div>
                  <div className="text-sm text-gray-400">만족도</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-yellow-400">3초</div>
                  <div className="text-sm text-gray-400">평균 추천</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      <style jsx>{`
        @keyframes fade-in {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes fade-in-delay {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }

        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .animate-fade-in {
          animation: fade-in 1s ease-out;
        }

        .animate-fade-in-delay {
          animation: fade-in-delay 1s ease-out 0.5s both;
        }

        .animate-spin-slow {
          animation: spin-slow 3s linear infinite;
        }

        .slider-animated .range-slider-thumb {
          transition: all 0.2s ease;
        }

        .slider-animated .range-slider-thumb:hover {
          transform: scale(1.2);
        }
      `}</style>
    </div>
  );
}