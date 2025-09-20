'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Brain, Bot, TrendingUp, Sparkles, CheckCircle, Cpu, Database, Users, BarChart, Target, Zap, Eye, Rocket } from 'lucide-react';
import { UserInput, RecommendationResult } from '@/types';

interface SearchingPhaseProps {
  searchData: UserInput;
  onResultsReady: (results: RecommendationResult) => void;
}

export function SearchingPhase({ searchData, onResultsReady }: SearchingPhaseProps) {
  const [progress, setProgress] = useState(0);
  const [currentStep, setCurrentStep] = useState(0);
  const [agentProgress, setAgentProgress] = useState({
    data_analysis: 'pending',
    vector_search: 'pending',
    collaborative_filtering: 'pending',
    market_analysis: 'pending',
    personalization: 'pending'
  });

  const steps = [
    {
      id: 'data_analysis',
      name: '🧠 데이터 분석 에이전트',
      description: '사용자 선호도를 512차원 벡터로 변환',
      icon: Brain,
      color: 'purple',
      duration: 2000,
      detail: '예산, 연식, 차종을 AI가 분석 중...'
    },
    {
      id: 'vector_search',
      name: '🔍 벡터 검색 에이전트',
      description: '15만개 차량 벡터 데이터베이스 탐색',
      icon: Database,
      color: 'blue',
      duration: 1500,
      detail: '코사인 유사도로 최적 매칭 계산 중...'
    },
    {
      id: 'collaborative_filtering',
      name: '👥 협업 필터링 에이전트',
      description: '유사 사용자 5만명 패턴 분석',
      icon: Users,
      color: 'green',
      duration: 2500,
      detail: '당신과 비슷한 취향의 사용자들 찾는 중...'
    },
    {
      id: 'market_analysis',
      name: '📊 시장 분석 에이전트',
      description: '실시간 중고차 시장 트렌드 반영',
      icon: BarChart,
      color: 'orange',
      duration: 1800,
      detail: '가격 변동, 인기도, 재판매가 분석 중...'
    },
    {
      id: 'personalization',
      name: '🎯 개인화 에이전트',
      description: '최종 개인 맞춤 추천 리스트 생성',
      icon: Target,
      color: 'pink',
      duration: 1200,
      detail: '당신만을 위한 완벽한 추천 완성 중...'
    }
  ];

  useEffect(() => {
    let stepIndex = 0;
    let currentProgress = 0;

    const processStep = () => {
      if (stepIndex >= steps.length) {
        // 모든 단계 완료 후 결과 생성
        setTimeout(() => {
          const mockResult: RecommendationResult = {
            sessionId: searchData.sessionId,
            cars: [],
            totalMatches: 127,
            executionTime: 8.4,
            agentScores: {
              data_analysis: 0.94,
              vector_search: 0.91,
              collaborative_filtering: 0.89,
              market_analysis: 0.87,
              personalization: 0.96
            }
          };
          onResultsReady(mockResult);
        }, 800);
        return;
      }

      const step = steps[stepIndex];

      // 현재 단계 시작
      setCurrentStep(stepIndex);
      setAgentProgress(prev => ({
        ...prev,
        [step.id]: 'running'
      }));

      // 진행률 업데이트
      const interval = setInterval(() => {
        currentProgress += (100 / steps.length) / (step.duration / 50);
        setProgress(Math.min(currentProgress, (stepIndex + 1) * (100 / steps.length)));
      }, 50);

      // 단계 완료
      setTimeout(() => {
        clearInterval(interval);
        setAgentProgress(prev => ({
          ...prev,
          [step.id]: 'completed'
        }));

        stepIndex++;
        setTimeout(() => processStep(), 300);
      }, step.duration);
    };

    processStep();
  }, [searchData, onResultsReady]);

  const getStatusColor = (status: string, baseColor: string) => {
    switch (status) {
      case 'completed':
        return 'text-green-400 bg-green-400/20';
      case 'running':
        return `text-${baseColor}-400 bg-${baseColor}-400/20`;
      default:
        return 'text-gray-500 bg-gray-500/10';
    }
  };

  const getIconAnimation = (status: string) => {
    switch (status) {
      case 'running':
        return 'animate-spin';
      case 'completed':
        return 'animate-pulse';
      default:
        return '';
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 relative overflow-hidden">
      {/* 배경 애니메이션 */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-20 w-96 h-96 bg-blue-500/5 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-20 w-80 h-80 bg-purple-500/5 rounded-full blur-3xl animate-pulse delay-1000"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-72 h-72 bg-cyan-500/5 rounded-full blur-3xl animate-pulse delay-2000"></div>

        {/* 매트릭스 스타일 떠다니는 점들 */}
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-1 h-1 bg-cyan-400/30 rounded-full animate-bounce"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${2 + Math.random() * 2}s`
            }}
          />
        ))}
      </div>

      <div className="relative z-10 min-h-screen flex items-center justify-center px-4">
        <div className="max-w-5xl w-full">

          {/* 헤더 */}
          <div className="text-center mb-12">
            <div className="flex items-center justify-center mb-6">
              <div className="relative">
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full blur-xl opacity-50 animate-pulse"></div>
                <div className="relative bg-gradient-to-r from-cyan-600 to-blue-600 p-6 rounded-full">
                  <Bot className="w-16 h-16 text-white animate-spin-slow" />
                </div>
              </div>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-white via-cyan-200 to-blue-200 bg-clip-text text-transparent mb-4">
              AI 분석 진행중
            </h1>

            <div className="flex items-center justify-center gap-3 mb-8">
              <Sparkles className="w-6 h-6 text-yellow-400 animate-pulse" />
              <p className="text-xl text-gray-300">
                5개 AI 에이전트가 협력하여 최적의 차량을 찾고 있습니다
              </p>
              <Sparkles className="w-6 h-6 text-yellow-400 animate-pulse" />
            </div>

            {/* 전체 진행률 */}
            <div className="mb-8 max-w-2xl mx-auto">
              <div className="flex justify-between items-center mb-3">
                <span className="text-white font-medium">전체 진행률</span>
                <span className="text-cyan-400 font-bold">{Math.round(progress)}%</span>
              </div>
              <div className="relative">
                <Progress
                  value={progress}
                  className="h-3 bg-white/10 rounded-full overflow-hidden"
                />
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-500/20 rounded-full animate-pulse"></div>
              </div>
            </div>
          </div>

          {/* 에이전트 상태 */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4 md:gap-6 mb-12">
            {steps.map((step, index) => {
              const status = agentProgress[step.id];
              const IconComponent = step.icon;

              return (
                <Card
                  key={step.id}
                  className={`relative overflow-hidden border transition-all duration-500 transform ${
                    status === 'running'
                      ? `bg-${step.color}-500/10 border-${step.color}-500/30 scale-105 shadow-2xl shadow-${step.color}-500/20`
                      : status === 'completed'
                      ? 'bg-green-500/10 border-green-500/30 shadow-lg shadow-green-500/20'
                      : 'bg-white/5 border-white/10'
                  }`}
                >
                  {/* 배경 그라데이션 */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${
                    status === 'running'
                      ? `from-${step.color}-500/5 to-transparent`
                      : status === 'completed'
                      ? 'from-green-500/5 to-transparent'
                      : 'from-transparent to-transparent'
                  }`} />

                  <CardContent className="p-6 relative z-10">
                    <div className="flex items-center gap-4 mb-4">
                      <div className={`p-3 rounded-full ${getStatusColor(status, step.color)}`}>
                        {status === 'completed' ? (
                          <CheckCircle className="w-8 h-8 text-green-400" />
                        ) : (
                          <IconComponent className={`w-8 h-8 ${getIconAnimation(status)}`} />
                        )}
                      </div>
                      <div className="flex-1">
                        <h3 className="text-lg font-semibold text-white mb-1">
                          {step.name}
                        </h3>
                        <p className={`text-sm ${
                          status === 'running' ? `text-${step.color}-300` : 'text-gray-400'
                        }`}>
                          {step.description}
                        </p>
                      </div>
                    </div>

                    {/* 세부 진행 상황 */}
                    {status === 'running' && (
                      <div className="space-y-2">
                        <div className={`w-full h-1 bg-${step.color}-500/20 rounded-full overflow-hidden`}>
                          <div className={`h-full bg-${step.color}-500 rounded-full animate-pulse`}
                               style={{ width: '70%' }} />
                        </div>
                        <p className="text-xs text-gray-400 animate-pulse">
                          {step.detail}
                        </p>
                      </div>
                    )}

                    {status === 'completed' && (
                      <div className="flex items-center gap-2 text-green-400 text-sm">
                        <CheckCircle className="w-4 h-4" />
                        <span>완료됨</span>
                      </div>
                    )}
                  </CardContent>
                </Card>
              );
            })}
          </div>

          {/* 실시간 통계 */}
          <Card className="bg-white/5 border-white/10 backdrop-blur-sm">
            <CardContent className="p-8">
              <h3 className="text-xl font-semibold text-white mb-6 text-center flex items-center justify-center gap-2">
                <BarChart className="w-6 h-6 text-cyan-400" />
                실시간 분석 현황
                <Eye className="w-6 h-6 text-cyan-400" />
              </h3>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 md:gap-6">
                <div className="text-center">
                  <div className="text-3xl font-bold text-cyan-400 mb-2 animate-pulse">
                    {progress < 20 ? '15만+' : progress < 60 ? '12만+' : '8만+'}
                  </div>
                  <div className="text-sm text-gray-400">분석된 차량</div>
                </div>

                <div className="text-center">
                  <div className="text-3xl font-bold text-blue-400 mb-2 animate-pulse">
                    {progress < 40 ? '5만+' : progress < 80 ? '3만+' : '127'}
                  </div>
                  <div className="text-sm text-gray-400">매칭 후보</div>
                </div>

                <div className="text-center">
                  <div className="text-3xl font-bold text-purple-400 mb-2 animate-pulse">
                    {Math.round(progress * 0.95)}%
                  </div>
                  <div className="text-sm text-gray-400">정확도</div>
                </div>

                <div className="text-center">
                  <div className="text-3xl font-bold text-pink-400 mb-2 animate-pulse">
                    {(progress * 0.084).toFixed(1)}초
                  </div>
                  <div className="text-sm text-gray-400">경과 시간</div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* 로딩 메시지 */}
          {currentStep < steps.length && (
            <div className="text-center mt-8">
              <div className="flex items-center justify-center gap-2 text-gray-300">
                <Rocket className="w-5 h-5 animate-bounce" />
                <span className="animate-pulse">
                  {steps[currentStep]?.detail || '분석 중...'}
                </span>
                <Zap className="w-5 h-5 animate-bounce delay-500" />
              </div>
            </div>
          )}
        </div>
      </div>

      <style jsx>{`
        @keyframes spin-slow {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }

        .animate-spin-slow {
          animation: spin-slow 4s linear infinite;
        }
      `}</style>
    </div>
  );
}