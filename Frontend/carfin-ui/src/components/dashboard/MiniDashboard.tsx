'use client';

import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Progress } from '@/components/ui/progress';
import { Badge } from '@/components/ui/badge';
import {
  TrendingUp,
  TrendingDown,
  BarChart3,
  PieChart,
  Activity,
  DollarSign,
  Car,
  Fuel,
  Calendar,
  Target,
  Eye,
  Brain
} from 'lucide-react';

interface MarketInsight {
  category: string;
  value: number;
  trend: 'up' | 'down' | 'stable';
  change: number;
}

interface PriceDistribution {
  range: string;
  count: number;
  percentage: number;
}

interface DashboardData {
  marketInsights: MarketInsight[];
  priceDistribution: PriceDistribution[];
  popularBrands: { brand: string; count: number; satisfaction: number }[];
  recommendations: {
    totalAnalyzed: number;
    processingTime: number;
    confidenceScore: number;
    userProfile: string;
  };
}

interface MiniDashboardProps {
  data?: DashboardData;
  loading?: boolean;
  userBudget?: [number, number];
  userPreferences?: {
    fuelType?: string;
    bodyType?: string;
  };
}

export function MiniDashboard({
  data,
  loading = false,
  userBudget,
  userPreferences
}: MiniDashboardProps) {
  const [activeTab, setActiveTab] = useState<'market' | 'insights' | 'recommendations'>('market');

  // 샘플 데이터 (실제로는 API에서 가져올 것)
  const sampleData: DashboardData = {
    marketInsights: [
      { category: '평균 가격', value: 2450, trend: 'up', change: 5.2 },
      { category: '연비 향상', value: 78, trend: 'up', change: 12.1 },
      { category: '안전성 점수', value: 92, trend: 'stable', change: 1.3 },
      { category: '중고차 공급', value: 1540, trend: 'down', change: -8.7 }
    ],
    priceDistribution: [
      { range: '1000-2000만원', count: 320, percentage: 28 },
      { range: '2000-3000만원', count: 450, percentage: 39 },
      { range: '3000-4000만원', count: 280, percentage: 24 },
      { range: '4000만원+', count: 110, percentage: 9 }
    ],
    popularBrands: [
      { brand: '현대', count: 180, satisfaction: 4.2 },
      { brand: '기아', count: 150, satisfaction: 4.1 },
      { brand: 'BMW', count: 95, satisfaction: 4.5 },
      { brand: '벤츠', count: 80, satisfaction: 4.4 },
      { brand: '토요타', count: 75, satisfaction: 4.3 }
    ],
    recommendations: {
      totalAnalyzed: 15420,
      processingTime: 2.3,
      confidenceScore: 94.2,
      userProfile: 'family_oriented'
    }
  };

  const dashboardData = data || sampleData;

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {[...Array(6)].map((_, i) => (
          <Card key={i} className="bg-white/10 border-white/20 backdrop-blur-xl animate-pulse">
            <CardContent className="p-6">
              <div className="h-4 bg-white/20 rounded mb-4"></div>
              <div className="h-8 bg-white/30 rounded mb-2"></div>
              <div className="h-3 bg-white/20 rounded w-2/3"></div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 대시보드 헤더 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">AI 시장 분석 대시보드</h2>
            <p className="text-gray-400">실시간 차량 시장 종합 분석 결과</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Eye className="w-4 h-4 text-gray-400" />
          <span className="text-sm text-gray-400">
            {dashboardData.recommendations.totalAnalyzed.toLocaleString()}대 분석 완료
          </span>
        </div>
      </div>

      {/* 탭 네비게이션 */}
      <div className="flex space-x-1 bg-white/10 backdrop-blur-xl rounded-xl p-1">
        {[
          { key: 'market' as const, label: '시장 동향', icon: BarChart3 },
          { key: 'insights' as const, label: '인사이트', icon: Target },
          { key: 'recommendations' as const, label: 'AI 분석', icon: Brain }
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
              activeTab === tab.key
                ? 'bg-gradient-to-r from-purple-600 to-pink-600 text-white shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-white/10'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* 탭 컨텐츠 */}
      {activeTab === 'market' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {dashboardData.marketInsights.map((insight, index) => (
            <Card key={index} className="bg-white/10 border-white/20 backdrop-blur-xl hover:bg-white/15 transition-all duration-300">
              <CardContent className="p-6">
                <div className="flex items-center justify-between mb-4">
                  <div className="p-2 bg-white/10 rounded-lg">
                    {insight.category === '평균 가격' && <DollarSign className="w-5 h-5 text-green-400" />}
                    {insight.category === '연비 향상' && <Fuel className="w-5 h-5 text-blue-400" />}
                    {insight.category === '안전성 점수' && <Target className="w-5 h-5 text-yellow-400" />}
                    {insight.category === '중고차 공급' && <Car className="w-5 h-5 text-purple-400" />}
                  </div>
                  <div className={`flex items-center gap-1 text-sm ${
                    insight.trend === 'up' ? 'text-green-400' :
                    insight.trend === 'down' ? 'text-red-400' : 'text-gray-400'
                  }`}>
                    {insight.trend === 'up' ? <TrendingUp className="w-4 h-4" /> :
                     insight.trend === 'down' ? <TrendingDown className="w-4 h-4" /> :
                     <Activity className="w-4 h-4" />}
                    {Math.abs(insight.change)}%
                  </div>
                </div>
                <div className="text-2xl font-bold text-white mb-1">
                  {insight.category === '평균 가격' || insight.category === '중고차 공급'
                    ? `${insight.value.toLocaleString()}${insight.category === '평균 가격' ? '만원' : '대'}`
                    : `${insight.value}${insight.category === '연비 향상' ? '%' : '점'}`
                  }
                </div>
                <div className="text-sm text-gray-400">{insight.category}</div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {activeTab === 'insights' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* 가격 분포 */}
          <Card className="bg-white/10 border-white/20 backdrop-blur-xl">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <PieChart className="w-5 h-5 text-blue-400" />
                가격대별 분포
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {dashboardData.priceDistribution.map((dist, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-300">{dist.range}</span>
                    <span className="text-white font-medium">{dist.count}대 ({dist.percentage}%)</span>
                  </div>
                  <Progress value={dist.percentage} className="h-2" />
                </div>
              ))}
            </CardContent>
          </Card>

          {/* 인기 브랜드 */}
          <Card className="bg-white/10 border-white/20 backdrop-blur-xl">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-purple-400" />
                브랜드별 인기도
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {dashboardData.popularBrands.map((brand, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
                      {index + 1}
                    </div>
                    <div>
                      <div className="text-white font-medium">{brand.brand}</div>
                      <div className="text-sm text-gray-400">{brand.count}대 등록</div>
                    </div>
                  </div>
                  <Badge
                    variant="secondary"
                    className="bg-white/10 text-yellow-400 border-yellow-400/20"
                  >
                    ★ {brand.satisfaction}
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      )}

      {activeTab === 'recommendations' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="bg-gradient-to-br from-purple-900/50 to-pink-900/50 border-purple-500/20 backdrop-blur-xl">
            <CardContent className="p-6 text-center">
              <Brain className="w-12 h-12 text-purple-400 mx-auto mb-4 animate-pulse" />
              <div className="text-3xl font-bold text-white mb-2">
                {dashboardData.recommendations.confidenceScore}%
              </div>
              <div className="text-sm text-gray-300">AI 신뢰도 점수</div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-blue-900/50 to-cyan-900/50 border-blue-500/20 backdrop-blur-xl">
            <CardContent className="p-6 text-center">
              <Activity className="w-12 h-12 text-blue-400 mx-auto mb-4 animate-pulse delay-300" />
              <div className="text-3xl font-bold text-white mb-2">
                {dashboardData.recommendations.processingTime}초
              </div>
              <div className="text-sm text-gray-300">분석 처리 시간</div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-br from-green-900/50 to-emerald-900/50 border-green-500/20 backdrop-blur-xl">
            <CardContent className="p-6 text-center">
              <Target className="w-12 h-12 text-green-400 mx-auto mb-4 animate-pulse delay-500" />
              <div className="text-3xl font-bold text-white mb-2">
                {dashboardData.recommendations.totalAnalyzed.toLocaleString()}
              </div>
              <div className="text-sm text-gray-300">총 분석 차량</div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* 사용자 맞춤 인사이트 */}
      {userBudget && (
        <Card className="bg-gradient-to-r from-indigo-900/30 to-purple-900/30 border-indigo-500/20 backdrop-blur-xl">
          <CardContent className="p-6">
            <div className="flex items-center gap-3 mb-4">
              <div className="p-2 bg-indigo-600 rounded-lg">
                <Target className="w-5 h-5 text-white" />
              </div>
              <h3 className="text-lg font-semibold text-white">맞춤 추천 인사이트</h3>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="text-gray-300">
                <span className="text-indigo-400 font-medium">예산 범위:</span> {userBudget[0].toLocaleString()} - {userBudget[1].toLocaleString()}만원
              </div>
              {userPreferences?.fuelType && (
                <div className="text-gray-300">
                  <span className="text-indigo-400 font-medium">선호 연료:</span> {userPreferences.fuelType}
                </div>
              )}
              {userPreferences?.bodyType && (
                <div className="text-gray-300">
                  <span className="text-indigo-400 font-medium">선호 차종:</span> {userPreferences.bodyType}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}