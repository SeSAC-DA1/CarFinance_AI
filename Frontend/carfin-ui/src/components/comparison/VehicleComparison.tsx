'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Car,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Shield,
  Fuel,
  Settings,
  Calendar,
  MapPin,
  Star,
  CheckCircle,
  XCircle,
  AlertCircle,
  CreditCard,
  Target,
  BarChart3
} from 'lucide-react';

interface DetailedAnalysis {
  pros: string[];
  cons: string[];
  market_position: string;
  depreciation_rate: number;
  maintenance_cost: number;
  insurance_cost: number;
  resale_value: number;
  reliability_score: number;
  safety_rating: number;
}

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
  detailed_analysis?: DetailedAnalysis;
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

interface VehicleComparisonProps {
  vehicles: Vehicle[];
  onSelectVehicle: (vehicleId: string) => void;
  onRequestFinancing: (vehicleId: string) => void;
  userProfile?: UserProfile;
}

export function VehicleComparison({
  vehicles,
  onSelectVehicle,
  onRequestFinancing,
  userProfile
}: VehicleComparisonProps) {
  const [selectedTab, setSelectedTab] = useState<'overview' | 'detailed' | 'financial'>('overview');

  const getScoreColor = (score: number) => {
    if (score >= 85) return 'text-green-600';
    if (score >= 70) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getScoreIcon = (score: number) => {
    if (score >= 85) return <CheckCircle className="w-4 h-4 text-green-600" />;
    if (score >= 70) return <AlertCircle className="w-4 h-4 text-yellow-600" />;
    return <XCircle className="w-4 h-4 text-red-600" />;
  };

  const getBestVehicle = (metric: string) => {
    switch (metric) {
      case 'price':
        return vehicles.reduce((best, current) =>
          current.price < best.price ? current : best
        );
      case 'fuel_efficiency':
        return vehicles.reduce((best, current) =>
          current.fuel_efficiency > best.fuel_efficiency ? current : best
        );
      case 'safety_rating':
        return vehicles.reduce((best, current) =>
          current.safety_rating > best.safety_rating ? current : best
        );
      case 'match_score':
        return vehicles.reduce((best, current) =>
          current.match_score > best.match_score ? current : best
        );
      default:
        return vehicles[0];
    }
  };

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent mb-4">
          🎯 최종 3차량 상세 비교
        </h1>
        <p className="text-gray-600 text-lg">
          AI가 학습한 당신의 선호도를 바탕으로 선별된 최적의 차량들입니다
        </p>
      </div>

      <div className="flex justify-center">
        <div className="bg-white/80 backdrop-blur-sm rounded-xl p-2 border border-purple-200/50">
          {[
            { id: 'overview', label: '종합 비교', icon: BarChart3 },
            { id: 'detailed', label: '상세 분석', icon: Target },
            { id: 'financial', label: '경제성 분석', icon: DollarSign }
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setSelectedTab(tab.id as 'overview' | 'detailed' | 'financial')}
              className={`px-6 py-3 rounded-lg font-medium transition-all duration-200 flex items-center gap-2 ${
                selectedTab === tab.id
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg'
                  : 'text-gray-600 hover:text-purple-600 hover:bg-purple-50'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {selectedTab === 'overview' && (
        <div className="space-y-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {vehicles.map((vehicle, index) => (
              <Card
                key={vehicle.id}
                className={`bg-white/90 backdrop-blur-sm border-2 transition-all duration-300 ${
                  index === 0
                    ? 'border-purple-500 shadow-lg shadow-purple-500/25 ring-2 ring-purple-500/20'
                    : 'border-purple-200/50'
                }`}
              >
                {index === 0 && (
                  <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white text-center py-2 rounded-t-lg text-sm font-medium">
                    🏆 AI 추천 1위
                  </div>
                )}

                <CardHeader className="pb-4">
                  <div className="relative h-40 bg-gradient-to-br from-purple-100 to-pink-100 rounded-lg overflow-hidden mb-4">
                    {vehicle.images && vehicle.images.length > 0 ? (
                      <img
                        src={vehicle.images[0]}
                        alt={`${vehicle.brand} ${vehicle.model}`}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center">
                        <Car className="w-12 h-12 text-purple-400" />
                      </div>
                    )}
                    <div className="absolute top-2 right-2">
                      <div className="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-2 py-1 rounded-full text-xs font-bold">
                        {vehicle.match_score}%
                      </div>
                    </div>
                  </div>

                  <CardTitle className="text-xl font-bold text-gray-900">
                    {vehicle.brand} {vehicle.model}
                  </CardTitle>
                  <div className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
                    {vehicle.price.toLocaleString()}만원
                  </div>
                </CardHeader>

                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-3 text-sm">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-gray-400" />
                      <span>{vehicle.year}년</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Fuel className="w-4 h-4 text-gray-400" />
                      <span>{vehicle.fuel_efficiency}km/L</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Settings className="w-4 h-4 text-gray-400" />
                      <span>{vehicle.mileage.toLocaleString()}km</span>
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
                      </div>
                    </div>
                  </div>

                  {vehicle.detailed_analysis && (
                    <div className="space-y-3">
                      <div className="text-sm font-medium text-gray-700">종합 점수</div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div className="bg-green-50 p-2 rounded-lg text-center">
                          <div className="text-green-600 font-bold">{vehicle.detailed_analysis.reliability_score}%</div>
                          <div className="text-green-600">신뢰성</div>
                        </div>
                        <div className="bg-blue-50 p-2 rounded-lg text-center">
                          <div className="text-blue-600 font-bold">{vehicle.detailed_analysis.resale_value}%</div>
                          <div className="text-blue-600">잔존가치</div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-1 gap-2">
                    <Button
                      onClick={() => onSelectVehicle(vehicle.id)}
                      className={`w-full ${
                        index === 0
                          ? 'bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600'
                          : 'bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800'
                      } text-white`}
                    >
                      <Target className="w-4 h-4 mr-2" />
                      {index === 0 ? '최적 선택' : '선택하기'}
                    </Button>

                    <Button
                      variant="outline"
                      onClick={() => onRequestFinancing(vehicle.id)}
                      className="w-full border-purple-200 text-purple-600 hover:bg-purple-50"
                    >
                      <CreditCard className="w-4 h-4 mr-2" />
                      금융상품 보기
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card className="bg-white/90 backdrop-blur-sm border border-purple-200/50">
            <CardHeader>
              <CardTitle className="text-center text-xl text-gray-900">
                주요 항목 비교
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-purple-200">
                      <th className="text-left py-3 px-4 font-medium text-gray-700">항목</th>
                      {vehicles.map((vehicle) => (
                        <th key={vehicle.id} className="text-center py-3 px-4 font-medium text-gray-700">
                          {vehicle.brand} {vehicle.model}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-purple-100">
                    {[
                      { key: 'price', label: '가격', unit: '만원', best: 'low' },
                      { key: 'fuel_efficiency', label: '연비', unit: 'km/L', best: 'high' },
                      { key: 'safety_rating', label: '안전등급', unit: '점', best: 'high' },
                      { key: 'match_score', label: '매칭점수', unit: '%', best: 'high' }
                    ].map((item) => {
                      const bestVehicle = getBestVehicle(item.key);
                      return (
                        <tr key={item.key}>
                          <td className="py-3 px-4 font-medium text-gray-700">{item.label}</td>
                          {vehicles.map((vehicle) => (
                            <td key={vehicle.id} className="text-center py-3 px-4">
                              <div className={`font-medium ${
                                vehicle.id === bestVehicle.id
                                  ? 'text-purple-600 font-bold'
                                  : 'text-gray-600'
                              }`}>
                                {vehicle[item.key as keyof Vehicle]?.toLocaleString?.()} {item.unit}
                                {vehicle.id === bestVehicle.id && (
                                  <div className="text-xs text-purple-500 mt-1">최고</div>
                                )}
                              </div>
                            </td>
                          ))}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {selectedTab === 'detailed' && (
        <div className="space-y-6">
          {vehicles.map((vehicle, index) => (
            <Card key={vehicle.id} className="bg-white/90 backdrop-blur-sm border border-purple-200/50">
              <CardHeader>
                <CardTitle className="flex items-center gap-3 text-xl">
                  {index === 0 && <span className="text-lg">🏆</span>}
                  {vehicle.brand} {vehicle.model} 상세 분석
                </CardTitle>
              </CardHeader>
              <CardContent>
                {vehicle.detailed_analysis && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    <div className="space-y-6">
                      <div>
                        <h4 className="text-lg font-semibold text-green-700 mb-3 flex items-center gap-2">
                          <CheckCircle className="w-5 h-5" />
                          장점
                        </h4>
                        <ul className="space-y-2">
                          {vehicle.detailed_analysis.pros.map((pro, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-green-600">
                              <div className="w-1.5 h-1.5 rounded-full bg-green-500 mt-2 flex-shrink-0"></div>
                              {pro}
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div>
                        <h4 className="text-lg font-semibold text-orange-700 mb-3 flex items-center gap-2">
                          <AlertCircle className="w-5 h-5" />
                          주의사항
                        </h4>
                        <ul className="space-y-2">
                          {vehicle.detailed_analysis.cons.map((con, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-orange-600">
                              <div className="w-1.5 h-1.5 rounded-full bg-orange-500 mt-2 flex-shrink-0"></div>
                              {con}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>

                    <div className="space-y-4">
                      <h4 className="text-lg font-semibold text-gray-800">세부 평가</h4>
                      <div className="grid grid-cols-2 gap-4">
                        {[
                          { label: '신뢰성', value: vehicle.detailed_analysis.reliability_score, icon: Shield },
                          { label: '잔존가치', value: vehicle.detailed_analysis.resale_value, icon: TrendingUp },
                          { label: '유지비', value: 100 - (vehicle.detailed_analysis.maintenance_cost / 2), icon: Settings },
                          { label: '보험료', value: 100 - (vehicle.detailed_analysis.insurance_cost / 2), icon: Shield }
                        ].map((metric) => (
                          <div key={metric.label} className="bg-gray-50 p-4 rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-sm text-gray-600 flex items-center gap-1">
                                <metric.icon className="w-4 h-4" />
                                {metric.label}
                              </span>
                              <span className={`text-lg font-bold ${getScoreColor(metric.value)}`}>
                                {metric.value}%
                              </span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className={`h-2 rounded-full ${
                                  metric.value >= 85 ? 'bg-green-500' :
                                  metric.value >= 70 ? 'bg-yellow-500' : 'bg-red-500'
                                }`}
                                style={{ width: `${metric.value}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>

                      <div className="bg-purple-50 p-4 rounded-lg">
                        <h5 className="font-semibold text-purple-800 mb-2">시장 포지셔닝</h5>
                        <p className="text-purple-700 text-sm">
                          이 차량은 <span className="font-bold">{vehicle.detailed_analysis.market_position}</span> 등급으로 분류됩니다.
                        </p>
                        <div className="mt-3 flex items-center gap-2 text-sm">
                          <TrendingDown className="w-4 h-4 text-purple-600" />
                          <span className="text-purple-600">
                            예상 감가율: <span className="font-bold">{vehicle.detailed_analysis.depreciation_rate}%/년</span>
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {selectedTab === 'financial' && (
        <div className="space-y-6">
          <Card className="bg-white/90 backdrop-blur-sm border border-purple-200/50">
            <CardHeader>
              <CardTitle className="text-xl text-gray-900 flex items-center gap-2">
                <DollarSign className="w-6 h-6 text-purple-600" />
                3년간 총 보유비용 비교
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-purple-200">
                      <th className="text-left py-3 px-4 font-medium text-gray-700">비용 항목</th>
                      {vehicles.map((vehicle) => (
                        <th key={vehicle.id} className="text-center py-3 px-4 font-medium text-gray-700">
                          {vehicle.brand} {vehicle.model}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-purple-100">
                    {vehicles[0]?.detailed_analysis && [
                      { key: 'purchase', label: '구매가격', getValue: (v: Vehicle) => v.price },
                      { key: 'maintenance', label: '유지비(3년)', getValue: (v: Vehicle) => v.detailed_analysis?.maintenance_cost! * 36 },
                      { key: 'insurance', label: '보험료(3년)', getValue: (v: Vehicle) => v.detailed_analysis?.insurance_cost! * 36 },
                      { key: 'depreciation', label: '감가비용(3년)', getValue: (v: Vehicle) => Math.round(v.price * (v.detailed_analysis?.depreciation_rate! / 100) * 3) },
                      { key: 'total', label: '총 비용', getValue: (v: Vehicle) => {
                        const maintenance = v.detailed_analysis?.maintenance_cost! * 36;
                        const insurance = v.detailed_analysis?.insurance_cost! * 36;
                        const depreciation = Math.round(v.price * (v.detailed_analysis?.depreciation_rate! / 100) * 3);
                        return v.price + maintenance + insurance + depreciation;
                      }}
                    ].map((item) => {
                      const costs = vehicles.map(v => item.getValue(v));
                      const minCost = Math.min(...costs);

                      return (
                        <tr key={item.key} className={item.key === 'total' ? 'border-t-2 border-purple-300 font-bold' : ''}>
                          <td className="py-3 px-4 font-medium text-gray-700">{item.label}</td>
                          {vehicles.map((vehicle) => {
                            const cost = item.getValue(vehicle);
                            return (
                              <td key={vehicle.id} className="text-center py-3 px-4">
                                <div className={`${
                                  cost === minCost && item.key === 'total'
                                    ? 'text-green-600 font-bold'
                                    : 'text-gray-600'
                                }`}>
                                  {cost.toLocaleString()}만원
                                  {cost === minCost && item.key === 'total' && (
                                    <div className="text-xs text-green-500 mt-1">최저비용</div>
                                  )}
                                </div>
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-gradient-to-r from-purple-50 to-pink-50 border border-purple-200/50">
            <CardHeader>
              <CardTitle className="text-lg text-purple-800">💡 AI 금융 분석가 추천</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {vehicles.map((vehicle, index) => (
                    <div key={vehicle.id} className="bg-white/70 p-4 rounded-lg">
                      <h4 className="font-semibold text-gray-800 mb-2">
                        {vehicle.brand} {vehicle.model}
                      </h4>
                      <div className="text-sm space-y-2">
                        <div className="flex justify-between">
                          <span className="text-gray-600">경제성 등급:</span>
                          <span className={`font-medium ${
                            index === 0 ? 'text-green-600' :
                            index === 1 ? 'text-yellow-600' : 'text-orange-600'
                          }`}>
                            {index === 0 ? '우수' : index === 1 ? '양호' : '보통'}
                          </span>
                        </div>
                        <div className="text-xs text-gray-500">
                          {index === 0 && "가장 경제적인 선택입니다"}
                          {index === 1 && "균형잡힌 가성비를 제공합니다"}
                          {index === 2 && "프리미엄 옵션으로 고려해보세요"}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}