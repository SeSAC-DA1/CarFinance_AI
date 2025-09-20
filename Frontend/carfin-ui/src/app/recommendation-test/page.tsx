'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { RecommendationVehicleGrid } from '@/components/recommendation/RecommendationVehicleGrid';
import { Badge } from '@/components/ui/badge';
import {
  User,
  Settings,
  RefreshCw,
  BarChart3,
  Target,
  Users,
  TrendingUp,
  GitCompare,
  ArrowLeft
} from 'lucide-react';

// 테스트용 사용자 프로필들
const TEST_USER_PROFILES = [
  {
    user_id: 'test_user_1',
    name: '김철수',
    email: 'kim@example.com',
    age: 28,
    income: 4500,
    preferences: ['연비', '안전성', 'BMW'],
    purpose: 'commute'
  },
  {
    user_id: 'test_user_2',
    name: '이영희',
    email: 'lee@example.com',
    age: 35,
    income: 6000,
    preferences: ['가족용', '넓은 공간', '현대'],
    purpose: 'family'
  },
  {
    user_id: 'test_user_3',
    name: '박민수',
    email: 'park@example.com',
    age: 45,
    income: 8000,
    preferences: ['럭셔리', '브랜드', '벤츠'],
    purpose: 'business'
  },
  {
    user_id: 'test_user_4',
    name: '정수정',
    email: 'jung@example.com',
    age: 24,
    income: 3000,
    preferences: ['가격', '경차', '연비'],
    purpose: 'commute'
  }
];

const RECOMMENDATION_TYPES = [
  { value: 'personalized', label: '개인화 추천', icon: Target, description: 'User-Based CF + Content-Based' },
  { value: 'homepage', label: '홈페이지 추천', icon: TrendingUp, description: '인기도 + 개인화 혼합' },
  { value: 'similar', label: '유사 차량', icon: GitCompare, description: 'Item-Based CF' }
] as const;

export default function RecommendationTestPage() {
  const [selectedUser, setSelectedUser] = useState(TEST_USER_PROFILES[0]);
  const [recommendationType, setRecommendationType] = useState<'personalized' | 'homepage' | 'similar'>('personalized');
  const [selectedVehicles, setSelectedVehicles] = useState<string[]>([]);
  const [currentVehicleId, setCurrentVehicleId] = useState<string>('car_1'); // 유사 차량 테스트용
  const [refreshKey, setRefreshKey] = useState(0);

  const handleUserChange = (user: typeof TEST_USER_PROFILES[0]) => {
    setSelectedUser(user);
    setSelectedVehicles([]);
    setRefreshKey(prev => prev + 1); // 강제 새로고침
  };

  const handleRecommendationTypeChange = (type: typeof recommendationType) => {
    setRecommendationType(type);
    setRefreshKey(prev => prev + 1);
  };

  const handleVehicleSelect = (vehicleId: string) => {
    setSelectedVehicles(prev => {
      if (prev.includes(vehicleId)) {
        return prev.filter(id => id !== vehicleId);
      } else if (prev.length < 3) {
        return [...prev, vehicleId];
      }
      return prev;
    });
  };

  const handleFinanceRequest = (vehicleId: string) => {
    alert(`${vehicleId} 차량에 대한 금융 상담을 요청했습니다!`);
  };

  const resetTest = () => {
    setSelectedVehicles([]);
    setRefreshKey(prev => prev + 1);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      {/* 헤더 */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-blue-100 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => window.history.back()}
                className="flex items-center gap-2"
              >
                <ArrowLeft className="w-4 h-4" />
                뒤로
              </Button>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                  🧪 추천 시스템 테스트
                </h1>
                <p className="text-sm text-gray-600">
                  협업 필터링 기반 차량 추천 시스템 실시간 테스트
                </p>
              </div>
            </div>

            <Button
              onClick={resetTest}
              variant="outline"
              size="sm"
              className="flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              초기화
            </Button>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* 사이드바 - 테스트 설정 */}
          <div className="lg:col-span-1 space-y-6">
            {/* 사용자 선택 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <User className="w-5 h-5" />
                  테스트 사용자
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {TEST_USER_PROFILES.map((user) => (
                  <button
                    key={user.user_id}
                    onClick={() => handleUserChange(user)}
                    className={`w-full p-3 rounded-lg border-2 transition-all text-left ${
                      selectedUser.user_id === user.user_id
                        ? 'border-purple-500 bg-purple-50'
                        : 'border-gray-200 hover:border-gray-300'
                    }`}
                  >
                    <div className="font-medium text-gray-900">{user.name}</div>
                    <div className="text-sm text-gray-600">{user.age}세 • {user.income}만원</div>
                    <div className="text-xs text-gray-500 mt-1">
                      {user.purpose} • {user.preferences.slice(0, 2).join(', ')}
                    </div>
                  </button>
                ))}
              </CardContent>
            </Card>

            {/* 추천 타입 선택 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <Settings className="w-5 h-5" />
                  추천 타입
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {RECOMMENDATION_TYPES.map((type) => {
                  const IconComponent = type.icon;
                  return (
                    <button
                      key={type.value}
                      onClick={() => handleRecommendationTypeChange(type.value)}
                      className={`w-full p-3 rounded-lg border-2 transition-all text-left ${
                        recommendationType === type.value
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <IconComponent className="w-4 h-4" />
                        <span className="font-medium text-gray-900">{type.label}</span>
                      </div>
                      <div className="text-xs text-gray-500">{type.description}</div>
                    </button>
                  );
                })}
              </CardContent>
            </Card>

            {/* 현재 설정 정보 */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                  <BarChart3 className="w-5 h-5" />
                  현재 설정
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <div className="text-sm font-medium text-gray-700 mb-1">선택된 사용자</div>
                  <Badge variant="outline">{selectedUser.name}</Badge>
                </div>

                <div>
                  <div className="text-sm font-medium text-gray-700 mb-1">추천 방식</div>
                  <Badge variant="outline">
                    {RECOMMENDATION_TYPES.find(t => t.value === recommendationType)?.label}
                  </Badge>
                </div>

                <div>
                  <div className="text-sm font-medium text-gray-700 mb-1">선택된 차량</div>
                  <div className="text-sm text-gray-600">
                    {selectedVehicles.length}/3 선택됨
                  </div>
                  {selectedVehicles.length > 0 && (
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedVehicles.map(id => (
                        <Badge key={id} variant="secondary" className="text-xs">
                          {id}
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>

                {recommendationType === 'similar' && (
                  <div>
                    <div className="text-sm font-medium text-gray-700 mb-1">기준 차량</div>
                    <select
                      value={currentVehicleId}
                      onChange={(e) => setCurrentVehicleId(e.target.value)}
                      className="w-full p-2 border border-gray-200 rounded text-sm"
                    >
                      {Array.from({ length: 20 }, (_, i) => (
                        <option key={`car_${i + 1}`} value={`car_${i + 1}`}>
                          car_{i + 1}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* 메인 콘텐츠 - 추천 결과 */}
          <div className="lg:col-span-3">
            <RecommendationVehicleGrid
              key={refreshKey} // 강제 새로고침용
              userProfile={selectedUser}
              onVehicleSelect={handleVehicleSelect}
              onRequestFinancing={handleFinanceRequest}
              selectedVehicles={selectedVehicles}
              recommendationType={recommendationType}
              currentVehicleId={recommendationType === 'similar' ? currentVehicleId : undefined}
              limit={9}
            />
          </div>
        </div>
      </div>

      {/* 푸터 정보 */}
      <div className="bg-white/50 backdrop-blur-sm border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-sm text-gray-600">
            <div>
              <h3 className="font-medium text-gray-900 mb-2">📚 테스트 기능</h3>
              <ul className="space-y-1">
                <li>• User-Based Collaborative Filtering</li>
                <li>• Item-Based Collaborative Filtering</li>
                <li>• Content-Based Filtering</li>
                <li>• Hybrid Recommendation Engine</li>
              </ul>
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-2">🎯 추천 알고리즘</h3>
              <ul className="space-y-1">
                <li>• Pearson 상관계수 (사용자 유사도)</li>
                <li>• 코사인 유사도 (차량 유사도)</li>
                <li>• 암시적 피드백 학습</li>
                <li>• 실시간 행동 추적</li>
              </ul>
            </div>
            <div>
              <h3 className="font-medium text-gray-900 mb-2">📊 데이터</h3>
              <ul className="space-y-1">
                <li>• 150개 더미 차량</li>
                <li>• 30명 가상 사용자</li>
                <li>• 450개 상호작용 데이터</li>
                <li>• 512차원 특성 벡터</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}