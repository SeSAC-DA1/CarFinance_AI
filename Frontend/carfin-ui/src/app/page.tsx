'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import {
  Car
} from 'lucide-react';
import { ModernSignupForm } from '@/components/auth/ModernSignupForm';
import { ModernLandingPage } from '@/components/landing/ModernLandingPage';
import { CoreThreeAgentChat } from '@/components/chat/CoreThreeAgentChat';
import { ModernVehicleGrid } from '@/components/vehicle/ModernVehicleGrid';
import { EnhancedAnalysisDashboard } from '@/components/analysis/EnhancedAnalysisDashboard';
import { FinanceConsultation } from '@/components/finance/FinanceConsultation';
import { ApiHealthCheck } from '@/components/debug/ApiHealthCheck';
import {
  UIUserProfile,
  AppPhase,
  Vehicle,
  VehicleFeedback
} from '@/types';

export default function CarFinPage() {
  const [userProfile, setUserProfile] = useState<UIUserProfile | null>(null);
  const [currentPhase, setCurrentPhase] = useState<AppPhase>('landing');
  const [collectedData, setCollectedData] = useState<Record<string, unknown>>({});
  const [selectedVehicle, setSelectedVehicle] = useState<Vehicle | null>(null);
  const [userFeedback, setUserFeedback] = useState<VehicleFeedback[]>([]);
  useEffect(() => {
    // Component initialization
  }, []);

  const handleSignupComplete = (profile: UIUserProfile) => {
    setUserProfile(profile);
    setCurrentPhase('grid'); // 직접 그리드로 이동
  };

  const handleSkip = () => {
    setUserProfile({
      user_id: `guest_${Date.now()}`,
      name: 'Guest',
      guest: true
    });
    setCurrentPhase('grid'); // 직접 그리드로 이동
  };

  const handleChatComplete = (data: Record<string, unknown>) => {
    setCollectedData(data);
    setCurrentPhase('grid');
  };

  const handleVehicleSelection = (vehicles: Vehicle[], feedback: VehicleFeedback[]) => {
    // 첫 번째 차량을 선택된 차량으로 설정
    setSelectedVehicle(vehicles[0] || null);
    setUserFeedback(feedback);
    setCurrentPhase('analysis');
  };

  const handleAnalysisComplete = () => {
    setCurrentPhase('finance');
  };

  const handleFinanceComplete = () => {
    // 완료 처리
    alert('축하합니다! 모든 절차가 완료되었습니다.');
    setCurrentPhase('landing');
  };

  const handleBackToVehicles = () => {
    setCurrentPhase('grid');
  };

  // 모던 랜딩 페이지
  if (currentPhase === 'landing') {
    return (
      <div>
        <ModernLandingPage onGetStarted={() => setCurrentPhase('signup')} />
        {/* 개발자용 API 상태 체크 */}
        <div className="fixed bottom-4 right-4 z-50">
          <ApiHealthCheck />
        </div>
      </div>
    );
  }

  // 회원가입 페이지
  if (currentPhase === 'signup') {
    return <ModernSignupForm onSignupComplete={handleSignupComplete} onSkip={handleSkip} />;
  }

  // 채팅 단계
  if (currentPhase === 'chat') {
    return (
      <div className="min-h-screen bg-gray-50">
        <div className="max-w-4xl mx-auto px-6 py-8">
          <div className="text-center mb-8">
            <h1 className="text-2xl font-bold text-gray-900 mb-2">
              AI 상담사와 대화하기
            </h1>
            <p className="text-gray-600">
              {userProfile?.name}님의 이상적인 중고차를 찾기 위해 몇 가지 질문드릴게요
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-sm border border-gray-200 p-6">
            <CoreThreeAgentChat
              userProfile={userProfile}
              onRecommendationComplete={handleChatComplete}
            />
          </div>
        </div>
      </div>
    );
  }

  // 차량 선택 단계
  if (currentPhase === 'grid') {
    return (
      <ModernVehicleGrid
        userProfile={{ ...userProfile, ...collectedData } || {}}
        onSelectionComplete={handleVehicleSelection}
      />
    );
  }

  // 분석 대시보드 단계
  if (currentPhase === 'analysis') {
    return (
      <EnhancedAnalysisDashboard
        selectedVehicles={selectedVehicle ? [selectedVehicle] : []}
        userFeedback={userFeedback}
        onProceedToFinance={handleAnalysisComplete}
        onSelectDifferentVehicle={handleBackToVehicles}
      />
    );
  }

  // 금융 상담 단계
  if (currentPhase === 'finance') {
    return (
      <FinanceConsultation
        selectedVehicle={selectedVehicle}
        userProfile={userProfile}
        onConsultationComplete={handleFinanceComplete}
      />
    );
  }

  // 기본 예외 처리
  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center max-w-md mx-auto px-6">
        <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
          <Car
            className="w-8 h-8 text-blue-600"
            aria-hidden="true"
            role="img"
          />
        </div>
        <h2 className="text-xl font-bold text-gray-900 mb-4">
          예상치 못한 오류가 발생했습니다
        </h2>
        <p className="text-gray-600 mb-6">
          메인 페이지로 돌아가서 다시 시도해주세요
        </p>
        <Button
          onClick={() => setCurrentPhase('landing')}
          className="bg-blue-600 hover:bg-blue-700 text-white"
          aria-label="메인 페이지로 돌아가기"
        >
          메인으로 돌아가기
        </Button>
      </div>
    </div>
  );
}