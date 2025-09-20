'use client';

import { useState } from 'react';
import { Container } from '@/components/design-system/layout/Container';
import { Button } from '@/components/ui/button';
import {
  Car,
  ArrowRight,
  User,
  Calendar,
  DollarSign,
  Heart,
  CheckCircle,
  ArrowLeft,
  Sparkles
} from 'lucide-react';

interface UserProfile {
  user_id: string;
  name?: string;
  email?: string;
  age?: number;
  income?: number;
  preferences?: string[];
  purpose?: string;
  guest?: boolean;
}

interface ModernSignupFormProps {
  onSignupComplete: (profile: UserProfile) => void;
  onSkip: () => void;
}

type Step = 1 | 2 | 3 | 4;

const ageRanges = [
  { label: '20대', value: 25, emoji: '🌱' },
  { label: '30대', value: 35, emoji: '🚀' },
  { label: '40대', value: 45, emoji: '💼' },
  { label: '50대+', value: 55, emoji: '🏆' }
];

const incomeRanges = [
  { label: '3천만원 미만', value: 2500, emoji: '🌟' },
  { label: '3-5천만원', value: 4000, emoji: '💎' },
  { label: '5-7천만원', value: 6000, emoji: '🔥' },
  { label: '7천만원 이상', value: 8000, emoji: '👑' }
];

const purposes = [
  { label: '출퇴근용', value: 'commute', emoji: '🚗', desc: '안정적이고 연비 좋은 차량' },
  { label: '가족용', value: 'family', emoji: '👨‍👩‍👧‍👦', desc: '넓고 안전한 공간' },
  { label: '레저용', value: 'leisure', emoji: '🏔️', desc: '주말 여행과 아웃도어' },
  { label: '비즈니스용', value: 'business', emoji: '💼', desc: '품격 있는 이미지' }
];

const preferences = [
  { label: '연비', value: 'fuel_efficiency', emoji: '⛽' },
  { label: '안전성', value: 'safety', emoji: '🛡️' },
  { label: '디자인', value: 'design', emoji: '✨' },
  { label: '성능', value: 'performance', emoji: '🏎️' },
  { label: '브랜드', value: 'brand', emoji: '🏷️' },
  { label: '가격', value: 'price', emoji: '💰' }
];

export function ModernSignupForm({ onSignupComplete, onSkip }: ModernSignupFormProps) {
  const [currentStep, setCurrentStep] = useState<Step>(1);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    age: 0,
    income: 0,
    purpose: '',
    preferences: [] as string[]
  });

  const handleNext = () => {
    if (currentStep < 4) {
      setCurrentStep((prev) => (prev + 1) as Step);
    } else {
      // 폼 완료
      handleComplete();
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep((prev) => (prev - 1) as Step);
    }
  };

  const handleComplete = () => {
    const profile: UserProfile = {
      user_id: `user_${Date.now()}`,
      name: formData.name || '사용자',
      email: formData.email,
      age: formData.age,
      income: formData.income,
      purpose: formData.purpose,
      preferences: formData.preferences
    };
    onSignupComplete(profile);
  };

  const togglePreference = (pref: string) => {
    setFormData(prev => ({
      ...prev,
      preferences: prev.preferences.includes(pref)
        ? prev.preferences.filter(p => p !== pref)
        : [...prev.preferences, pref]
    }));
  };

  const isStepValid = () => {
    switch (currentStep) {
      case 1: return formData.name.length >= 2;
      case 2: return formData.age > 0;
      case 3: return formData.income > 0;
      case 4: return formData.purpose.length > 0;
      default: return false;
    }
  };

  const getStepProgress = () => (currentStep / 4) * 100;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50">
      {/* 헤더 */}
      <div className="bg-white/80 backdrop-blur-sm border-b border-blue-100 sticky top-0 z-10">
        <Container>
          <div className="py-6">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl flex items-center justify-center">
                  <Car className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900">CarFin AI</h1>
                  <p className="text-sm text-gray-600">AI 중고차 매칭 서비스</p>
                </div>
              </div>

              <Button
                variant="ghost"
                size="sm"
                onClick={onSkip}
                aria-label="회원가입 건너뛰고 게스트로 시작하기"
              >
                건너뛰기
              </Button>
            </div>

            {/* 진행률 바 */}
            <div className="mt-6">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-gray-700">
                  {currentStep}/4 단계
                </span>
                <span className="text-sm text-gray-500">
                  {Math.round(getStepProgress())}% 완료
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-gradient-to-r from-blue-600 to-blue-700 h-2 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${getStepProgress()}%` }}
                />
              </div>
            </div>
          </div>
        </Container>
      </div>

      {/* 메인 콘텐츠 */}
      <Container size="md">
        <div className="py-12">
          <div className="max-w-lg mx-auto">

            {/* Step 1: 기본 정보 */}
            {currentStep === 1 && (
              <div className="text-center space-y-8">
                <div className="space-y-4">
                  <div className="w-16 h-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl flex items-center justify-center mx-auto">
                    <User className="w-8 h-8 text-white" />
                  </div>
                  <h2 className="text-3xl font-bold text-gray-900">
                    안녕하세요! 👋
                  </h2>
                  <p className="text-lg text-gray-600">
                    먼저 어떻게 불러드릴까요?
                  </p>
                </div>

                <div className="space-y-6">
                  <div>
                    <input
                      type="text"
                      placeholder="이름을 입력해주세요"
                      value={formData.name}
                      onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                      className="w-full px-6 py-4 text-lg border border-gray-200 rounded-2xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200"
                      autoFocus
                    />
                  </div>

                  <div>
                    <input
                      type="email"
                      placeholder="이메일 (선택사항)"
                      value={formData.email}
                      onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                      className="w-full px-6 py-4 text-lg border border-gray-200 rounded-2xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all duration-200"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Step 2: 연령대 */}
            {currentStep === 2 && (
              <div className="text-center space-y-8">
                <div className="space-y-4">
                  <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-blue-600 rounded-2xl flex items-center justify-center mx-auto">
                    <Calendar className="w-8 h-8 text-white" />
                  </div>
                  <h2 className="text-3xl font-bold text-gray-900">
                    {formData.name}님의 연령대는?
                  </h2>
                  <p className="text-lg text-gray-600">
                    연령대별 맞춤 추천을 위해 필요해요
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  {ageRanges.map((range) => (
                    <button
                      key={range.value}
                      onClick={() => setFormData(prev => ({ ...prev, age: range.value }))}
                      className={`p-6 rounded-2xl border-2 transition-all duration-200 ${
                        formData.age === range.value
                          ? 'border-blue-500 bg-blue-50 shadow-lg shadow-blue-100'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      <div className="text-3xl mb-2">{range.emoji}</div>
                      <div className="font-semibold text-gray-900">{range.label}</div>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 3: 소득 구간 */}
            {currentStep === 3 && (
              <div className="text-center space-y-8">
                <div className="space-y-4">
                  <div className="w-16 h-16 bg-gradient-to-r from-yellow-500 to-orange-600 rounded-2xl flex items-center justify-center mx-auto">
                    <DollarSign className="w-8 h-8 text-white" />
                  </div>
                  <h2 className="text-3xl font-bold text-gray-900">
                    소득 구간을 선택해주세요
                  </h2>
                  <p className="text-lg text-gray-600">
                    적정 가격대 차량을 추천해드릴게요
                  </p>
                </div>

                <div className="space-y-3">
                  {incomeRanges.map((range) => (
                    <button
                      key={range.value}
                      onClick={() => setFormData(prev => ({ ...prev, income: range.value }))}
                      className={`w-full p-5 rounded-2xl border-2 transition-all duration-200 flex items-center gap-4 ${
                        formData.income === range.value
                          ? 'border-blue-500 bg-blue-50 shadow-lg shadow-blue-100'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      <span className="text-2xl">{range.emoji}</span>
                      <div className="text-left">
                        <div className="font-semibold text-gray-900">{range.label}</div>
                      </div>
                      {formData.income === range.value && (
                        <CheckCircle className="w-6 h-6 text-blue-600 ml-auto" />
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Step 4: 사용 목적 */}
            {currentStep === 4 && (
              <div className="text-center space-y-8">
                <div className="space-y-4">
                  <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-600 rounded-2xl flex items-center justify-center mx-auto">
                    <Heart className="w-8 h-8 text-white" />
                  </div>
                  <h2 className="text-3xl font-bold text-gray-900">
                    주로 어떤 용도로 사용하시나요?
                  </h2>
                  <p className="text-lg text-gray-600">
                    용도에 맞는 최적의 차량을 찾아드릴게요
                  </p>
                </div>

                <div className="space-y-3">
                  {purposes.map((purpose) => (
                    <button
                      key={purpose.value}
                      onClick={() => setFormData(prev => ({ ...prev, purpose: purpose.value }))}
                      className={`w-full p-6 rounded-2xl border-2 transition-all duration-200 ${
                        formData.purpose === purpose.value
                          ? 'border-blue-500 bg-blue-50 shadow-lg shadow-blue-100'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                      }`}
                    >
                      <div className="flex items-center gap-4">
                        <span className="text-3xl">{purpose.emoji}</span>
                        <div className="text-left flex-1">
                          <div className="font-semibold text-lg text-gray-900">{purpose.label}</div>
                          <div className="text-sm text-gray-600">{purpose.desc}</div>
                        </div>
                        {formData.purpose === purpose.value && (
                          <CheckCircle className="w-6 h-6 text-blue-600" />
                        )}
                      </div>
                    </button>
                  ))}
                </div>

                {/* 선호사항 */}
                {formData.purpose && (
                  <div className="mt-8 pt-8 border-t border-gray-200">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">
                      중요하게 생각하는 요소는? (복수 선택 가능)
                    </h3>
                    <div className="grid grid-cols-3 gap-3">
                      {preferences.map((pref) => (
                        <button
                          key={pref.value}
                          onClick={() => togglePreference(pref.value)}
                          className={`p-4 rounded-xl border-2 transition-all duration-200 ${
                            formData.preferences.includes(pref.value)
                              ? 'border-blue-500 bg-blue-50'
                              : 'border-gray-200 hover:border-gray-300'
                          }`}
                        >
                          <div className="text-2xl mb-2">{pref.emoji}</div>
                          <div className="text-sm font-medium text-gray-900">{pref.label}</div>
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 네비게이션 버튼 */}
            <div className="flex items-center gap-4 mt-12">
              {currentStep > 1 && (
                <Button
                  variant="outline"
                  size="lg"
                  onClick={handleBack}
                  icon={<ArrowLeft className="w-5 h-5" aria-hidden="true" />}
                  aria-label="이전 단계로 돌아가기"
                >
                  이전
                </Button>
              )}

              <Button
                variant="default"
                size="lg"
                className="flex-1"
                onClick={handleNext}
                disabled={!isStepValid()}
                icon={currentStep === 4 ? <Sparkles className="w-5 h-5" aria-hidden="true" /> : <ArrowRight className="w-5 h-5" aria-hidden="true" />}
                iconPosition="right"
                aria-label={currentStep === 4 ? 'AI 상담 시작하기' : '다음 단계로 진행하기'}
              >
                {currentStep === 4 ? 'AI 상담 시작하기' : '다음'}
              </Button>
            </div>
          </div>
        </div>
      </Container>
    </div>
  );
}