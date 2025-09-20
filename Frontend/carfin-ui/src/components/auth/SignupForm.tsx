'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { UserPlus, Sparkles, SkipForward, Mail, User, Calendar, DollarSign, Heart } from 'lucide-react';

interface UserProfile {
  user_id: string;
  name: string;
  email: string;
  age: number;
  income: number;
  preferences: string[];
  purpose: string;
  created_at: string;
}

interface SignupFormProps {
  onSignupComplete: (userProfile: UserProfile) => void;
  onSkip: () => void;
}

export function SignupForm({ onSignupComplete, onSkip }: SignupFormProps) {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    age: '',
    income: '',
    preferences: [] as string[],
    purpose: ''
  });

  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);

    const userProfile = {
      user_id: `user_${Date.now()}`,
      name: formData.name,
      email: formData.email,
      age: parseInt(formData.age) || 30,
      income: parseInt(formData.income) || 300,
      preferences: formData.preferences,
      purpose: formData.purpose,
      created_at: new Date().toISOString()
    };

    try {
      await fetch('http://localhost:8000/api/users/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userProfile)
      }).catch(() => {
        console.log('Backend not available, proceeding with local profile');
      });

      onSignupComplete(userProfile);
    } catch (error) {
      console.error('Signup error:', error);
      onSignupComplete(userProfile);
    } finally {
      setIsSubmitting(false);
    }
  };

  const togglePreference = (pref: string) => {
    setFormData(prev => ({
      ...prev,
      preferences: prev.preferences.includes(pref)
        ? prev.preferences.filter(p => p !== pref)
        : [...prev.preferences, pref]
    }));
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-white via-purple-50 to-purple-100 relative overflow-hidden">
      <div className="absolute inset-0">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-purple-200/20 via-white to-purple-300/30"></div>
        <div className="absolute top-20 right-20 w-96 h-96 bg-gradient-to-br from-purple-400 to-pink-400 rounded-full blur-3xl opacity-20"></div>
        <div className="absolute bottom-20 left-20 w-80 h-80 bg-gradient-to-br from-indigo-400 to-purple-500 rounded-full blur-3xl opacity-15"></div>
        <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-64 h-64 bg-gradient-to-br from-white to-purple-200 rounded-full blur-2xl opacity-30"></div>
      </div>

      <div className="relative z-10 min-h-screen flex items-center justify-center p-4">
        <Card className="w-full max-w-2xl bg-white/80 backdrop-blur-xl border-purple-200/50 shadow-2xl shadow-purple-500/10">
          <CardHeader className="text-center space-y-4 pb-8">
            <div className="flex items-center justify-center mb-4">
              <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-4 rounded-2xl shadow-lg">
                <UserPlus className="w-8 h-8 text-white" />
              </div>
            </div>

            <CardTitle className="text-3xl font-bold bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
              개인화된 추천을 위한 프로필 생성
            </CardTitle>

            <p className="text-gray-600 text-lg">
              더 정확한 차량 추천을 위해 간단한 정보를 입력해주세요
            </p>
          </CardHeader>

          <CardContent className="space-y-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label className="text-gray-700 font-medium flex items-center gap-2">
                    <User className="w-4 h-4 text-purple-500" />
                    이름
                  </label>
                  <Input
                    type="text"
                    value={formData.name}
                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="홍길동"
                    className="bg-white/70 border-purple-200 focus:border-purple-400 focus:ring-purple-400/20"
                    required
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-gray-700 font-medium flex items-center gap-2">
                    <Mail className="w-4 h-4 text-purple-500" />
                    이메일
                  </label>
                  <Input
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                    placeholder="hong@example.com"
                    className="bg-white/70 border-purple-200 focus:border-purple-400 focus:ring-purple-400/20"
                    required
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-2">
                  <label className="text-gray-700 font-medium flex items-center gap-2">
                    <Calendar className="w-4 h-4 text-purple-500" />
                    연령대
                  </label>
                  <Select value={formData.age} onValueChange={(value) => setFormData(prev => ({ ...prev, age: value }))}>
                    <SelectTrigger className="bg-white/70 border-purple-200 focus:border-purple-400">
                      <SelectValue placeholder="선택해주세요" />
                    </SelectTrigger>
                    <SelectContent className="bg-white border-purple-200">
                      <SelectItem value="20-29">20대</SelectItem>
                      <SelectItem value="30-39">30대</SelectItem>
                      <SelectItem value="40-49">40대</SelectItem>
                      <SelectItem value="50-59">50대</SelectItem>
                      <SelectItem value="60+">60대 이상</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <label className="text-gray-700 font-medium flex items-center gap-2">
                    <DollarSign className="w-4 h-4 text-purple-500" />
                    월 소득 (만원)
                  </label>
                  <Select value={formData.income} onValueChange={(value) => setFormData(prev => ({ ...prev, income: value }))}>
                    <SelectTrigger className="bg-white/70 border-purple-200 focus:border-purple-400">
                      <SelectValue placeholder="선택해주세요" />
                    </SelectTrigger>
                    <SelectContent className="bg-white border-purple-200">
                      <SelectItem value="200">200만원 미만</SelectItem>
                      <SelectItem value="300">200-300만원</SelectItem>
                      <SelectItem value="400">300-400만원</SelectItem>
                      <SelectItem value="500">400-500만원</SelectItem>
                      <SelectItem value="600">500만원 이상</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="space-y-4">
                <label className="text-gray-700 font-medium flex items-center gap-2">
                  <Heart className="w-4 h-4 text-purple-500" />
                  중요하게 생각하는 요소 (복수 선택 가능)
                </label>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {[
                    { id: 'fuel_efficiency', label: '연비', emoji: '⛽' },
                    { id: 'safety', label: '안전성', emoji: '🛡️' },
                    { id: 'comfort', label: '편의성', emoji: '🛋️' },
                    { id: 'design', label: '디자인', emoji: '🎨' },
                    { id: 'performance', label: '성능', emoji: '⚡' },
                    { id: 'price', label: '가격', emoji: '💰' }
                  ].map((pref) => (
                    <button
                      key={pref.id}
                      type="button"
                      onClick={() => togglePreference(pref.id)}
                      className={`p-3 rounded-xl border-2 transition-all duration-200 text-sm font-medium ${
                        formData.preferences.includes(pref.id)
                          ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white border-purple-500 shadow-lg shadow-purple-500/25'
                          : 'bg-white/50 border-purple-200 text-gray-700 hover:border-purple-300 hover:bg-purple-50'
                      }`}
                    >
                      <div className="text-lg mb-1">{pref.emoji}</div>
                      {pref.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-2">
                <label className="text-gray-700 font-medium">구매 목적</label>
                <Select value={formData.purpose} onValueChange={(value) => setFormData(prev => ({ ...prev, purpose: value }))}>
                  <SelectTrigger className="bg-white/70 border-purple-200 focus:border-purple-400">
                    <SelectValue placeholder="주요 사용 목적을 선택해주세요" />
                  </SelectTrigger>
                  <SelectContent className="bg-white border-purple-200">
                    <SelectItem value="commuting">출퇴근</SelectItem>
                    <SelectItem value="family">가족용</SelectItem>
                    <SelectItem value="business">업무용</SelectItem>
                    <SelectItem value="leisure">레저/취미</SelectItem>
                    <SelectItem value="first_car">첫 차</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-6">
                <Button
                  type="button"
                  onClick={onSkip}
                  variant="outline"
                  className="border-purple-200 text-purple-600 hover:bg-purple-50 hover:border-purple-300 py-3"
                >
                  <SkipForward className="w-4 h-4 mr-2" />
                  게스트로 계속하기
                </Button>

                <Button
                  type="submit"
                  disabled={isSubmitting || !formData.name || !formData.email}
                  className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white py-3 shadow-lg shadow-purple-500/25"
                >
                  <div className="flex items-center gap-2">
                    {isSubmitting ? (
                      <>
                        <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        생성 중...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-4 h-4" />
                        프로필 생성하기
                      </>
                    )}
                  </div>
                </Button>
              </div>
            </form>

            <div className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-200/50">
              <h4 className="text-purple-800 font-semibold mb-3 flex items-center gap-2">
                <Sparkles className="w-5 h-5" />
                개인화 추천의 장점
              </h4>
              <ul className="text-purple-700 text-sm space-y-2">
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-purple-400 rounded-full"></div>
                  당신의 라이프스타일에 맞는 차량 추천
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-purple-400 rounded-full"></div>
                  AI 학습을 통한 점점 더 정확한 추천
                </li>
                <li className="flex items-center gap-2">
                  <div className="w-1.5 h-1.5 bg-purple-400 rounded-full"></div>
                  맞춤형 금융 상품 추천
                </li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}