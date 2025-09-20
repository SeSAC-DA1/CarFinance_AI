'use client';

import { useState, useEffect } from 'react';
import { Container } from '@/components/design-system/layout/Container';
import { Button } from '@/components/ui/button';
import {
  Car,
  ArrowRight,
  MessageCircle,
  Play,
  CheckCircle,
  Sparkles,
  Zap,
  Shield,
  TrendingUp,
  Users,
  Star,
  ChevronDown,
  Brain,
  Target,
  Globe
} from 'lucide-react';

interface ModernLandingPageProps {
  onGetStarted: () => void;
}

export function ModernLandingPage({ onGetStarted }: ModernLandingPageProps) {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const features = [
    {
      icon: <Shield className="w-8 h-8" />,
      title: "투명한 실시간 분석",
      description: "숨겨진 비용 없이 모든 차량 정보와 시장가를 실시간으로 투명하게 분석해드려요",
      color: "from-blue-500 to-purple-600"
    },
    {
      icon: <Zap className="w-8 h-8" />,
      title: "원클릭 편리함",
      description: "복잡한 절차 없이 간단한 대화만으로 당신에게 딱 맞는 차량을 찾아드려요",
      color: "from-green-500 to-blue-600"
    },
    {
      icon: <TrendingUp className="w-8 h-8" />,
      title: "최저 금리 보장",
      description: "4대 전문 AI가 협업하여 시중 최저 금리와 최적 금융 상품을 찾아드려요",
      color: "from-purple-500 to-pink-600"
    },
    {
      icon: <Target className="w-8 h-8" />,
      title: "특화된 맞춤 추천",
      description: "당신의 라이프스타일과 예산에 완벽하게 맞는 최적의 차량만 추천해드려요",
      color: "from-yellow-500 to-orange-600"
    }
  ];

  const stats = [
    { number: "15만+", label: "추천 완료", icon: <TrendingUp className="w-5 h-5" /> },
    { number: "98%", label: "만족도", icon: <Star className="w-5 h-5" /> },
    { number: "3분", label: "평균 시간", icon: <Zap className="w-5 h-5" /> },
    { number: "24/7", label: "AI 상담", icon: <MessageCircle className="w-5 h-5" /> }
  ];

  const steps = [
    {
      number: "01",
      title: "간단한 대화로 시작",
      description: "전문 상담사 AI와 자연스러운 대화로 당신의 차량 필요와 예산을 파악해요",
      icon: <MessageCircle className="w-8 h-8" />,
      color: "text-blue-600"
    },
    {
      number: "02",
      title: "실시간 매물 분석",
      description: "차량 전문가 AI가 시장의 모든 매물을 실시간으로 분석하여 최적 차량을 찾아요",
      icon: <Car className="w-8 h-8" />,
      color: "text-green-600"
    },
    {
      number: "03",
      title: "맞춤 추천 완성",
      description: "당신만의 개인화된 분석 결과로 차량의 장단점을 명확하게 비교해드려요",
      icon: <Target className="w-8 h-8" />,
      color: "text-purple-600"
    },
    {
      number: "04",
      title: "최저 금리 상담",
      description: "금융 전문가 AI가 최저 금리 대출부터 리스까지 최적의 구매 방법을 제안해요",
      icon: <CheckCircle className="w-8 h-8" />,
      color: "text-orange-600"
    }
  ];

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/95 backdrop-blur-md border-b border-gray-100 z-50">
        <Container>
          <div className="flex items-center justify-between py-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl flex items-center justify-center">
                <Car className="w-6 h-6 text-white" />
              </div>
              <div>
                <span className="text-xl font-bold text-gray-900">CarFin AI</span>
                <span className="ml-2 px-2 py-1 bg-blue-100 text-blue-700 text-xs font-medium rounded-full">
                  Beta
                </span>
              </div>
            </div>

            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="sm"
                aria-label="서비스 소개 페이지로 이동"
              >
                서비스 소개
              </Button>
              <Button
                variant="default"
                size="sm"
                onClick={onGetStarted}
                icon={<Sparkles className="w-4 h-4" aria-hidden="true" />}
                aria-label="AI 차량 추천 서비스 시작하기"
              >
                시작하기
              </Button>
            </div>
          </div>
        </Container>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 bg-gradient-to-br from-blue-50 via-white to-purple-50">
        <Container>
          <div className={`text-center max-w-4xl mx-auto transition-all duration-1000 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-8'
          }`}>

            {/* Hero Badge */}
            <div className="inline-flex items-center gap-2 bg-gradient-to-r from-purple-50 to-blue-50 text-purple-700 px-6 py-3 rounded-full mb-8 border border-purple-200">
              <Brain className="w-5 h-5" aria-hidden="true" />
              <span className="text-sm font-bold">🤖 초개인화 AI 매칭 시스템</span>
            </div>

            {/* Hero Title */}
            <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-6 leading-tight">
              <span className="bg-gradient-to-r from-purple-600 to-blue-600 bg-clip-text text-transparent">
                멀티 에이전트가 협업하는
              </span><br />
              초개인화 중고차 매칭
            </h1>

            {/* Hero Description */}
            <div className="space-y-4 mb-8">
              <p className="text-xl md:text-2xl font-bold text-gray-800 leading-relaxed">
                🕰️ <span className="text-blue-600">원클릭</span> + 🎯 <span className="text-purple-600">초개인화</span> + 🤖 <span className="text-green-600">멀티 에이전트</span>
              </p>
              <p className="text-lg md:text-xl text-gray-600 leading-relaxed max-w-4xl mx-auto">
                당신의 라이프스타일, 예산, 선호도를 학습한 3대 AI 전문가가<br/>
                수만 대 매물 중에서 <strong className="text-gray-800">당신만을 위한 딱 그 차</strong>를 원클릭으로 찾아드립니다
              </p>
            </div>

            {/* AI Process Preview */}
            <div className="bg-gradient-to-r from-blue-50 via-purple-50 to-pink-50 rounded-2xl p-8 mb-12 max-w-5xl mx-auto border border-purple-100">
              <div className="text-center mb-6">
                <h3 className="text-lg font-bold text-gray-800 mb-2">🚀 왜 CarFin AI인가?</h3>
                <p className="text-sm text-gray-600">기존 중고차 사이트와는 완전히 다른 경험</p>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Brain className="w-8 h-8 text-white" />
                  </div>
                  <div className="font-bold text-lg text-gray-800">초개인화 학습</div>
                  <div className="text-sm text-gray-600 mt-2">당신의 라이프스타일을 학습한 AI가<br/>딱 맞는 차량만 선별</div>
                </div>
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Zap className="w-8 h-8 text-white" />
                  </div>
                  <div className="font-bold text-lg text-gray-800">원클릭 매칭</div>
                  <div className="text-sm text-gray-600 mt-2">복잡한 비교쇼핑 없이<br/>바로 최적 매물 + 금융상품 제안</div>
                </div>
                <div className="text-center">
                  <div className="w-16 h-16 bg-gradient-to-r from-green-500 to-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
                    <Users className="w-8 h-8 text-white" />
                  </div>
                  <div className="font-bold text-lg text-gray-800">3대 AI 협업</div>
                  <div className="text-sm text-gray-600 mt-2">정보수집 + 차량전문가 + 금융전문가<br/>동시 협업으로 완벽 분석</div>
                </div>
              </div>
            </div>


            {/* Hero CTAs */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
              <Button
                variant="default"
                size="lg"
                onClick={onGetStarted}
                icon={<Sparkles className="w-5 h-5" aria-hidden="true" />}
                className="bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white shadow-xl transform hover:scale-105 transition-all duration-200"
                aria-label="AI 원클릭 매칭 체험하기"
              >
                🎆 원클릭 매칭 체험하기
              </Button>

              <Button
                variant="outline"
                size="lg"
                icon={<Play className="w-6 h-6" aria-hidden="true" />}
                className="border-2 border-gray-300 hover:border-purple-500 hover:text-purple-600"
                aria-label="서비스 데모 비디오 재생하기"
              >
                데모 보기 (2분)
              </Button>
            </div>

            {/* Service Stats */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              {stats.map((stat, index) => (
                <div
                  key={index}
                  className={`bg-white rounded-2xl p-6 shadow-sm border border-gray-100 text-center transition-all duration-500 ${
                    isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
                  }`}
                  style={{ transitionDelay: `${index * 150}ms` }}
                >
                  <div className="flex items-center justify-center gap-2 text-blue-600 mb-2">
                    {stat.icon}
                  </div>
                  <div className="text-2xl font-bold text-gray-900 mb-1">{stat.number}</div>
                  <div className="text-sm text-gray-600">{stat.label}</div>
                </div>
              ))}
            </div>
          </div>
        </Container>

        {/* Scroll Indicator */}
        <div className="text-center mt-16">
          <ChevronDown className="w-6 h-6 text-gray-400 mx-auto animate-bounce" />
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-white">
        <Container>
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              당신을 위한 특별한 경험
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              복잡한 중고차 구매 과정을 간단하고 투명하게 만들어, 당신에게 완벽한 차량을 찾아드립니다.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <div
                key={index}
                className="bg-gradient-to-br from-white to-gray-50 rounded-3xl p-8 border border-gray-100 hover:shadow-2xl transition-all duration-300 group"
              >
                <div className={`w-16 h-16 bg-gradient-to-r ${feature.color} rounded-2xl flex items-center justify-center text-white mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  {feature.icon}
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4">{feature.title}</h3>
                <p className="text-gray-600 text-lg leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>
        </Container>
      </section>

      {/* Benefits Section */}
      <section className="py-24 bg-gradient-to-br from-gray-900 to-blue-900 text-white">
        <Container>
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              🤖 4대 전문 AI가 협업하는 이유
            </h2>
            <p className="text-xl text-blue-100 max-w-3xl mx-auto">
              각각의 전문 영역에서 최고 성능을 발휘하는 Gemini AI 에이전트들이<br/>
              실시간 협업하여 당신만을 위한 완벽한 솔루션을 만들어냅니다
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-4 gap-6 mb-16">
            {/* 상담사 AI */}
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <div className="w-14 h-14 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl flex items-center justify-center mb-4">
                <Users className="w-7 h-7 text-white" />
              </div>
              <h3 className="text-xl font-bold mb-3">전문 상담사 AI</h3>
              <p className="text-blue-100 mb-3 text-sm">
                친근한 전문가처럼 대화하며 당신의 요구사항을 정확하게 파악
              </p>
              <ul className="text-xs text-blue-200 space-y-1">
                <li>• 환영 및 서비스 안내</li>
                <li>• 24시간 친근한 서비스</li>
              </ul>
            </div>

            {/* 차량전문가 AI */}
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <div className="w-14 h-14 bg-gradient-to-r from-green-500 to-blue-600 rounded-xl flex items-center justify-center mb-4">
                <Car className="w-7 h-7 text-white" />
              </div>
              <h3 className="text-xl font-bold mb-3">차량 전문가 AI</h3>
              <p className="text-blue-100 mb-3 text-sm">
                실시간 시장 데이터를 분석하여 당신에게 딱 맞는 차량만 선별
              </p>
              <ul className="text-xs text-blue-200 space-y-1">
                <li>• 수만 대 매물 중 3대 선별</li>
                <li>• 성능, 연비, 안전성 종합</li>
              </ul>
            </div>

            {/* 정보수집 AI */}
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <div className="w-14 h-14 bg-gradient-to-r from-orange-500 to-red-600 rounded-xl flex items-center justify-center mb-4">
                <Brain className="w-7 h-7 text-white" />
              </div>
              <h3 className="text-xl font-bold mb-3">정보수집 AI</h3>
              <p className="text-blue-100 mb-3 text-sm">
                자연스러운 대화로 당신의 요구사항을 정확하게 파악
              </p>
              <ul className="text-xs text-blue-200 space-y-1">
                <li>• 예산, 용도, 선호도 분석</li>
                <li>• 대화형 정보 수집</li>
              </ul>
            </div>

            {/* 금융전문가 AI */}
            <div className="bg-white/10 backdrop-blur-sm rounded-2xl p-6 border border-white/20">
              <div className="w-14 h-14 bg-gradient-to-r from-purple-500 to-pink-600 rounded-xl flex items-center justify-center mb-4">
                <TrendingUp className="w-7 h-7 text-white" />
              </div>
              <h3 className="text-xl font-bold mb-3">금융 전문가 AI</h3>
              <p className="text-blue-100 mb-3 text-sm">
                주요 은행의 최신 금리를 실시간 비교하여 최저 비용 옵션 제안
              </p>
              <ul className="text-xs text-blue-200 space-y-1">
                <li>• 대출, 리스, 할부 비교</li>
                <li>• 개인 맞춤 최적 조건</li>
              </ul>
            </div>
          </div>

          {/* AI Technology Highlight */}
          <div className="bg-white/5 rounded-2xl p-8 border border-white/10">
            <div className="text-center mb-6">
              <h3 className="text-2xl font-bold text-white mb-2">🤖 최신 Gemini AI 기술 기반</h3>
              <p className="text-blue-100">각 전문 영역에서 최고 성능을 발휘하는 멀티 에이전트 시스템</p>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
              <div>
                <div className="text-3xl font-bold text-blue-300 mb-2">4대</div>
                <div className="text-blue-100">전문 AI 협업</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-green-300 mb-2">실시간</div>
                <div className="text-blue-100">시장데이터 분석</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-purple-300 mb-2">3분</div>
                <div className="text-blue-100">전체 프로세스 완료</div>
              </div>
              <div>
                <div className="text-3xl font-bold text-orange-300 mb-2">무료</div>
                <div className="text-blue-100">수수료 없는 서비스</div>
              </div>
            </div>
          </div>
        </Container>
      </section>

      {/* Process Section */}
      <section className="py-24 bg-gradient-to-br from-gray-50 to-blue-50">
        <Container>
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
              간단한 4단계 프로세스
            </h2>
            <p className="text-xl text-gray-600">
              대화만으로 시작해서 최적의 차량과 금융상품까지 한 번에 해결합니다
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <div key={index} className="text-center group">
                {/* Step Number */}
                <div className="relative mb-6">
                  <div className="w-20 h-20 bg-white rounded-full shadow-lg flex items-center justify-center mx-auto border-4 border-gray-100 group-hover:border-blue-200 transition-all duration-300">
                    <span className="text-2xl font-bold text-gray-400">{step.number}</span>
                  </div>
                  <div className={`absolute inset-0 w-20 h-20 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full opacity-0 group-hover:opacity-10 transition-opacity duration-300 mx-auto`} />
                </div>

                {/* Step Icon */}
                <div className={`inline-flex items-center justify-center w-12 h-12 rounded-xl mb-4 ${step.color} bg-white shadow-sm`}>
                  {step.icon}
                </div>

                {/* Step Content */}
                <h3 className="text-xl font-bold text-gray-900 mb-3">{step.title}</h3>
                <p className="text-gray-600">{step.description}</p>

                {/* Connection Line */}
                {index < steps.length - 1 && (
                  <div className="hidden lg:block absolute top-10 left-1/2 w-full h-px bg-gradient-to-r from-gray-200 to-transparent transform translate-x-10" />
                )}
              </div>
            ))}
          </div>
        </Container>
      </section>

      {/* CTA Section */}
      <section className="py-24 bg-gradient-to-r from-blue-600 to-purple-700 text-white">
        <Container>
          <div className="text-center max-w-4xl mx-auto">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              이제 당신 차례예요
            </h2>
            <p className="text-xl mb-12 opacity-90">
              벌써 15만 명이 CarFin AI와 함께 딱 맞는 차량을 찾았어요.<br />
              다음은 당신 차례입니다. 지금 바로 시작해보세요!
            </p>

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button
                variant="secondary"
                size="lg"
                onClick={onGetStarted}
                icon={<Sparkles className="w-6 h-6" aria-hidden="true" />}
                className="bg-white text-blue-700 hover:bg-gray-50 shadow-xl"
                aria-label="무료로 AI 차량 추천 서비스 시작하기"
              >
                무료로 시작하기
              </Button>

              <Button
                variant="outline"
                size="lg"
                className="border-white text-white hover:bg-white/10"
                aria-label="서비스 상세 정보 더 알아보기"
              >
                더 알아보기
              </Button>
            </div>
          </div>
        </Container>
      </section>

      {/* Footer */}
      <footer className="py-12 bg-gray-900 text-white">
        <Container>
          <div className="flex flex-col md:flex-row items-center justify-between">
            <div className="flex items-center gap-3 mb-4 md:mb-0">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-blue-700 rounded-lg flex items-center justify-center">
                <Car className="w-5 h-5 text-white" />
              </div>
              <span className="text-lg font-semibold">CarFin AI</span>
            </div>

            <div className="text-center md:text-right">
              <p className="text-gray-400 text-sm mb-1">
                AI 기반 중고차 추천 플랫폼
              </p>
              <p className="text-gray-500 text-xs">
                © 2025 CarFin AI. All rights reserved.
              </p>
            </div>
          </div>
        </Container>
      </footer>
    </div>
  );
}