'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
  Bot,
  Send,
  Loader2,
  ArrowRight,
  CheckCircle
} from 'lucide-react';
import { GeminiMultiAgent, type UserProfile, type Vehicle } from '@/lib/gemini-agents';

interface Message {
  id: string;
  type: 'user' | 'agent';
  content: string;
  timestamp: Date;
  agent?: 'consultant' | 'data_collector' | 'vehicle_expert' | 'finance_expert';
}

interface GeminiMultiAgentChatProps {
  onVehicleSelect?: (vehicles: Vehicle[]) => void;
  onUserProfileUpdate?: (profile: UserProfile) => void;
}

export function GeminiMultiAgentChat({ onVehicleSelect, onUserProfileUpdate }: GeminiMultiAgentChatProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'agent',
      content: '안녕하세요! 저는 CarFin AI 상담사입니다. 몇 가지 간단한 질문으로 당신에게 완벽한 중고차를 찾아드릴게요. 우선 어떤 용도로 차량을 찾고 계신가요?',
      timestamp: new Date(),
      agent: 'consultant'
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentStep, setCurrentStep] = useState(1);
  const [totalSteps] = useState(5);
  const [userProfile, setUserProfile] = useState<Partial<UserProfile>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const geminiAgent = useRef(new GeminiMultiAgent(process.env.NEXT_PUBLIC_GEMINI_API_KEY || ''));

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const quickResponses = [
    { step: 1, options: ['출퇴근용', '가족용', '레저용', '사업용'] },
    { step: 2, options: ['3000만원 이하', '3000-5000만원', '5000만원 이상', '예산 상담 필요'] },
    { step: 3, options: ['가솔린', '하이브리드', '전기차', '디젤'] },
    { step: 4, options: ['연비 중시', '안전성 중시', '디자인 중시', '브랜드 중시'] },
    { step: 5, options: ['즉시 구매', '1개월 내', '3개월 내', '천천히 알아보기'] }
  ];

  const handleSend = async (message?: string) => {
    const userMessage = message || input.trim();
    if (!userMessage || isLoading) return;

    setInput('');
    setIsLoading(true);

    // 사용자 메시지 추가
    const newUserMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: userMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newUserMessage]);

    try {
      let response: any;
      let currentAgent: 'consultant' | 'data_collector' | 'vehicle_expert' | 'finance_expert' = 'data_collector';
      let agentContent = '';

      // 진행 단계에 따른 에이전트 선택
      if (currentStep <= 3) {
        // 1-3단계: 데이터 수집 AI
        response = await geminiAgent.current.conversationalDataCollection(userMessage, userProfile);
        currentAgent = 'data_collector';
        agentContent = response.response;

        // 프로필 업데이트
        const updatedProfile = { ...userProfile, ...response.collected_data };
        setUserProfile(updatedProfile);
        onUserProfileUpdate?.(updatedProfile as UserProfile);
      }
      else if (currentStep === 4) {
        // 4단계: 차량 전문가 등장
        const collaborativeResult = await geminiAgent.current.collaborativeRecommendation(userMessage, userProfile);

        // 차량 전문가 응답 먼저
        const vehicleExpertResponse = collaborativeResult.agent_discussion.find(a => a.agent === 'vehicle_expert');
        currentAgent = 'vehicle_expert';
        agentContent = vehicleExpertResponse?.message || '차량 분석을 진행하고 있습니다...';

        // 잠시 후 금융 전문가도 등장
        setTimeout(() => {
          const financeExpertResponse = collaborativeResult.agent_discussion.find(a => a.agent === 'finance_expert');
          const financeMessage: Message = {
            id: (Date.now() + 3).toString(),
            type: 'agent',
            content: financeExpertResponse?.message || '금융 옵션을 분석해드리겠습니다.',
            timestamp: new Date(),
            agent: 'finance_expert'
          };
          setMessages(prev => [...prev, financeMessage]);
        }, 2000);
      }
      else {
        // 5단계: 종합 상담 (협업 모드)
        const collaborativeResult = await geminiAgent.current.collaborativeRecommendation(userMessage, userProfile);

        // 두 전문가가 번갈아 응답
        for (const agentResponse of collaborativeResult.agent_discussion) {
          const agentMessage: Message = {
            id: (Date.now() + Math.random()).toString(),
            type: 'agent',
            content: agentResponse.message,
            timestamp: new Date(),
            agent: agentResponse.agent as 'vehicle_expert' | 'finance_expert'
          };

          setTimeout(() => {
            setMessages(prev => [...prev, agentMessage]);
          }, collaborativeResult.agent_discussion.indexOf(agentResponse) * 1500);
        }

        currentAgent = 'consultant';
        agentContent = '두 전문가의 협업 분석이 완료되었습니다. 위의 추천 내용을 참고해주세요!';
      }

      // 현재 에이전트 응답 추가
      const agentMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'agent',
        content: agentContent,
        timestamp: new Date(),
        agent: currentAgent
      };

      setMessages(prev => [...prev, agentMessage]);

      // 다음 단계 진행 로직
      if (currentStep >= totalSteps) {
        // 최종 완료 - 차량 선택 단계로 이동
        setTimeout(() => {
          const finalMessage: Message = {
            id: (Date.now() + 2).toString(),
            type: 'agent',
            content: '상담이 완료되었습니다! 이제 추천된 차량들을 확인해보세요.',
            timestamp: new Date(),
            agent: 'consultant'
          };
          setMessages(prev => [...prev, finalMessage]);

          setTimeout(() => {
            if (onVehicleSelect) {
              onVehicleSelect(userProfile);
            }
          }, 2000);
        }, 1000);
      } else {
        // 다음 단계로 이동
        setCurrentStep(Math.min(currentStep + 1, totalSteps));
      }

    } catch (error) {
      console.error('Chat error:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'agent',
        content: '죄송합니다. 잠시 문제가 발생했습니다. 다시 말씀해주시겠어요?',
        timestamp: new Date(),
        agent: 'consultant'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const getCurrentQuickResponses = () => {
    return quickResponses.find(q => q.step === currentStep)?.options || [];
  };

  const getAgentInfo = (agent?: string) => {
    switch (agent) {
      case 'consultant':
        return { name: '상담사', icon: '👨‍💼', color: 'bg-blue-600' };
      case 'data_collector':
        return { name: '정보수집 AI', icon: '🔍', color: 'bg-green-600' };
      case 'vehicle_expert':
        return { name: '차량전문가', icon: '🚗', color: 'bg-purple-600' };
      case 'finance_expert':
        return { name: '금융전문가', icon: '💳', color: 'bg-orange-600' };
      default:
        return { name: 'AI 어시스턴트', icon: '🤖', color: 'bg-blue-600' };
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto">
      {/* 진행률 표시 */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700">진행률</span>
          <span className="text-sm text-gray-500">{currentStep}/{totalSteps}</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${(currentStep / totalSteps) * 100}%` }}
          ></div>
        </div>
      </div>

      {/* 메시지 영역 */}
      <div className="h-96 overflow-y-auto mb-6 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex items-start gap-3 max-w-xs lg:max-w-md`}>
              {message.type === 'agent' && (
                <div className="flex flex-col items-center gap-1">
                  <div className={`w-8 h-8 ${getAgentInfo(message.agent).color} rounded-full flex items-center justify-center flex-shrink-0`}>
                    <span className="text-sm">{getAgentInfo(message.agent).icon}</span>
                  </div>
                  <span className="text-xs text-gray-500 text-center leading-tight">
                    {getAgentInfo(message.agent).name}
                  </span>
                </div>
              )}

              <div
                className={`px-4 py-3 rounded-2xl ${
                  message.type === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-900'
                }`}
              >
                <p className="text-sm leading-relaxed">{message.content}</p>
              </div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="flex items-start gap-3">
              <div className="flex flex-col items-center gap-1">
                <div className={`w-8 h-8 ${currentStep <= 3 ? 'bg-green-600' : currentStep === 4 ? 'bg-purple-600' : 'bg-orange-600'} rounded-full flex items-center justify-center`}>
                  <span className="text-sm">
                    {currentStep <= 3 ? '🔍' : currentStep === 4 ? '🚗' : '💳'}
                  </span>
                </div>
                <span className="text-xs text-gray-500 text-center leading-tight">
                  {currentStep <= 3 ? '정보수집 AI' : currentStep === 4 ? '차량전문가' : '금융전문가'}
                </span>
              </div>
              <div className="bg-gray-100 px-4 py-3 rounded-2xl">
                <div className="flex items-center gap-2">
                  <Loader2 className={`w-4 h-4 animate-spin ${currentStep <= 3 ? 'text-green-600' : currentStep === 4 ? 'text-purple-600' : 'text-orange-600'}`} />
                  <span className="text-sm text-gray-600">
                    {currentStep <= 3 ? '정보수집 AI가 분석하고 있어요...' :
                     currentStep === 4 ? '차량전문가가 추천차량을 분석하고 있어요...' :
                     '금융전문가가 최적 금융상품을 찾고 있어요...'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* 빠른 응답 버튼 */}
      {getCurrentQuickResponses().length > 0 && !isLoading && (
        <div className="mb-4">
          <p className="text-xs text-gray-500 mb-2">빠른 선택:</p>
          <div className="flex flex-wrap gap-2">
            {getCurrentQuickResponses().map((option, index) => (
              <Button
                key={index}
                variant="outline"
                size="sm"
                onClick={() => handleSend(option)}
                className="text-xs border-gray-300 text-gray-700 hover:bg-blue-50 hover:border-blue-300 hover:text-blue-700"
              >
                {option}
              </Button>
            ))}
          </div>
        </div>
      )}

      {/* 입력 영역 */}
      <div className="flex gap-3">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="메시지를 입력하세요..."
          disabled={isLoading}
          className="flex-1 border-gray-300 focus:border-blue-500 focus:ring-blue-500"
        />
        <Button
          onClick={() => handleSend()}
          disabled={isLoading || !input.trim()}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4"
        >
          <Send className="w-4 h-4" />
        </Button>
      </div>

      {/* 완료 상태 */}
      {currentStep >= totalSteps && (
        <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-xl">
          <div className="flex items-center gap-2 text-green-700">
            <CheckCircle className="w-5 h-5" />
            <span className="font-medium">정보 수집 완료!</span>
          </div>
          <p className="text-sm text-green-600 mt-1">
            이제 당신에게 맞는 차량들을 찾아보겠습니다.
          </p>
        </div>
      )}
    </div>
  );
}