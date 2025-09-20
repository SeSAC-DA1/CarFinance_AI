'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import {
  Car,
  DollarSign,
  BarChart3,
  Users,
  Send,
  Bot,
  Loader2,
  MessageCircle,
  Brain,
  Sparkles
} from 'lucide-react';

interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'agent';
  agentType?: AgentType;
  timestamp: Date;
  agentName?: string;
  agentEmoji?: string;
  thinking?: boolean;
}

enum AgentType {
  VEHICLE_EXPERT = 'VEHICLE_EXPERT',
  FINANCE_ADVISOR = 'FINANCE_ADVISOR'
}

interface Agent {
  type: AgentType;
  name: string;
  emoji: string;
  role: string;
  status: 'idle' | 'thinking' | 'responding';
  specialties: string[];
}

const AGENTS: Agent[] = [
  {
    type: AgentType.VEHICLE_EXPERT,
    name: "차량 추천 전문가",
    emoji: "🚗",
    role: "차량 추천, 성능 분석, 시장 트렌드",
    status: 'idle',
    specialties: ['차량 추천', '성능 분석', '시장 트렌드', '협업 필터링']
  },
  {
    type: AgentType.FINANCE_ADVISOR,
    name: "금융 상담 전문가",
    emoji: "💰",
    role: "자동차 금융, 대출, 리스, 할부 상담",
    status: 'idle',
    specialties: ['자동차 대출', '리스 상품', '할부 계산', '금융 옵션']
  }
];

interface MultiAgentChatProps {
  userProfile?: any;
  onRecommendationRequest?: (chatHistory: ChatMessage[]) => void;
}

export function MultiAgentChat({ userProfile, onRecommendationRequest }: MultiAgentChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      content: "안녕하세요! 🚗 차량 구매가 처음이신가요? 걱정 마세요! 저희가 자연스러운 대화로 딱 맞는 중고차를 찾아드릴게요. 간단한 질문 몇 가지만 답해주시면 실시간으로 매물을 찾아가며 협업 필터링으로 최적화해드립니다!",
      sender: 'agent',
      agentType: AgentType.VEHICLE_EXPERT,
      agentName: "차량 추천 전문가",
      agentEmoji: "🚗",
      timestamp: new Date()
    }
  ]);

  const [inputMessage, setInputMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [activeAgents, setActiveAgents] = useState<AgentType[]>([]);
  const [agents, setAgents] = useState<Agent[]>(AGENTS);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // 대화 흐름 관리
  const [conversationStep, setConversationStep] = useState(0);
  const [userPreferences, setUserPreferences] = useState({
    budget: '',
    purpose: '',
    fuelType: '',
    brand: '',
    bodyType: ''
  });
  const [isSearchingVehicles, setIsSearchingVehicles] = useState(false);
  const [foundVehicles, setFoundVehicles] = useState<any[]>([]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const updateAgentStatus = (agentType: AgentType, status: Agent['status']) => {
    setAgents(prev => prev.map(agent =>
      agent.type === agentType ? { ...agent, status } : agent
    ));
  };

  const simulateAgentThinking = (agentType: AgentType, duration: number) => {
    updateAgentStatus(agentType, 'thinking');
    setTimeout(() => {
      updateAgentStatus(agentType, 'idle');
    }, duration);
  };

  // 원클릭 시작 함수
  const startOneClickFlow = async () => {
    setIsProcessing(true);
    setConversationStep(1);

    // 첫 번째 질문 자동 시작
    await new Promise(resolve => setTimeout(resolve, 1000));

    const firstQuestion: ChatMessage = {
      id: Date.now().toString(),
      content: "먼저 예산부터 간단히 물어볼게요! 💰 대략 어느 정도 예산을 생각하고 계신가요?",
      sender: 'agent',
      agentType: AgentType.FINANCE_ADVISOR,
      agentName: "금융 상담 전문가",
      agentEmoji: "💰",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, firstQuestion]);
    setIsProcessing(false);
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isProcessing) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputMessage;
    setInputMessage('');
    setIsProcessing(true);

    // 대화 단계별 처리
    await handleConversationalFlow(currentInput);
    setIsProcessing(false);
  };

  // 빠른 응답 버튼 클릭 처리
  const handleQuickResponse = async (response: string) => {
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: response,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsProcessing(true);
    await handleConversationalFlow(response);
    setIsProcessing(false);
  };

  // 대화형 흐름 처리 (핵심 함수)
  const handleConversationalFlow = async (userInput: string) => {
    switch (conversationStep) {
      case 1: // 예산 수집
        setUserPreferences(prev => ({ ...prev, budget: userInput }));
        await askPurpose();
        setConversationStep(2);
        break;

      case 2: // 용도 수집
        setUserPreferences(prev => ({ ...prev, purpose: userInput }));
        await askFuelType();
        setConversationStep(3);
        break;

      case 3: // 연료 타입 수집
        setUserPreferences(prev => ({ ...prev, fuelType: userInput }));
        await startVehicleSearch();
        setConversationStep(4);
        break;

      case 4: // 추가 요구사항
        await handleAdditionalRequests(userInput);
        break;

      default:
        await simulateMultiAgentResponse(userInput);
    }
  };

  // 단계별 질문 함수들
  const askPurpose = async () => {
    await new Promise(resolve => setTimeout(resolve, 800));

    const purposeQuestion: ChatMessage = {
      id: Date.now().toString(),
      content: "좋아요! 이제 주로 어떤 용도로 사용하실 건가요? 🚗 (통근용, 가족용, 레저용 등)",
      sender: 'agent',
      agentType: AgentType.VEHICLE_EXPERT,
      agentName: "차량 추천 전문가",
      agentEmoji: "🚗",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, purposeQuestion]);
  };

  const askFuelType = async () => {
    await new Promise(resolve => setTimeout(resolve, 800));

    const fuelQuestion: ChatMessage = {
      id: Date.now().toString(),
      content: "연료 타입은 어떤 걸 선호하시나요? ⛽ 연비를 중시하시나요?",
      sender: 'agent',
      agentType: AgentType.VEHICLE_EXPERT,
      agentName: "차량 추천 전문가",
      agentEmoji: "🚗",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, fuelQuestion]);
  };

  // 실시간 차량 검색 시뮬레이션
  const startVehicleSearch = async () => {
    setIsSearchingVehicles(true);

    // 검색 시작 메시지
    const searchStart: ChatMessage = {
      id: Date.now().toString(),
      content: "완벽해요! 지금 실시간으로 중고차 매물을 검색하고 있습니다... 🔍",
      sender: 'agent',
      agentType: AgentType.VEHICLE_EXPERT,
      agentName: "차량 추천 전문가",
      agentEmoji: "🚗",
      timestamp: new Date()
    };
    setMessages(prev => [...prev, searchStart]);

    // 협업 필터링 시뮬레이션
    await simulateCollaborativeFiltering();

    setIsSearchingVehicles(false);
  };

  // 협업 필터링 시뮬레이션
  const simulateCollaborativeFiltering = async () => {
    updateAgentStatus(AgentType.VEHICLE_EXPERT, 'thinking');

    // 단계별 진행 상황 표시
    const steps = [
      "📊 유사한 고객 프로필 분석 중...",
      "🔄 협업 필터링 알고리즘 실행 중...",
      "🎯 개인화 추천 모델 적용 중...",
      "✨ 최적 매물 선별 완료!"
    ];

    for (let i = 0; i < steps.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1500));

      const progressMessage: ChatMessage = {
        id: Date.now().toString() + `_progress_${i}`,
        content: steps[i],
        sender: 'agent',
        agentType: AgentType.VEHICLE_EXPERT,
        agentName: "차량 추천 전문가",
        agentEmoji: "🚗",
        timestamp: new Date()
      };

      setMessages(prev => [...prev, progressMessage]);
    }

    // 최종 결과
    await new Promise(resolve => setTimeout(resolve, 1000));

    const finalResult: ChatMessage = {
      id: Date.now().toString(),
      content: `🎉 찾았어요! 예산 ${userPreferences.budget}, ${userPreferences.purpose} 용도에 딱 맞는 차량 3대를 발견했습니다! 협업 필터링으로 고객님과 비슷한 취향의 분들이 선택한 인기 차량들이에요. 금융 옵션도 함께 분석했습니다!`,
      sender: 'agent',
      agentType: AgentType.FINANCE_ADVISOR,
      agentName: "금융 상담 전문가",
      agentEmoji: "💰",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, finalResult]);
    updateAgentStatus(AgentType.VEHICLE_EXPERT, 'idle');
  };

  const handleAdditionalRequests = async (userInput: string) => {
    // 추가 요구사항 처리
    const response: ChatMessage = {
      id: Date.now().toString(),
      content: `"${userInput}"에 대해 추가로 분석해드릴게요! 더 정확한 매칭을 위해 검색 조건을 업데이트하고 있습니다...`,
      sender: 'agent',
      agentType: AgentType.VEHICLE_EXPERT,
      agentName: "차량 추천 전문가",
      agentEmoji: "🚗",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, response]);
  };

  const simulateMultiAgentResponse = async (userInput: string) => {
    // 1. 차량 전문가가 먼저 분석 시작
    updateAgentStatus(AgentType.VEHICLE_EXPERT, 'thinking');
    await new Promise(resolve => setTimeout(resolve, 1500));

    const vehicleResponse = await simulateVehicleExpertResponse(userInput);
    setMessages(prev => [...prev, vehicleResponse]);
    updateAgentStatus(AgentType.VEHICLE_EXPERT, 'idle');

    // 2. 금융 전문가가 후속 분석
    await new Promise(resolve => setTimeout(resolve, 800));
    updateAgentStatus(AgentType.FINANCE_ADVISOR, 'thinking');
    await new Promise(resolve => setTimeout(resolve, 1500));

    const financeResponse = await simulateFinanceAdvisorResponse(userInput);
    setMessages(prev => [...prev, financeResponse]);
    updateAgentStatus(AgentType.FINANCE_ADVISOR, 'idle');

    // 3. 최종 종합 의견 (차량 전문가가 마무리)
    await new Promise(resolve => setTimeout(resolve, 1000));
    const finalMessage: ChatMessage = {
      id: Date.now().toString(),
      content: "차량 추천과 금융 옵션 분석이 완료되었습니다! 더 구체적인 추천을 원하시면 말씀해 주세요. 🚗💰",
      sender: 'agent',
      agentType: AgentType.VEHICLE_EXPERT,
      agentName: "차량 추천 전문가",
      agentEmoji: "🚗",
      timestamp: new Date()
    };
    setMessages(prev => [...prev, finalMessage]);
  };

  const simulateVehicleExpertResponse = async (input: string): Promise<ChatMessage> => {
    // 실제로는 Gemini 차량 추천 에이전트 API 호출
    return {
      id: Date.now().toString() + '_vehicle',
      content: `차량 추천 전문가입니다. "${input}"에 대해 분석한 결과, 현재 시장에서 가성비 좋은 모델들을 협업 필터링과 ML 엔진으로 선별했습니다. 연비, 안전성, 유지비용을 종합 고려한 맞춤 옵션들을 준비했어요! 🚗`,
      sender: 'agent',
      agentType: AgentType.VEHICLE_EXPERT,
      agentName: "차량 추천 전문가",
      agentEmoji: "🚗",
      timestamp: new Date()
    };
  };

  const simulateFinanceAdvisorResponse = async (input: string): Promise<ChatMessage> => {
    // 실제로는 Gemini 금융 상담 에이전트 API 호출
    return {
      id: Date.now().toString() + '_finance',
      content: "금융 상담 전문가입니다. 현재 금리 상황과 고객님의 조건을 분석한 결과, 할부/리스/대출 옵션별 최적 상품을 찾았습니다. 월 부담금 시뮬레이션과 함께 가장 유리한 금융 옵션을 제안드릴게요! 💰",
      sender: 'agent',
      agentType: AgentType.FINANCE_ADVISOR,
      agentName: "금융 상담 전문가",
      agentEmoji: "💰",
      timestamp: new Date()
    };
  };

  const getAgentBadgeColor = (agentType: AgentType) => {
    switch (agentType) {
      case AgentType.VEHICLE_EXPERT: return 'bg-blue-100 text-blue-800';
      case AgentType.FINANCE_ADVISOR: return 'bg-green-100 text-green-800';
      case AgentType.DATA_ANALYST: return 'bg-purple-100 text-purple-800';
      case AgentType.COORDINATOR: return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <Card className="w-full max-w-4xl mx-auto bg-white/95 backdrop-blur-xl border-purple-200/50 shadow-2xl">
      <CardHeader className="border-b border-purple-100">
        <CardTitle className="flex items-center gap-3">
          <div className="bg-gradient-to-r from-purple-500 to-pink-500 p-2 rounded-lg">
            <MessageCircle className="w-6 h-6 text-white" />
          </div>
          실시간 AI 멀티에이전트 상담
        </CardTitle>

        {/* 에이전트 상태 표시 */}
        <div className="flex flex-wrap gap-2 mt-4">
          {agents.map((agent) => (
            <div key={agent.type} className="flex items-center gap-2">
              <Badge className={`${getAgentBadgeColor(agent.type)} border-none`}>
                <span className="mr-1">{agent.emoji}</span>
                {agent.name.split(' ')[0]}
                {agent.status === 'thinking' && (
                  <Loader2 className="w-3 h-3 ml-1 animate-spin" />
                )}
              </Badge>
            </div>
          ))}
        </div>
      </CardHeader>

      <CardContent className="p-0">
        {/* 메시지 영역 */}
        <div className="h-96 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-2xl ${
                message.sender === 'user'
                  ? 'bg-gradient-to-r from-purple-500 to-pink-500 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}>
                {message.sender === 'agent' && (
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm">{message.agentEmoji}</span>
                    <span className="text-xs font-medium text-gray-600">
                      {message.agentName}
                    </span>
                  </div>
                )}
                <p className="text-sm">{message.content}</p>
                <p className={`text-xs mt-1 ${
                  message.sender === 'user' ? 'text-white/70' : 'text-gray-500'
                }`}>
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* 입력 영역 */}
        <div className="border-t border-purple-100 p-4">
          <div className="flex gap-2">
            <Input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="차량에 대해 궁금한 것을 물어보세요..."
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              disabled={isProcessing}
              className="flex-1"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isProcessing}
              className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600"
            >
              {isProcessing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>

        {/* 원클릭 시작 또는 빠른 응답 버튼들 */}
        <div className="border-t border-purple-100 p-4 bg-purple-50">
          {conversationStep === 0 ? (
            // 원클릭 시작 버튼
            <div className="text-center">
              <Button
                onClick={startOneClickFlow}
                disabled={isProcessing}
                className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white px-8 py-4 text-lg font-semibold rounded-xl shadow-lg transform hover:scale-105 transition-all duration-300"
              >
                {isProcessing ? (
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                ) : (
                  <Sparkles className="w-5 h-5 mr-2" />
                )}
                🚗 원클릭으로 차량 찾기 시작!
              </Button>
              <p className="text-xs text-gray-500 mt-2">
                복잡한 입력 없이 자연스러운 대화로 시작해요
              </p>
            </div>
          ) : conversationStep === 1 ? (
            // 예산 빠른 선택
            <div>
              <p className="text-sm text-gray-600 mb-2">💰 빠른 선택:</p>
              <div className="flex flex-wrap gap-2">
                {['1000만원 이하', '1000-2000만원', '2000-3000만원', '3000만원 이상'].map((budget) => (
                  <Button
                    key={budget}
                    variant="outline"
                    size="sm"
                    onClick={() => handleQuickResponse(budget)}
                    className="text-xs hover:bg-purple-50"
                  >
                    {budget}
                  </Button>
                ))}
              </div>
            </div>
          ) : conversationStep === 2 ? (
            // 용도 빠른 선택
            <div>
              <p className="text-sm text-gray-600 mb-2">🚗 빠른 선택:</p>
              <div className="flex flex-wrap gap-2">
                {['출퇴근용', '가족용', '레저용', '업무용', '첫차'].map((purpose) => (
                  <Button
                    key={purpose}
                    variant="outline"
                    size="sm"
                    onClick={() => handleQuickResponse(purpose)}
                    className="text-xs hover:bg-purple-50"
                  >
                    {purpose}
                  </Button>
                ))}
              </div>
            </div>
          ) : conversationStep === 3 ? (
            // 연료 타입 빠른 선택
            <div>
              <p className="text-sm text-gray-600 mb-2">⛽ 빠른 선택:</p>
              <div className="flex flex-wrap gap-2">
                {['가솔린', '디젤', '하이브리드', '전기차', '연비 중시'].map((fuel) => (
                  <Button
                    key={fuel}
                    variant="outline"
                    size="sm"
                    onClick={() => handleQuickResponse(fuel)}
                    className="text-xs hover:bg-purple-50"
                  >
                    {fuel}
                  </Button>
                ))}
              </div>
            </div>
          ) : (
            // 추가 요청 버튼들
            <div>
              <p className="text-sm text-gray-600 mb-2">✨ 더 알아보기:</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleQuickResponse("다른 브랜드도 보여주세요")}
                  className="text-xs hover:bg-purple-50"
                >
                  🔄 다른 옵션
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleQuickResponse("금융 옵션 자세히 알려주세요")}
                  className="text-xs hover:bg-purple-50"
                >
                  💰 금융 상세
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleQuickResponse("실제 매물 보기")}
                  className="text-xs hover:bg-purple-50"
                >
                  📋 매물 보기
                </Button>
              </div>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}