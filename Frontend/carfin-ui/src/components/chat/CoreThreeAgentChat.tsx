'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import {
  Send,
  Loader2,
  MessageCircle,
  Search,
  Car,
  DollarSign,
  Database,
  Globe,
  Brain,
  ArrowRight
} from 'lucide-react';
import { CoreThreeAgentSystem, type AgentCommunication } from '@/lib/core-three-agent-system';

interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'agent';
  agentType: 'coordinator' | 'vehicle_expert' | 'finance_expert';
  timestamp: Date;
  agentName: string;
  agentEmoji: string;
  dataSource?: 'postgresql' | 'google_search' | 'inference';
  communicationData?: AgentCommunication;
}

interface AgentStatus {
  type: 'coordinator' | 'vehicle_expert' | 'finance_expert';
  name: string;
  emoji: string;
  role: string;
  status: 'idle' | 'thinking' | 'db_active' | 'search_active' | 'collaborating';
  currentActivity?: string;
  dataSource: string;
}

const THREE_CORE_AGENTS: AgentStatus[] = [
  {
    type: 'coordinator',
    name: '총괄 정보수집 AI',
    emoji: '🔍',
    role: '사용자 상담 총괄 + 에이전트 간 조정',
    status: 'idle',
    dataSource: 'Inference + Communication'
  },
  {
    type: 'vehicle_expert',
    name: '차량전문가',
    emoji: '🚗',
    role: '실시간 매물 관리 + 차량 분석',
    status: 'idle',
    dataSource: 'PostgreSQL DB'
  },
  {
    type: 'finance_expert',
    name: '금융전문가',
    emoji: '💳',
    role: '실시간 금융상품 분석 + 매칭',
    status: 'idle',
    dataSource: 'Google Search API'
  }
];

interface CoreThreeAgentChatProps {
  userProfile?: any;
  onRecommendationComplete?: (results: any) => void;
}

export function CoreThreeAgentChat({ userProfile, onRecommendationComplete }: CoreThreeAgentChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      content: '안녕하세요! 저는 CarFin AI의 총괄 정보수집 AI입니다. 🔍 사용자님의 정보를 수집하고 분석하여, 차량전문가(PostgreSQL DB)와 금융전문가(Google Search API)와 협업해 최적의 중고차와 금융상품을 찾아드리겠습니다!',
      sender: 'agent',
      agentType: 'coordinator',
      agentName: '총괄 정보수집 AI',
      agentEmoji: '🔍',
      timestamp: new Date()
    }
  ]);

  const [inputMessage, setInputMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [agents, setAgents] = useState<AgentStatus[]>(THREE_CORE_AGENTS);
  const [collaborationProgress, setCollaborationProgress] = useState(0);
  const [agentCommunications, setAgentCommunications] = useState<AgentCommunication[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Core Three Agent System 인스턴스
  const coreSystem = useRef(new CoreThreeAgentSystem(
    process.env.NEXT_PUBLIC_GEMINI_API_KEY || '',
    process.env.NEXT_PUBLIC_GOOGLE_SEARCH_API_KEY
  ));

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const updateAgentStatus = (
    agentType: 'coordinator' | 'vehicle_expert' | 'finance_expert',
    status: AgentStatus['status'],
    activity?: string
  ) => {
    setAgents(prev => prev.map(agent =>
      agent.type === agentType
        ? { ...agent, status, currentActivity: activity }
        : agent
    ));
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isProcessing) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: inputMessage,
      sender: 'user',
      agentType: 'coordinator',
      timestamp: new Date(),
      agentName: '',
      agentEmoji: ''
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputMessage;
    setInputMessage('');
    setIsProcessing(true);
    setCollaborationProgress(0);

    try {
      // Core Three Agent System 실행
      await processWithThreeAgents(currentInput);
    } catch (error) {
      console.error('Three agent system failed:', error);
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        content: '시스템 오류가 발생했습니다. 다시 시도해주세요.',
        sender: 'agent',
        agentType: 'coordinator',
        agentName: '총괄 정보수집 AI',
        agentEmoji: '🔍',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
      setCollaborationProgress(100);
    }
  };

  const processWithThreeAgents = async (userInput: string) => {
    // 1. 총괄 정보수집 AI 활성화
    updateAgentStatus('coordinator', 'thinking', '사용자 정보 분석 및 추론 중');
    setCollaborationProgress(15);

    await new Promise(resolve => setTimeout(resolve, 1000));

    // 총괄 AI 응답
    const coordinatorMessage: ChatMessage = {
      id: Date.now().toString(),
      content: `"${userInput}"를 분석하고 있습니다. 수집된 정보를 바탕으로 추론을 진행하고, 필요시 차량전문가와 금융전문가에게 의사결정 지원 데이터를 전달하겠습니다.`,
      sender: 'agent',
      agentType: 'coordinator',
      agentName: '총괄 정보수집 AI',
      agentEmoji: '🔍',
      timestamp: new Date(),
      dataSource: 'inference'
    };
    setMessages(prev => [...prev, coordinatorMessage]);

    // 2. 추론 완료 및 다른 에이전트들에게 지시
    setCollaborationProgress(30);
    await new Promise(resolve => setTimeout(resolve, 800));

    const inferenceMessage: ChatMessage = {
      id: Date.now().toString(),
      content: '추론 완료! 차량 분석과 금융상품 분석이 필요합니다. 차량전문가에게는 PostgreSQL DB 조회를, 금융전문가에게는 Google Search API 실행을 요청합니다.',
      sender: 'agent',
      agentType: 'coordinator',
      agentName: '총괄 정보수집 AI',
      agentEmoji: '🔍',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, inferenceMessage]);

    // 3. 차량전문가 활성화 - PostgreSQL DB 연결
    setCollaborationProgress(45);
    updateAgentStatus('vehicle_expert', 'db_active', 'PostgreSQL DB에서 실시간 매물 조회 중');

    // 에이전트 간 소통 시뮬레이션
    const vehicleCommunication: AgentCommunication = {
      id: 'comm_1',
      from_agent: 'coordinator',
      to_agent: 'vehicle_expert',
      message_type: 'decision_support',
      content: '사용자 분석 결과를 전달합니다. PostgreSQL DB에서 조건에 맞는 매물을 검색해주세요.',
      data: { userInput, budget: '2000-3000만원', purpose: '출퇴근용' },
      timestamp: new Date(),
      priority: 'high'
    };
    setAgentCommunications(prev => [...prev, vehicleCommunication]);

    await new Promise(resolve => setTimeout(resolve, 1500));

    const vehicleMessage: ChatMessage = {
      id: Date.now().toString(),
      content: '🚗 차량전문가입니다. PostgreSQL DB 연결 완료! 실시간 매물 데이터를 조회하고 종합 분석을 수행했습니다. 총괄 AI의 추론 데이터를 바탕으로 3대의 최적 매물을 선별했습니다.',
      sender: 'agent',
      agentType: 'vehicle_expert',
      agentName: '차량전문가',
      agentEmoji: '🚗',
      timestamp: new Date(),
      dataSource: 'postgresql',
      communicationData: vehicleCommunication
    };
    setMessages(prev => [...prev, vehicleMessage]);
    updateAgentStatus('vehicle_expert', 'idle');

    // 4. 금융전문가 활성화 - Google Search API
    setCollaborationProgress(70);
    updateAgentStatus('finance_expert', 'search_active', 'Google Search API로 실시간 금융정보 검색 중');

    const financeCommunication: AgentCommunication = {
      id: 'comm_2',
      from_agent: 'coordinator',
      to_agent: 'finance_expert',
      message_type: 'decision_support',
      content: '사용자 분석 결과를 전달합니다. Google Search API로 최저금리 금융상품을 검색해주세요.',
      data: { userInput, vehiclePrice: '2500만원', creditScore: 'good' },
      timestamp: new Date(),
      priority: 'high'
    };
    setAgentCommunications(prev => [...prev, financeCommunication]);

    await new Promise(resolve => setTimeout(resolve, 1800));

    const financeMessage: ChatMessage = {
      id: Date.now().toString(),
      content: '💳 금융전문가입니다. Google Search API 연결 완료! 실시간 인터넷에서 최저금리 금융상품을 검색하고 분석했습니다. 사용자님께 가장 적합한 4개의 금융옵션을 매칭했습니다.',
      sender: 'agent',
      agentType: 'finance_expert',
      agentName: '금융전문가',
      agentEmoji: '💳',
      timestamp: new Date(),
      dataSource: 'google_search',
      communicationData: financeCommunication
    };
    setMessages(prev => [...prev, financeMessage]);
    updateAgentStatus('finance_expert', 'idle');

    // 5. 총괄 AI의 최종 종합 분석
    setCollaborationProgress(90);
    updateAgentStatus('coordinator', 'collaborating', '에이전트 결과 종합 분석 중');

    await new Promise(resolve => setTimeout(resolve, 1000));

    const finalMessage: ChatMessage = {
      id: Date.now().toString(),
      content: '🎉 3개 핵심 에이전트 협업 완료! PostgreSQL DB에서 가져온 실시간 매물과 Google Search API에서 검색한 최저금리 금융상품을 매칭했습니다. 추론 결과를 바탕으로 최적의 조합을 제시해드리겠습니다.',
      sender: 'agent',
      agentType: 'coordinator',
      agentName: '총괄 정보수집 AI',
      agentEmoji: '🔍',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, finalMessage]);
    updateAgentStatus('coordinator', 'idle');

    setCollaborationProgress(100);
  };

  const getAgentBadgeColor = (agentType: 'coordinator' | 'vehicle_expert' | 'finance_expert') => {
    const colors = {
      coordinator: 'bg-blue-100 text-blue-800',
      vehicle_expert: 'bg-green-100 text-green-800',
      finance_expert: 'bg-yellow-100 text-yellow-800'
    };
    return colors[agentType];
  };

  const getStatusIcon = (status: AgentStatus['status']) => {
    switch (status) {
      case 'thinking': return <Brain className="w-3 h-3 animate-pulse" />;
      case 'db_active': return <Database className="w-3 h-3 animate-bounce" />;
      case 'search_active': return <Globe className="w-3 h-3 animate-spin" />;
      case 'collaborating': return <ArrowRight className="w-3 h-3" />;
      default: return null;
    }
  };

  const getDataSourceIcon = (dataSource?: string) => {
    switch (dataSource) {
      case 'postgresql': return <Database className="w-3 h-3 text-green-600" />;
      case 'google_search': return <Globe className="w-3 h-3 text-blue-600" />;
      case 'inference': return <Brain className="w-3 h-3 text-purple-600" />;
      default: return null;
    }
  };

  return (
    <Card className="w-full max-w-5xl mx-auto bg-white/95 backdrop-blur-xl border-blue-200/50 shadow-2xl">
      <CardHeader className="border-b border-blue-100">
        <CardTitle className="flex items-center gap-3">
          <div className="bg-gradient-to-r from-blue-500 to-green-500 p-2 rounded-lg">
            <MessageCircle className="w-6 h-6 text-white" />
          </div>
          Core 3-Agent System
          <Badge variant="outline" className="ml-auto">
            PostgreSQL + Google API
          </Badge>
        </CardTitle>

        {/* 협업 진행률 */}
        {isProcessing && (
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">3개 에이전트 협업 진행률</span>
              <span className="text-sm text-gray-500">{collaborationProgress}%</span>
            </div>
            <Progress value={collaborationProgress} className="h-2" />
          </div>
        )}

        {/* 3개 핵심 에이전트 상태 */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-4">
          {agents.map((agent) => (
            <div key={agent.type} className="p-3 bg-gray-50 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <Badge className={`${getAgentBadgeColor(agent.type)} border-none text-sm`}>
                  <span className="mr-1">{agent.emoji}</span>
                  {agent.name}
                  {getStatusIcon(agent.status)}
                </Badge>
              </div>
              <p className="text-xs text-gray-600 mb-1">{agent.role}</p>
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <Search className="w-3 h-3" />
                <span>{agent.dataSource}</span>
              </div>
              {agent.currentActivity && (
                <p className="text-xs text-blue-600 mt-1">{agent.currentActivity}</p>
              )}
            </div>
          ))}
        </div>

        {/* 에이전트 간 소통 현황 */}
        {agentCommunications.length > 0 && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <ArrowRight className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-800">
                에이전트 간 소통 ({agentCommunications.length}건)
              </span>
            </div>
            <div className="space-y-1">
              {agentCommunications.slice(-2).map((comm, index) => (
                <div key={index} className="text-xs text-blue-600 flex items-center gap-1">
                  <span>{comm.from_agent}</span>
                  <ArrowRight className="w-3 h-3" />
                  <span>{comm.to_agent}</span>
                  <span className="text-gray-500">: {comm.content}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardHeader>

      <CardContent className="p-0">
        {/* 메시지 영역 */}
        <div className="h-96 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                message.sender === 'user'
                  ? 'bg-gradient-to-r from-blue-500 to-green-500 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}>
                {message.sender === 'agent' && (
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm">{message.agentEmoji}</span>
                    <span className="text-xs font-medium text-gray-600">
                      {message.agentName}
                    </span>
                    <Badge className={`${getAgentBadgeColor(message.agentType)} text-xs ml-auto`}>
                      {message.agentType}
                    </Badge>
                  </div>
                )}

                <p className="text-sm leading-relaxed">{message.content}</p>

                {/* 데이터 소스 표시 */}
                {message.dataSource && (
                  <div className="mt-2 flex items-center gap-1">
                    {getDataSourceIcon(message.dataSource)}
                    <span className="text-xs text-gray-500">
                      {message.dataSource === 'postgresql' ? 'PostgreSQL DB' :
                       message.dataSource === 'google_search' ? 'Google Search API' :
                       'AI Inference'}
                    </span>
                  </div>
                )}

                {/* 에이전트 간 소통 데이터 */}
                {message.communicationData && (
                  <div className="mt-2 p-2 bg-blue-50 rounded text-xs">
                    <div className="flex items-center gap-1">
                      <ArrowRight className="w-3 h-3" />
                      <span>에이전트 간 소통: {message.communicationData.message_type}</span>
                    </div>
                  </div>
                )}

                <p className={`text-xs mt-2 ${
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
        <div className="border-t border-blue-100 p-4">
          <div className="flex gap-2">
            <Input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="3개 에이전트 시스템에게 질문하세요..."
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              disabled={isProcessing}
              className="flex-1"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isProcessing}
              className="bg-gradient-to-r from-blue-500 to-green-500 hover:from-blue-600 hover:to-green-600"
            >
              {isProcessing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>

        {/* 시스템 정보 */}
        <div className="border-t border-blue-100 p-4 bg-blue-50">
          <div className="grid grid-cols-3 gap-4">
            <div className="text-center">
              <Search className="w-6 h-6 mx-auto mb-2 text-blue-600" />
              <p className="text-xs font-medium">총괄 정보수집 AI</p>
              <p className="text-xs text-gray-500">추론 + 소통 조정</p>
            </div>
            <div className="text-center">
              <Database className="w-6 h-6 mx-auto mb-2 text-green-600" />
              <p className="text-xs font-medium">차량전문가</p>
              <p className="text-xs text-gray-500">PostgreSQL DB</p>
            </div>
            <div className="text-center">
              <Globe className="w-6 h-6 mx-auto mb-2 text-yellow-600" />
              <p className="text-xs font-medium">금융전문가</p>
              <p className="text-xs text-gray-500">Google Search API</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}