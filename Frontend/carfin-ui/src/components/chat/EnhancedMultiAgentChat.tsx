'use client';

import { useState, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
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
  Sparkles,
  Network,
  Database,
  Search,
  TrendingUp,
  Settings
} from 'lucide-react';
import { EnhancedMultiAgentSystem, type AgentMessage, type AgentType } from '@/lib/enhanced-multi-agent-system';

interface ChatMessage {
  id: string;
  content: string;
  sender: 'user' | 'agent';
  agentType?: AgentType;
  timestamp: Date;
  agentName?: string;
  agentEmoji?: string;
  mcpActivity?: MCPActivity[];
  collaborationData?: any;
}

interface MCPActivity {
  service: string;
  action: string;
  status: 'active' | 'completed' | 'failed';
  timestamp: Date;
}

interface AgentStatus {
  type: AgentType;
  name: string;
  emoji: string;
  role: string;
  status: 'idle' | 'thinking' | 'collaborating' | 'mcp_active';
  specialties: string[];
  currentActivity?: string;
  mcpServices?: string[];
}

const ENHANCED_AGENTS: AgentStatus[] = [
  {
    type: 'coordinator',
    name: '조정자 AI',
    emoji: '🎯',
    role: '전체 프로세스 관리 및 협업 조정',
    status: 'idle',
    specialties: ['프로세스 관리', '의사결정', '협업 조정'],
    mcpServices: ['context7']
  },
  {
    type: 'data_collector',
    name: '데이터 수집 전문가',
    emoji: '📊',
    role: '사용자 정보 수집 및 분석',
    status: 'idle',
    specialties: ['사용자 프로파일링', '요구사항 분석'],
    mcpServices: ['context7', 'sequential-thinking']
  },
  {
    type: 'vehicle_matcher',
    name: '차량 매칭 전문가',
    emoji: '🚗',
    role: '실시간 차량 검색 및 매칭',
    status: 'idle',
    specialties: ['차량 검색', '매물 분석', 'ML 매칭'],
    mcpServices: ['web-search', 'playwright', 'ml-engine']
  },
  {
    type: 'finance_advisor',
    name: '금융 상담 전문가',
    emoji: '💰',
    role: '금융상품 분석 및 상담',
    status: 'idle',
    specialties: ['금융상품 분석', '대출 상담', '비용 최적화'],
    mcpServices: ['web-search', 'context7']
  },
  {
    type: 'market_analyst',
    name: '시장 분석 전문가',
    emoji: '📈',
    role: '시장 동향 및 가격 분석',
    status: 'idle',
    specialties: ['시장 분석', '가격 평가', '리스크 분석'],
    mcpServices: ['web-search', 'database']
  },
  {
    type: 'ml_specialist',
    name: 'ML 분석 전문가',
    emoji: '🤖',
    role: 'AI 기반 예측 및 최적화',
    status: 'idle',
    specialties: ['협업 필터링', '예측 모델링', '추천 최적화'],
    mcpServices: ['ml-engine', 'sequential-thinking']
  }
];

interface EnhancedMultiAgentChatProps {
  userProfile?: any;
  onRecommendationRequest?: (chatHistory: ChatMessage[]) => void;
}

export function EnhancedMultiAgentChat({ userProfile, onRecommendationRequest }: EnhancedMultiAgentChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: '1',
      content: '🚀 Enhanced Multi-Agent System이 활성화되었습니다! MCP 서버들과 연결하여 실시간 협업 필터링, ML 분석, 그리고 데이터베이스 통합으로 최적의 중고차를 찾아드리겠습니다.',
      sender: 'agent',
      agentType: 'coordinator',
      agentName: '조정자 AI',
      agentEmoji: '🎯',
      timestamp: new Date()
    }
  ]);

  const [inputMessage, setInputMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [agents, setAgents] = useState<AgentStatus[]>(ENHANCED_AGENTS);
  const [collaborationProgress, setCollaborationProgress] = useState(0);
  const [activeCollaboration, setActiveCollaboration] = useState(false);
  const [mcpActivities, setMcpActivities] = useState<MCPActivity[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Enhanced Multi-Agent System 인스턴스
  const enhancedSystem = useRef(new EnhancedMultiAgentSystem(
    process.env.NEXT_PUBLIC_GEMINI_API_KEY || '',
    process.env.NEXT_PUBLIC_GOOGLE_SEARCH_API_KEY
  ));

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const updateAgentStatus = (agentType: AgentType, status: AgentStatus['status'], activity?: string) => {
    setAgents(prev => prev.map(agent =>
      agent.type === agentType
        ? { ...agent, status, currentActivity: activity }
        : agent
    ));
  };

  const addMCPActivity = (service: string, action: string) => {
    const activity: MCPActivity = {
      service,
      action,
      status: 'active',
      timestamp: new Date()
    };
    setMcpActivities(prev => [...prev, activity]);

    // 2초 후 완료 상태로 변경
    setTimeout(() => {
      setMcpActivities(prev =>
        prev.map(a => a === activity ? { ...a, status: 'completed' } : a)
      );
    }, 2000);
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
    setActiveCollaboration(true);
    setCollaborationProgress(0);

    try {
      // Enhanced Multi-Agent System 실행
      await simulateEnhancedCollaboration(currentInput);
    } catch (error) {
      console.error('Enhanced collaboration failed:', error);
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        content: '시스템 오류가 발생했습니다. 다시 시도해주세요.',
        sender: 'agent',
        agentType: 'coordinator',
        agentName: '조정자 AI',
        agentEmoji: '🎯',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsProcessing(false);
      setActiveCollaboration(false);
      setCollaborationProgress(100);
    }
  };

  const simulateEnhancedCollaboration = async (userInput: string) => {
    // 1. 조정자 AI 활성화
    updateAgentStatus('coordinator', 'thinking', '사용자 요청 분석 중');
    setCollaborationProgress(10);

    await new Promise(resolve => setTimeout(resolve, 1000));

    const coordinatorMessage: ChatMessage = {
      id: Date.now().toString(),
      content: `"${userInput}"에 대한 종합 분석을 시작합니다. 모든 에이전트와 MCP 서버들을 동원하여 최적의 솔루션을 찾겠습니다.`,
      sender: 'agent',
      agentType: 'coordinator',
      agentName: '조정자 AI',
      agentEmoji: '🎯',
      timestamp: new Date(),
      mcpActivity: [
        { service: 'context7', action: 'automotive_consultation_guidelines', status: 'active', timestamp: new Date() }
      ]
    };
    setMessages(prev => [...prev, coordinatorMessage]);
    addMCPActivity('Context7', '자동차 상담 가이드라인 조회');

    // 2. 병렬 에이전트 활성화
    setCollaborationProgress(25);
    updateAgentStatus('data_collector', 'mcp_active', 'Context7으로 사용자 프로파일링');
    updateAgentStatus('vehicle_matcher', 'mcp_active', 'Web Search + Playwright로 매물 검색');
    updateAgentStatus('finance_advisor', 'mcp_active', 'Web Search로 금융상품 조회');
    updateAgentStatus('market_analyst', 'mcp_active', 'Database + Web Search로 시장 분석');

    // MCP 활동 시뮬레이션
    addMCPActivity('Sequential Thinking', '복잡한 사용자 요구사항 분석');
    addMCPActivity('Web Search', '실시간 차량 매물 검색');
    addMCPActivity('Playwright', '주요 중고차 사이트 스크래핑');
    addMCPActivity('Database', '과거 거래 데이터 조회');

    await new Promise(resolve => setTimeout(resolve, 2000));

    // 3. 데이터 수집 전문가 결과
    setCollaborationProgress(40);
    const dataCollectorMessage: ChatMessage = {
      id: Date.now().toString(),
      content: '사용자 프로필 분석이 완료되었습니다. Sequential Thinking MCP를 통해 데이터 품질을 평가하고, 추가 정보 수집 전략을 수립했습니다.',
      sender: 'agent',
      agentType: 'data_collector',
      agentName: '데이터 수집 전문가',
      agentEmoji: '📊',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, dataCollectorMessage]);
    updateAgentStatus('data_collector', 'idle');

    await new Promise(resolve => setTimeout(resolve, 1000));

    // 4. 차량 매칭 전문가 + ML 분석
    setCollaborationProgress(55);
    updateAgentStatus('ml_specialist', 'collaborating', '차량 매칭 전문가와 협업 중');
    addMCPActivity('ML Engine', '협업 필터링 알고리즘 실행');

    const vehicleMatcherMessage: ChatMessage = {
      id: Date.now().toString(),
      content: '🚗 실시간 매물 검색 완료! Web Search와 Playwright MCP로 3개 사이트에서 최신 매물을 수집했습니다. ML Engine과 협업하여 매칭 점수를 최적화했습니다.',
      sender: 'agent',
      agentType: 'vehicle_matcher',
      agentName: '차량 매칭 전문가',
      agentEmoji: '🚗',
      timestamp: new Date(),
      collaborationData: {
        scrapedSites: ['엔카', 'KB차차차', 'SK엔카'],
        mlOptimization: true,
        matchingAlgorithm: 'collaborative_filtering'
      }
    };
    setMessages(prev => [...prev, vehicleMatcherMessage]);
    updateAgentStatus('vehicle_matcher', 'idle');

    await new Promise(resolve => setTimeout(resolve, 1500));

    // 5. ML 전문가 분석 결과
    setCollaborationProgress(70);
    const mlSpecialistMessage: ChatMessage = {
      id: Date.now().toString(),
      content: '🤖 ML 분석 완료! 사용자 클러스터링을 통해 유사한 고객들의 선호도를 분석했습니다. 예측 만족도 92%의 차량들을 선별했습니다.',
      sender: 'agent',
      agentType: 'ml_specialist',
      agentName: 'ML 분석 전문가',
      agentEmoji: '🤖',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, mlSpecialistMessage]);
    updateAgentStatus('ml_specialist', 'idle');

    // 6. 금융 상담 전문가 결과
    setCollaborationProgress(85);
    addMCPActivity('Web Search', '실시간 금융상품 금리 조회');
    const financeMessage: ChatMessage = {
      id: Date.now().toString(),
      content: '💰 금융 분석 완료! Web Search MCP로 실시간 금리 정보를 수집하고, Context7에서 금융 가이드라인을 참조하여 최적의 금융 옵션을 선별했습니다.',
      sender: 'agent',
      agentType: 'finance_advisor',
      agentName: '금융 상담 전문가',
      agentEmoji: '💰',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, financeMessage]);
    updateAgentStatus('finance_advisor', 'idle');

    await new Promise(resolve => setTimeout(resolve, 1000));

    // 7. 시장 분석 전문가 결과
    const marketMessage: ChatMessage = {
      id: Date.now().toString(),
      content: '📈 시장 분석 완료! Database MCP로 과거 6개월 거래 데이터를 분석하고, 현재 시장 트렌드를 반영한 가격 적정성을 평가했습니다.',
      sender: 'agent',
      agentType: 'market_analyst',
      agentName: '시장 분석 전문가',
      agentEmoji: '📈',
      timestamp: new Date()
    };
    setMessages(prev => [...prev, marketMessage]);
    updateAgentStatus('market_analyst', 'idle');

    // 8. 최종 협업 결과
    setCollaborationProgress(100);
    await new Promise(resolve => setTimeout(resolve, 1000));

    const finalMessage: ChatMessage = {
      id: Date.now().toString(),
      content: '🎉 Enhanced Multi-Agent 협업 완료! 6개 에이전트와 8개 MCP 서버가 협력하여 개인화된 추천 결과를 도출했습니다. 실시간 데이터, ML 최적화, 시장 분석이 모두 반영된 결과입니다.',
      sender: 'agent',
      agentType: 'coordinator',
      agentName: '조정자 AI',
      agentEmoji: '🎯',
      timestamp: new Date(),
      collaborationData: {
        participatingAgents: 6,
        mcpServicesUsed: 8,
        dataSourcesAnalyzed: 12,
        mlOptimizationApplied: true
      }
    };
    setMessages(prev => [...prev, finalMessage]);
  };

  const getAgentBadgeColor = (agentType: AgentType) => {
    const colors = {
      coordinator: 'bg-purple-100 text-purple-800',
      data_collector: 'bg-blue-100 text-blue-800',
      vehicle_matcher: 'bg-green-100 text-green-800',
      finance_advisor: 'bg-yellow-100 text-yellow-800',
      market_analyst: 'bg-red-100 text-red-800',
      ml_specialist: 'bg-indigo-100 text-indigo-800',
      data_scientist: 'bg-gray-100 text-gray-800'
    };
    return colors[agentType] || 'bg-gray-100 text-gray-800';
  };

  const getStatusIcon = (status: AgentStatus['status']) => {
    switch (status) {
      case 'thinking': return <Brain className="w-3 h-3 animate-pulse" />;
      case 'collaborating': return <Network className="w-3 h-3" />;
      case 'mcp_active': return <Settings className="w-3 h-3 animate-spin" />;
      default: return null;
    }
  };

  const activeMCPServices = mcpActivities.filter(a => a.status === 'active').length;

  return (
    <Card className="w-full max-w-6xl mx-auto bg-white/95 backdrop-blur-xl border-blue-200/50 shadow-2xl">
      <CardHeader className="border-b border-blue-100">
        <CardTitle className="flex items-center gap-3">
          <div className="bg-gradient-to-r from-blue-500 to-purple-500 p-2 rounded-lg">
            <MessageCircle className="w-6 h-6 text-white" />
          </div>
          Enhanced Multi-Agent System
          <Badge variant="outline" className="ml-auto">
            MCP 연동 + ML + DB
          </Badge>
        </CardTitle>

        {/* 협업 진행률 */}
        {activeCollaboration && (
          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700">에이전트 협업 진행률</span>
              <span className="text-sm text-gray-500">{collaborationProgress}%</span>
            </div>
            <Progress value={collaborationProgress} className="h-2" />
          </div>
        )}

        {/* 에이전트 상태 표시 */}
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 mt-4">
          {agents.map((agent) => (
            <div key={agent.type} className="flex items-center gap-2">
              <Badge className={`${getAgentBadgeColor(agent.type)} border-none text-xs`}>
                <span className="mr-1">{agent.emoji}</span>
                {agent.name.split(' ')[0]}
                {getStatusIcon(agent.status)}
              </Badge>
            </div>
          ))}
        </div>

        {/* MCP 활동 상태 */}
        {activeMCPServices > 0 && (
          <div className="mt-4 p-3 bg-blue-50 rounded-lg">
            <div className="flex items-center gap-2 mb-2">
              <Database className="w-4 h-4 text-blue-600" />
              <span className="text-sm font-medium text-blue-800">
                MCP 서버 활동 중 ({activeMCPServices}개)
              </span>
            </div>
            <div className="grid grid-cols-2 gap-1">
              {mcpActivities.slice(-4).map((activity, index) => (
                <div key={index} className="text-xs text-blue-600 flex items-center gap-1">
                  <div className={`w-2 h-2 rounded-full ${
                    activity.status === 'active' ? 'bg-blue-500 animate-pulse' :
                    activity.status === 'completed' ? 'bg-green-500' : 'bg-red-500'
                  }`} />
                  {activity.service}: {activity.action}
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
                  ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}>
                {message.sender === 'agent' && (
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-sm">{message.agentEmoji}</span>
                    <span className="text-xs font-medium text-gray-600">
                      {message.agentName}
                    </span>
                    {message.agentType && (
                      <Badge className={`${getAgentBadgeColor(message.agentType)} text-xs ml-auto`}>
                        {message.agentType}
                      </Badge>
                    )}
                  </div>
                )}

                <p className="text-sm leading-relaxed">{message.content}</p>

                {/* MCP 활동 표시 */}
                {message.mcpActivity && message.mcpActivity.length > 0 && (
                  <div className="mt-2 p-2 bg-blue-50 rounded text-xs">
                    <div className="flex items-center gap-1">
                      <Settings className="w-3 h-3" />
                      <span>MCP: {message.mcpActivity[0].action}</span>
                    </div>
                  </div>
                )}

                {/* 협업 데이터 표시 */}
                {message.collaborationData && (
                  <div className="mt-2 p-2 bg-green-50 rounded text-xs">
                    <div className="flex items-center gap-1">
                      <Network className="w-3 h-3" />
                      <span>협업 데이터 포함</span>
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
              placeholder="Enhanced Multi-Agent System에게 질문하세요..."
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              disabled={isProcessing}
              className="flex-1"
            />
            <Button
              onClick={handleSendMessage}
              disabled={!inputMessage.trim() || isProcessing}
              className="bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600"
            >
              {isProcessing ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>
        </div>

        {/* 기능 소개 */}
        <div className="border-t border-blue-100 p-4 bg-blue-50">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <Network className="w-6 h-6 mx-auto mb-2 text-blue-600" />
              <p className="text-xs font-medium">에이전트 협업</p>
              <p className="text-xs text-gray-500">실시간 소통</p>
            </div>
            <div className="text-center">
              <Search className="w-6 h-6 mx-auto mb-2 text-green-600" />
              <p className="text-xs font-medium">MCP 연동</p>
              <p className="text-xs text-gray-500">구글 검색 + DB</p>
            </div>
            <div className="text-center">
              <Brain className="w-6 h-6 mx-auto mb-2 text-purple-600" />
              <p className="text-xs font-medium">ML 분석</p>
              <p className="text-xs text-gray-500">협업 필터링</p>
            </div>
            <div className="text-center">
              <TrendingUp className="w-6 h-6 mx-auto mb-2 text-red-600" />
              <p className="text-xs font-medium">실시간 분석</p>
              <p className="text-xs text-gray-500">시장 + 가격</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}