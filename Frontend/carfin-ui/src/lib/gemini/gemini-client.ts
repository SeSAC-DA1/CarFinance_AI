// Gemini API 클라이언트 - 멀티 에이전트 최적화
import { GeminiRequest, GeminiResponse, AgentType } from '@/types';

export class OptimizedGeminiClient {
  private apiKey: string;
  private baseUrl = 'https://generativelanguage.googleapis.com/v1beta';
  private tokenBudget: Map<AgentType, number> = new Map();
  private promptCache: Map<string, GeminiResponse> = new Map();

  constructor(apiKey: string) {
    this.apiKey = apiKey;
    this.initializeTokenBudgets();
  }

  private initializeTokenBudgets() {
    // 에이전트별 시간당 토큰 예산
    this.tokenBudget.set(AgentType.DATA_EXTRACTOR, 10000);
    this.tokenBudget.set(AgentType.COLLABORATIVE_FILTER, 15000);
    this.tokenBudget.set(AgentType.LEARNING_AGENT, 10000);
    this.tokenBudget.set(AgentType.MARKET_AGENT, 15000);
    this.tokenBudget.set(AgentType.COORDINATOR, 8000);
  }

  async generateContent(request: GeminiRequest): Promise<GeminiResponse> {
    // 1. 캐시 확인
    const cacheKey = this.generateCacheKey(request);
    if (this.promptCache.has(cacheKey)) {
      return this.promptCache.get(cacheKey)!;
    }

    // 2. 프롬프트 최적화
    const optimizedPrompt = this.optimizePrompt(request.prompt);

    // 3. API 호출
    const response = await this.executeGeminiRequest({
      ...request,
      prompt: optimizedPrompt
    });

    // 4. 캐싱
    this.promptCache.set(cacheKey, response);

    return response;
  }

  private async executeGeminiRequest(request: GeminiRequest): Promise<GeminiResponse> {
    const model = request.model || 'gemini-1.5-flash';
    const url = `${this.baseUrl}/models/${model}:generateContent?key=${this.apiKey}`;

    const requestBody = {
      contents: [{
        parts: [{
          text: request.prompt
        }]
      }],
      generationConfig: {
        temperature: request.temperature || 0.2,
        maxOutputTokens: request.maxTokens || 1000,
        candidateCount: 1
      }
    };

    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`Gemini API error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();

      if (!data.candidates || data.candidates.length === 0) {
        throw new Error('No response from Gemini API');
      }

      const content = data.candidates[0].content.parts[0].text;
      const usage = data.usageMetadata || { inputTokens: 0, outputTokens: 0, totalTokens: 0 };

      return {
        content,
        usage: {
          input_tokens: usage.promptTokenCount || 0,
          output_tokens: usage.candidatesTokenCount || 0,
          total_tokens: usage.totalTokenCount || 0
        },
        model,
        timestamp: Date.now()
      };

    } catch (error) {
      console.error('Gemini API request failed:', error);
      throw error;
    }
  }

  private optimizePrompt(prompt: string): string {
    return prompt
      // 불필요한 단어 제거
      .replace(/please|kindly|if you could|would you mind/gi, '')
      // 중복 표현 제거
      .replace(/(\b\w+\b)(\s+\1\b)+/gi, '$1')
      // 간결한 표현으로 변경
      .replace(/in order to/gi, 'to')
      .replace(/due to the fact that/gi, 'because')
      .replace(/at this point in time/gi, 'now')
      // JSON 응답 형식 명시
      .concat('\n\nRespond in JSON format only.');
  }

  private generateCacheKey(request: GeminiRequest): string {
    return `${request.agentType}_${this.hashString(request.prompt)}_${request.temperature || 0.2}`;
  }

  private hashString(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash).toString(36);
  }

  // 토큰 사용량 추적
  trackTokenUsage(agentType: AgentType, tokensUsed: number): void {
    const currentBudget = this.tokenBudget.get(agentType) || 0;
    this.tokenBudget.set(agentType, Math.max(0, currentBudget - tokensUsed));
  }

  // 토큰 예산 확인
  canUseTokens(agentType: AgentType, tokensNeeded: number): boolean {
    const currentBudget = this.tokenBudget.get(agentType) || 0;
    return currentBudget >= tokensNeeded;
  }
}