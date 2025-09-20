/**
 * CarFin API Client - FastAPI 백엔드와 통신
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

class CarFinAPIClient {
  private baseURL: string;

  constructor() {
    this.baseURL = API_BASE_URL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    try {
      const url = `${this.baseURL}${endpoint}`;
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return { success: true, data };
    } catch (error) {
      console.error('API request failed:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  // 헬스체크
  async healthCheck() {
    return this.request('/health');
  }

  // 사용자 등록
  async registerUser(userData: {
    full_name: string;
    email: string;
    age: number;
    phone?: string;
  }) {
    return this.request('/api/v1/users/register', {
      method: 'POST',
      body: JSON.stringify(userData),
    });
  }

  // 사용자 선호도 저장
  async saveUserPreferences(userId: string, preferences: Record<string, any>) {
    return this.request(`/api/v1/users/${userId}/preferences`, {
      method: 'POST',
      body: JSON.stringify(preferences),
    });
  }

  // 차량 목록 가져오기
  async getVehicles(params?: {
    limit?: number;
    offset?: number;
    brand?: string;
    min_price?: number;
    max_price?: number;
  }) {
    const queryParams = new URLSearchParams();
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        if (value !== undefined) {
          queryParams.append(key, value.toString());
        }
      });
    }

    const endpoint = `/api/v1/vehicles${queryParams.toString() ? '?' + queryParams.toString() : ''}`;
    return this.request(endpoint);
  }

  // 특정 차량 상세 정보
  async getVehicle(vehicleId: string) {
    return this.request(`/api/v1/vehicles/${vehicleId}`);
  }

  // AI 추천 요청
  async getRecommendations(userProfile: Record<string, any>) {
    return this.request('/api/v1/recommendations', {
      method: 'POST',
      body: JSON.stringify(userProfile),
    });
  }
}

// Singleton 패턴
export const apiClient = new CarFinAPIClient();
export default apiClient;