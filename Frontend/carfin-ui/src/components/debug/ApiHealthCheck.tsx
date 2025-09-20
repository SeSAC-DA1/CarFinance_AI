'use client';

import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { apiClient } from '@/lib/api-client';
import { CheckCircle, XCircle, Loader2 } from 'lucide-react';

export function ApiHealthCheck() {
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const checkHealth = async () => {
    setStatus('loading');
    setError(null);

    const response = await apiClient.healthCheck();

    if (response.success) {
      setStatus('success');
      setData(response.data);
    } else {
      setStatus('error');
      setError(response.error || 'Unknown error');
    }
  };

  useEffect(() => {
    // 자동으로 한번 체크
    checkHealth();
  }, []);

  const getStatusIcon = () => {
    switch (status) {
      case 'loading':
        return <Loader2 className="w-5 h-5 animate-spin" />;
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-600" />;
      default:
        return null;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'loading':
        return 'border-yellow-200 bg-yellow-50';
      case 'success':
        return 'border-green-200 bg-green-50';
      case 'error':
        return 'border-red-200 bg-red-50';
      default:
        return 'border-gray-200 bg-gray-50';
    }
  };

  return (
    <div className={`p-4 border rounded-lg ${getStatusColor()}`}>
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold">Backend API Status</h3>
        <div className="flex items-center space-x-2">
          {getStatusIcon()}
          <span className="text-sm font-medium">
            {status === 'loading' && 'Checking...'}
            {status === 'success' && 'Connected'}
            {status === 'error' && 'Failed'}
            {status === 'idle' && 'Ready'}
          </span>
        </div>
      </div>

      {status === 'success' && data && (
        <div className="text-sm text-gray-600 mb-3">
          <p><strong>Environment:</strong> {data.environment}</p>
          <p><strong>PostgreSQL:</strong> {data.databases?.postgresql}</p>
          <p><strong>Supabase:</strong> {data.databases?.supabase_auth}</p>
        </div>
      )}

      {status === 'error' && error && (
        <div className="text-sm text-red-600 mb-3">
          <p><strong>Error:</strong> {error}</p>
        </div>
      )}

      <Button
        onClick={checkHealth}
        disabled={status === 'loading'}
        variant="outline"
        size="sm"
      >
        {status === 'loading' ? 'Checking...' : 'Test Connection'}
      </Button>
    </div>
  );
}