'use client';

import { useEffect, useRef } from 'react';

interface PentagonData {
  performance: number;  // 성능 (0-100)
  price: number;       // 가격 매력도 (0-100)
  safety: number;      // 안전성 (0-100)
  fuel_efficiency: number; // 연비 (0-100)
  design: number;      // 디자인 (0-100)
}

interface PentagonChartProps {
  data: PentagonData;
  size?: number;
  className?: string;
}

export function PentagonChart({ data, size = 200, className = '' }: PentagonChartProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    drawChart();
  }, [data, size]);

  const drawChart = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // 캔버스 초기화
    ctx.clearRect(0, 0, size, size);

    const centerX = size / 2;
    const centerY = size / 2;
    const radius = size * 0.35; // 여백을 위해 35%만 사용

    // 5각형의 각 꼭짓점 계산 (시계 방향, 12시부터 시작)
    const points = [];
    for (let i = 0; i < 5; i++) {
      const angle = (i * 2 * Math.PI / 5) - Math.PI / 2; // -90도부터 시작
      points.push({
        x: centerX + radius * Math.cos(angle),
        y: centerY + radius * Math.sin(angle)
      });
    }

    // 배경 격자 그리기 (20%, 40%, 60%, 80%, 100%)
    for (let level = 1; level <= 5; level++) {
      const levelRadius = radius * (level / 5);

      ctx.strokeStyle = level === 5 ? '#e5e7eb' : '#f3f4f6'; // 외곽선은 진하게
      ctx.lineWidth = level === 5 ? 2 : 1;
      ctx.beginPath();

      for (let i = 0; i < 5; i++) {
        const angle = (i * 2 * Math.PI / 5) - Math.PI / 2;
        const x = centerX + levelRadius * Math.cos(angle);
        const y = centerY + levelRadius * Math.sin(angle);

        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.closePath();
      ctx.stroke();
    }

    // 축 선 그리기
    ctx.strokeStyle = '#e5e7eb';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      ctx.beginPath();
      ctx.moveTo(centerX, centerY);
      ctx.lineTo(points[i].x, points[i].y);
      ctx.stroke();
    }

    // 데이터 포인트 계산
    const dataPoints = [
      data.performance,
      data.price,
      data.safety,
      data.fuel_efficiency,
      data.design
    ].map((value, i) => {
      const normalizedValue = Math.max(0, Math.min(100, value)) / 100;
      const angle = (i * 2 * Math.PI / 5) - Math.PI / 2;
      return {
        x: centerX + radius * normalizedValue * Math.cos(angle),
        y: centerY + radius * normalizedValue * Math.sin(angle),
        value
      };
    });

    // 데이터 영역 채우기
    ctx.fillStyle = 'rgba(59, 130, 246, 0.2)'; // 파란색 투명
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
    ctx.lineWidth = 3;

    ctx.beginPath();
    dataPoints.forEach((point, i) => {
      if (i === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // 데이터 포인트에 원 그리기
    ctx.fillStyle = '#3b82f6';
    dataPoints.forEach(point => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
      ctx.fill();
    });

    // 라벨 그리기
    const labels = ['성능', '가격', '안전성', '연비', '디자인'];
    const values = [data.performance, data.price, data.safety, data.fuel_efficiency, data.design];

    ctx.fillStyle = '#374151';
    ctx.font = 'bold 12px sans-serif';
    ctx.textAlign = 'center';

    labels.forEach((label, i) => {
      const angle = (i * 2 * Math.PI / 5) - Math.PI / 2;
      const labelRadius = radius + 25;
      const x = centerX + labelRadius * Math.cos(angle);
      const y = centerY + labelRadius * Math.sin(angle);

      // 라벨 위치 조정
      let textAlign: CanvasTextAlign = 'center';
      let offsetY = 0;

      if (i === 0) { // 성능 (위)
        offsetY = -5;
      } else if (i === 1) { // 가격 (오른쪽 위)
        textAlign = 'left';
        offsetY = -5;
      } else if (i === 2) { // 안전성 (오른쪽 아래)
        textAlign = 'left';
        offsetY = 5;
      } else if (i === 3) { // 연비 (왼쪽 아래)
        textAlign = 'right';
        offsetY = 5;
      } else if (i === 4) { // 디자인 (왼쪽 위)
        textAlign = 'right';
        offsetY = -5;
      }

      ctx.textAlign = textAlign;
      ctx.fillText(label, x, y + offsetY);

      // 점수 표시
      ctx.font = '10px sans-serif';
      ctx.fillStyle = '#6b7280';
      ctx.fillText(`${Math.round(values[i])}`, x, y + offsetY + 15);
    });
  };

  return (
    <div className={`inline-block ${className}`}>
      <canvas
        ref={canvasRef}
        width={size}
        height={size}
        className="max-w-full h-auto"
      />
    </div>
  );
}