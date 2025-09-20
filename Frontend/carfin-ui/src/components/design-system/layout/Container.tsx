'use client';

import { ReactNode } from 'react';
import { designTokens } from '@/styles/design-tokens';

interface ContainerProps {
  children: ReactNode;
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  className?: string;
}

const sizeClasses = {
  sm: 'max-w-2xl',   // 672px
  md: 'max-w-4xl',   // 896px
  lg: 'max-w-6xl',   // 1152px
  xl: 'max-w-7xl',   // 1280px
  full: 'max-w-full'
};

export function Container({ children, size = 'lg', className = '' }: ContainerProps) {
  return (
    <div className={`mx-auto px-6 ${sizeClasses[size]} ${className}`}>
      {children}
    </div>
  );
}