'use client';

import { ReactNode, ButtonHTMLAttributes } from 'react';
import { designTokens } from '@/styles/design-tokens';
import { Loader2 } from 'lucide-react';

interface EnhancedButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg' | 'xl';
  isLoading?: boolean;
  icon?: ReactNode;
  iconPosition?: 'left' | 'right';
  fullWidth?: boolean;
  className?: string;
}

const variantClasses = {
  primary: 'bg-blue-600 hover:bg-blue-700 active:bg-blue-800 text-white shadow-sm border border-blue-600',
  secondary: 'bg-gray-100 hover:bg-gray-200 active:bg-gray-300 text-gray-900 shadow-sm border border-gray-200',
  outline: 'bg-transparent hover:bg-blue-50 active:bg-blue-100 text-blue-600 border border-blue-600',
  ghost: 'bg-transparent hover:bg-gray-50 active:bg-gray-100 text-gray-700 border border-transparent',
  danger: 'bg-red-600 hover:bg-red-700 active:bg-red-800 text-white shadow-sm border border-red-600'
};

const sizeClasses = {
  sm: 'px-3 py-2 text-sm font-medium rounded-lg',
  md: 'px-4 py-2.5 text-sm font-medium rounded-lg',
  lg: 'px-6 py-3 text-base font-semibold rounded-xl',
  xl: 'px-8 py-4 text-lg font-semibold rounded-xl'
};

export function EnhancedButton({
  children,
  variant = 'primary',
  size = 'md',
  isLoading = false,
  icon,
  iconPosition = 'left',
  fullWidth = false,
  className = '',
  disabled,
  ...props
}: EnhancedButtonProps) {
  const baseClasses = 'inline-flex items-center justify-center gap-2 font-medium transition-all duration-200 ease-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:pointer-events-none';

  const classes = [
    baseClasses,
    variantClasses[variant],
    sizeClasses[size],
    fullWidth ? 'w-full' : '',
    className
  ].join(' ').trim();

  const IconComponent = isLoading ? Loader2 : icon;

  return (
    <button
      className={classes}
      disabled={disabled || isLoading}
      {...props}
    >
      {iconPosition === 'left' && IconComponent && (
        <span className={`${isLoading ? 'animate-spin' : ''}`}>
          {IconComponent}
        </span>
      )}
      {children}
      {iconPosition === 'right' && IconComponent && (
        <span className={`${isLoading ? 'animate-spin' : ''}`}>
          {IconComponent}
        </span>
      )}
    </button>
  );
}