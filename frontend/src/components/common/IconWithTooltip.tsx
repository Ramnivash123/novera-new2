import React from 'react';
import type { LucideIcon } from 'lucide-react';

interface IconWithTooltipProps {
  icon: LucideIcon;
  className?: string;
  title?: string;
  'aria-label'?: string;
}

export const IconWithTooltip: React.FC<IconWithTooltipProps> = ({
  icon: Icon,
  className = '',
  title,
  'aria-label': ariaLabel,
}) => {
  return (
    <span title={title} aria-label={ariaLabel || title}>
      <Icon className={className} />
    </span>
  );
};