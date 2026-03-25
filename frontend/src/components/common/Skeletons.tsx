// Reusable skeleton components for consistent loading states

interface SkeletonProps {
  className?: string;
}

// Base skeleton with shimmer animation
export function Skeleton({ className = '' }: SkeletonProps) {
  return (
    <div className={`animate-pulse bg-gray-200 rounded ${className}`} />
  );
}

// Circular skeleton (for avatars)
export function SkeletonCircle({ className = 'w-10 h-10' }: SkeletonProps) {
  return (
    <div className={`animate-pulse bg-gray-200 rounded-full ${className}`} />
  );
}

// Skeleton with shimmer effect (enhanced version)
export function SkeletonShimmer({ className = '' }: SkeletonProps) {
  return (
    <div className={`relative overflow-hidden bg-gray-200 rounded ${className}`}>
      <div className="absolute inset-0 -translate-x-full animate-shimmer bg-gradient-to-r from-gray-200 via-gray-100 to-gray-200" />
    </div>
  );
}

// Message Bubble Skeleton
export function MessageBubbleSkeleton() {
  return (
    <div className="flex items-start gap-2 sm:gap-4 w-full">
      {/* Avatar */}
      <SkeletonCircle className="w-7 h-7 sm:w-9 sm:h-9 flex-shrink-0" />
      
      {/* Message Content */}
      <div className="flex-1 space-y-2">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-5/6" />
      </div>
    </div>
  );
}

// Document Card Skeleton
export function DocumentCardSkeleton() {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-3 sm:p-4">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-2 flex-1">
          <SkeletonCircle className="w-4 h-4 sm:w-5 sm:h-5" />
          <Skeleton className="h-5 flex-1" />
        </div>
      </div>
      <Skeleton className="h-6 w-20 mb-3" />
      <div className="space-y-2">
        <div className="flex justify-between">
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-24" />
        </div>
        <div className="flex justify-between">
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-20" />
        </div>
        <div className="flex justify-between">
          <Skeleton className="h-4 w-16" />
          <Skeleton className="h-4 w-16" />
        </div>
      </div>
      <div className="mt-4 pt-4 border-t border-gray-200 flex justify-between">
        <Skeleton className="h-8 w-16" />
        <Skeleton className="h-8 w-16" />
      </div>
    </div>
  );
}

// Conversation Card Skeleton
export function ConversationCardSkeleton() {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-3 sm:p-4">
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 space-y-2">
          <div className="flex items-center gap-2">
            <SkeletonCircle className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
            <Skeleton className="h-3 w-32" />
          </div>
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-6 w-24" />
        </div>
        <SkeletonCircle className="w-4 h-4 flex-shrink-0" />
      </div>
    </div>
  );
}

// Chunk Card Skeleton
export function ChunkCardSkeleton() {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3 sm:p-4">
      <div className="flex items-start justify-between gap-2 mb-2 sm:mb-3">
        <div className="flex-1 space-y-2">
          <div className="flex items-center gap-2">
            <Skeleton className="h-5 w-32" />
            <Skeleton className="h-5 w-20" />
          </div>
          <Skeleton className="h-3 w-full" />
        </div>
        <Skeleton className="h-8 w-16" />
      </div>
      <div className="bg-gray-50 rounded-lg p-2.5 sm:p-3 border border-gray-200">
        <Skeleton className="h-3 w-full mb-1" />
        <Skeleton className="h-3 w-full mb-1" />
        <Skeleton className="h-3 w-2/3" />
      </div>
      <div className="mt-2 sm:mt-3 flex justify-between">
        <Skeleton className="h-3 w-20" />
        <Skeleton className="h-3 w-24" />
      </div>
    </div>
  );
}

// User Table Row Skeleton (Desktop)
export function UserTableRowSkeleton() {
  return (
    <tr className="hover:bg-gray-50">
      <td className="px-6 py-4 whitespace-nowrap">
        <div className="flex items-center">
          <SkeletonCircle className="h-10 w-10" />
          <div className="ml-4 space-y-1">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-40" />
          </div>
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <Skeleton className="h-6 w-16" />
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <Skeleton className="h-6 w-20" />
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <Skeleton className="h-4 w-12" />
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <Skeleton className="h-4 w-24" />
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-right">
        <div className="flex justify-end gap-2">
          <SkeletonCircle className="w-8 h-8" />
          <SkeletonCircle className="w-8 h-8" />
          <SkeletonCircle className="w-8 h-8" />
        </div>
      </td>
    </tr>
  );
}

// User Card Skeleton (Mobile)
export function UserCardSkeleton() {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4">
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <SkeletonCircle className="h-12 w-12" />
          <div className="space-y-1">
            <Skeleton className="h-4 w-32" />
            <Skeleton className="h-3 w-40" />
          </div>
        </div>
      </div>
      <div className="flex gap-2 mb-3">
        <Skeleton className="h-6 w-16" />
        <Skeleton className="h-6 w-20" />
      </div>
      <div className="grid grid-cols-2 gap-2 mb-3">
        <Skeleton className="h-4 w-full" />
        <Skeleton className="h-4 w-full" />
      </div>
      <div className="flex gap-2">
        <Skeleton className="h-10 flex-1" />
        <Skeleton className="h-10 flex-1" />
        <Skeleton className="h-10 w-10" />
      </div>
    </div>
  );
}

// Stats Card Skeleton
export function StatsCardSkeleton() {
  return (
    <div className="bg-gray-50 rounded-lg p-3 sm:p-4 border border-gray-200">
      <div className="flex items-center justify-between mb-3 sm:mb-4">
        <SkeletonCircle className="w-10 h-10 sm:w-12 sm:h-12" />
      </div>
      <Skeleton className="h-8 w-20 mb-1" />
      <Skeleton className="h-4 w-24" />
    </div>
  );
}

// Source Card Skeleton
export function SourceCardSkeleton() {
  return (
    <div className="bg-white rounded-lg p-2.5 sm:p-3 border border-gray-200">
      <div className="flex items-start gap-2">
        <SkeletonCircle className="w-3.5 h-3.5 sm:w-4 sm:h-4 mt-0.5" />
        <div className="flex-1 space-y-1">
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-3 w-3/4" />
        </div>
        <SkeletonCircle className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
      </div>
    </div>
  );
}

// Table Skeleton (Generic)
export function TableSkeleton({ rows = 5 }: { rows?: number }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <th key={i} className="px-6 py-3">
                  <Skeleton className="h-4 w-full" />
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {Array.from({ length: rows }).map((_, i) => (
              <UserTableRowSkeleton key={i} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Full Page Skeleton
export function FullPageSkeleton() {
  return (
    <div className="h-full overflow-y-auto bg-gray-50 p-4 sm:p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="space-y-2">
          <Skeleton className="h-8 w-64" />
          <Skeleton className="h-4 w-96" />
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <StatsCardSkeleton key={i} />
          ))}
        </div>

        {/* Content */}
        <div className="space-y-4">
          <Skeleton className="h-6 w-48" />
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <DocumentCardSkeleton key={i} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
  
}