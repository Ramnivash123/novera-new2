import { FileText, Clock, X, Sparkles, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';

interface ContextIndicatorProps {
  contextSummary?: {
    primary_document?: string;
    active_documents?: string[];
    recent_time_period?: string;
    message_count?: number;
  };
  onClearContext?: () => void;
}

export default function ContextIndicator({ contextSummary, onClearContext }: ContextIndicatorProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!contextSummary || (!contextSummary.primary_document && !contextSummary.recent_time_period)) {
    return null;
  }

  return (
    <div className="mb-3 sm:mb-4 px-3 sm:px-0">
      <div className="max-w-4xl mx-auto">
        {/* Clean white card — no heavy gradient */}
        <div className="bg-white dark:bg-gray-800 border border-blue-200 dark:border-blue-800 rounded-lg p-2.5 sm:p-3 shadow-sm">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-start gap-2 sm:gap-3 flex-1 min-w-0">

              {/* Icon */}
              <div className="flex items-center justify-center w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-blue-50 dark:bg-blue-900/50 flex-shrink-0">
                <Sparkles className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-blue-600 dark:text-blue-400" />
              </div>

              <div className="flex-1 min-w-0">
                {/* Pill tags */}
                <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">

                  {contextSummary.primary_document && (
                    <div className="flex items-center gap-1 sm:gap-1.5 px-2 py-1 bg-blue-50 dark:bg-blue-900/40 rounded-md border border-blue-200 dark:border-blue-700 shadow-sm max-w-full">
                      <FileText className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
                      <span className="text-xs font-medium text-blue-800 dark:text-blue-200 truncate">
                        {contextSummary.primary_document}
                      </span>
                    </div>
                  )}

                  {contextSummary.recent_time_period && (
                    <div className="flex items-center gap-1 sm:gap-1.5 px-2 py-1 bg-purple-50 dark:bg-purple-900/40 rounded-md border border-purple-200 dark:border-purple-700 shadow-sm">
                      <Clock className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-purple-600 dark:text-purple-400 flex-shrink-0" />
                      <span className="text-xs font-medium text-purple-800 dark:text-purple-200 whitespace-nowrap">
                        {contextSummary.recent_time_period}
                      </span>
                    </div>
                  )}

                  {contextSummary.active_documents && contextSummary.active_documents.length > 1 && (
                    <button
                      onClick={() => setIsExpanded(!isExpanded)}
                      className="flex items-center gap-1 px-2 py-1 text-xs font-medium text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-md transition-colors min-touch-target"
                    >
                      <span>+{contextSummary.active_documents.length - 1} more</span>
                      {isExpanded
                        ? <ChevronUp className="w-3 h-3" />
                        : <ChevronDown className="w-3 h-3" />
                      }
                    </button>
                  )}
                </div>

                {/* Subtitle */}
                <p className="text-xs text-blue-600 dark:text-blue-400 mt-1 hidden sm:block">
                  Focusing on this context for your questions
                </p>
              </div>
            </div>

            {/* Clear context button */}
            {onClearContext && (
              <button
                onClick={onClearContext}
                className="p-1.5 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors flex-shrink-0 min-touch-target"
                title="Clear context"
                aria-label="Clear context"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Expanded document list */}
          {isExpanded && contextSummary.active_documents && contextSummary.active_documents.length > 1 && (
            <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700 animate-slideUp">
              <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">Active Documents:</p>
              <div className="space-y-1.5">
                {contextSummary.active_documents.map((doc, idx) => (
                  <div
                    key={idx}
                    className="flex items-center gap-2 px-2 sm:px-2.5 py-1.5 bg-gray-50 dark:bg-gray-700 rounded border border-gray-200 dark:border-gray-600 text-xs text-gray-700 dark:text-gray-300"
                  >
                    <FileText className="w-3 h-3 text-gray-400 dark:text-gray-500 flex-shrink-0" />
                    <span className="truncate flex-1">{doc}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}