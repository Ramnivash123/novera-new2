import { FileText, ExternalLink } from 'lucide-react';
import { Source } from '../../services/api';

interface SourceCardProps {
  source: Source;
}

export default function SourceCard({ source }: SourceCardProps) {
  return (
    <div className="bg-white rounded-lg p-2.5 sm:p-3 border border-gray-200 hover:border-primary-300 transition-colors">
      <div className="flex items-start gap-2">
        <FileText className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-400 mt-0.5 flex-shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-xs sm:text-sm font-medium text-gray-900 truncate" title={source.document}>
            {source.document}
          </p>
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-1">
            {source.page && (
              <span className="text-xs text-gray-500">Page {source.page}</span>
            )}
            {source.section && (
              <>
                <span className="text-xs text-gray-300 hidden xs:inline">•</span>
                <span className="text-xs text-gray-500 truncate max-w-[150px]" title={source.section}>
                  {source.section}
                </span>
              </>
            )}
          </div>
        </div>
        <button
          className="text-gray-400 hover:text-primary-600 transition-colors p-1 flex-shrink-0 min-touch-target"
          title="View source"
          aria-label="View source details"
        >
          <ExternalLink className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
        </button>
      </div>
    </div>
  );
}