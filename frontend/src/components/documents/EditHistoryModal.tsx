import { useState, useEffect } from 'react';
import { X, Clock, User, Loader2, ChevronDown, ChevronUp } from 'lucide-react';
import api, { EditHistoryItem } from '../../services/api';

interface EditHistoryModalProps {
  chunkId: string;
  onClose: () => void;
}

export default function EditHistoryModal({ chunkId, onClose }: EditHistoryModalProps) {
  const [history, setHistory] = useState<EditHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    loadHistory();
  }, [chunkId]);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const data = await api.getChunkHistory(chunkId, 20);
      setHistory(data);
    } catch (error) {
      console.error('Failed to load edit history:', error);
    } finally {
      setLoading(false);
    }
  };

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-black/50 backdrop-blur-sm">
      <div className="flex items-center justify-center min-h-screen p-2 sm:p-4 md:p-8">
        <div className="relative bg-white rounded-lg sm:rounded-xl shadow-xl w-full max-w-4xl max-h-[95vh] sm:max-h-[90vh] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between px-4 sm:px-6 py-3 sm:py-4 border-b border-gray-200 flex-shrink-0">
            <h3 className="text-base sm:text-lg font-semibold text-gray-900">Edit History</h3>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors p-2 -mr-2 min-touch-target"
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-3 sm:p-6 scroll-smooth-touch">
            {loading ? (
              <div className="flex items-center justify-center h-64">
                <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
              </div>
            ) : history.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-64 text-center px-4">
                <Clock className="w-12 h-12 sm:w-16 sm:h-16 text-gray-300 mb-4" />
                <p className="text-sm sm:text-base text-gray-600">No edit history available</p>
              </div>
            ) : (
              <div className="space-y-3 sm:space-y-4">
                {history.map((item) => (
                  <div
                    key={item.id}
                    className="bg-white border border-gray-200 rounded-lg overflow-hidden"
                  >
                    {/* History Item Header */}
                    <div className="p-3 sm:p-4 bg-gray-50 border-b border-gray-200">
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex items-start gap-2 sm:gap-3 flex-1 min-w-0">
                          <User className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-400 flex-shrink-0 mt-0.5" />
                          <div className="min-w-0 flex-1">
                            <p className="text-xs sm:text-sm font-medium text-gray-900 truncate">
                              Editor ID: {item.edited_by}
                            </p>
                            <div className="flex items-center gap-1.5 sm:gap-2 text-xs text-gray-500 mt-0.5 sm:mt-1">
                              <Clock className="w-3 h-3 flex-shrink-0" />
                              <span className="truncate">{new Date(item.edited_at).toLocaleString()}</span>
                            </div>
                          </div>
                        </div>
                        {item.change_summary && (
                          <span className="text-xs sm:text-sm text-gray-600 italic flex-shrink-0 hidden sm:block">
                            {item.change_summary}
                          </span>
                        )}
                      </div>
                    </div>

                    {/* Content Comparison */}
                    <div className="p-3 sm:p-4">
                      <button
                        onClick={() => toggleExpand(item.id)}
                        className="flex items-center justify-between w-full text-xs sm:text-sm font-medium text-gray-700 hover:text-gray-900 transition-colors min-touch-target"
                      >
                        <span>View Changes</span>
                        {expandedId === item.id ? (
                          <ChevronUp className="w-4 h-4" />
                        ) : (
                          <ChevronDown className="w-4 h-4" />
                        )}
                      </button>

                      {expandedId === item.id && (
                        <div className="mt-3 sm:mt-4 space-y-3 sm:space-y-0 sm:grid sm:grid-cols-2 sm:gap-4">
                          {/* Old Content */}
                          <div>
                            <h4 className="text-xs font-semibold text-red-600 mb-2">
                              OLD CONTENT
                            </h4>
                            <div className="bg-red-50 border border-red-200 rounded-lg p-2.5 sm:p-3 max-h-48 sm:max-h-64 overflow-y-auto scroll-smooth-touch">
                              <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono break-words">
                                {item.old_content}
                              </pre>
                            </div>
                          </div>

                          {/* New Content */}
                          <div>
                            <h4 className="text-xs font-semibold text-green-600 mb-2">
                              NEW CONTENT
                            </h4>
                            <div className="bg-green-50 border border-green-200 rounded-lg p-2.5 sm:p-3 max-h-48 sm:max-h-64 overflow-y-auto scroll-smooth-touch">
                              <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono break-words">
                                {item.new_content}
                              </pre>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-4 sm:px-6 py-3 sm:py-4 bg-gray-50 border-t border-gray-200 flex items-center justify-end flex-shrink-0">
            <button
              onClick={onClose}
              className="w-full sm:w-auto px-4 py-2.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors min-touch-target"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}