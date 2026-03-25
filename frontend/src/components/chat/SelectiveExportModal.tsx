import { useState, useMemo } from 'react';
import { X, Download, FileText, FileJson, FileType, Check, Square, CheckSquare } from 'lucide-react';
import { ChatMessage } from '../../services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { toast } from '../../utils/toast';

interface SelectiveExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  conversationId: string;
  messages: ChatMessage[];
}

export default function SelectiveExportModal({
  isOpen,
  onClose,
  conversationId,
  messages
}: SelectiveExportModalProps) {
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set());
  const [exportFormat, setExportFormat] = useState<'pdf' | 'markdown' | 'json'>('markdown');
  const [isExporting, setIsExporting] = useState(false);

  const handleSelectAll = () => {
    if (selectedIndices.size === messages.length) {
      setSelectedIndices(new Set());
    } else {
      setSelectedIndices(new Set(messages.map((_, idx) => idx)));
    }
  };

  const toggleMessage = (index: number) => {
    const newSelected = new Set(selectedIndices);
    if (newSelected.has(index)) {
      newSelected.delete(index);
    } else {
      newSelected.add(index);
    }
    setSelectedIndices(newSelected);
  };

  const selectedMessages = useMemo(() => {
    return Array.from(selectedIndices)
      .sort((a, b) => a - b)
      .map(idx => messages[idx]);
  }, [selectedIndices, messages]);

  const handleExport = async () => {
    if (selectedMessages.length === 0) {
      toast.error('Please select at least one message to export');
      return;
    }

    setIsExporting(true);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const token = localStorage.getItem('access_token');

      const response = await fetch(
        `${apiUrl}/api/v1/chat/conversations/${conversationId}/export-selected`,
        {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            message_indices: Array.from(selectedIndices).sort((a, b) => a - b),
            format: exportFormat
          })
        }
      );

      if (!response.ok) {
        throw new Error('Export failed');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `conversation_selective_${conversationId.slice(0, 8)}.${exportFormat}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast.success(`Exported ${selectedMessages.length} messages as ${exportFormat.toUpperCase()}`);
      onClose();
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Failed to export. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-2 sm:p-4">
      <div className="bg-white rounded-lg sm:rounded-xl shadow-2xl max-w-4xl w-full max-h-[95vh] sm:max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="px-4 sm:px-6 py-3 sm:py-4 border-b border-gray-200 flex items-center justify-between flex-shrink-0">
          <div>
            <h2 className="text-base sm:text-xl font-bold text-gray-900">Select Messages to Export</h2>
            <p className="text-xs sm:text-sm text-gray-500 mt-1">
              {selectedIndices.size} of {messages.length} messages selected
            </p>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors flex-shrink-0 min-touch-target"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Selection Controls */}
        <div className="px-4 sm:px-6 py-2.5 sm:py-3 bg-gray-50 border-b border-gray-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 flex-shrink-0">
          <button
            onClick={handleSelectAll}
            className="flex items-center justify-center sm:justify-start gap-2 px-3 py-2 text-xs sm:text-sm font-medium text-blue-600 hover:bg-blue-50 rounded-lg transition-colors min-touch-target"
          >
            {selectedIndices.size === messages.length ? (
              <>
                <CheckSquare className="w-4 h-4" />
                Deselect All
              </>
            ) : (
              <>
                <Square className="w-4 h-4" />
                Select All
              </>
            )}
          </button>

          {/* Format Selector */}
          <div className="flex items-center gap-2">
            <span className="text-xs sm:text-sm text-gray-600 font-medium">Format:</span>
            <div className="flex gap-1.5 sm:gap-2 flex-1 sm:flex-none">
              <button
                onClick={() => setExportFormat('pdf')}
                className={`flex-1 sm:flex-none px-2.5 sm:px-3 py-1.5 text-xs sm:text-sm font-medium rounded-lg transition-colors min-touch-target ${
                  exportFormat === 'pdf'
                    ? 'bg-red-100 text-red-700 border border-red-300'
                    : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                }`}
              >
                <FileType className="w-3.5 h-3.5 sm:w-4 sm:h-4 inline mr-1" />
                PDF
              </button>
              <button
                onClick={() => setExportFormat('markdown')}
                className={`flex-1 sm:flex-none px-2.5 sm:px-3 py-1.5 text-xs sm:text-sm font-medium rounded-lg transition-colors min-touch-target ${
                  exportFormat === 'markdown'
                    ? 'bg-blue-100 text-blue-700 border border-blue-300'
                    : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                }`}
              >
                <FileText className="w-3.5 h-3.5 sm:w-4 sm:h-4 inline mr-1" />
                MD
              </button>
              <button
                onClick={() => setExportFormat('json')}
                className={`flex-1 sm:flex-none px-2.5 sm:px-3 py-1.5 text-xs sm:text-sm font-medium rounded-lg transition-colors min-touch-target ${
                  exportFormat === 'json'
                    ? 'bg-green-100 text-green-700 border border-green-300'
                    : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
                }`}
              >
                <FileJson className="w-3.5 h-3.5 sm:w-4 sm:h-4 inline mr-1" />
                JSON
              </button>
            </div>
          </div>
        </div>

        {/* Messages List */}
        <div className="flex-1 overflow-y-auto p-3 sm:p-6 space-y-2 sm:space-y-3 scroll-smooth-touch">
          {messages.map((message, index) => {
            const isSelected = selectedIndices.has(index);
            const isUser = message.role === 'user';

            return (
              <div
                key={index}
                onClick={() => toggleMessage(index)}
                className={`relative cursor-pointer rounded-lg border-2 transition-all ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 shadow-md'
                    : 'border-gray-200 bg-white hover:border-blue-300 hover:shadow-sm'
                }`}
              >
                {/* Checkbox */}
                <div className="absolute top-2.5 sm:top-3 left-2.5 sm:left-3">
                  <div
                    className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-colors ${
                      isSelected
                        ? 'bg-blue-600 border-blue-600'
                        : 'bg-white border-gray-300'
                    }`}
                  >
                    {isSelected && <Check className="w-3 h-3 text-white" />}
                  </div>
                </div>

                {/* Message Content */}
                <div className="pl-10 sm:pl-12 pr-3 sm:pr-4 py-2.5 sm:py-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className={`text-xs sm:text-sm font-semibold ${
                        isUser ? 'text-green-700' : 'text-purple-700'
                      }`}
                    >
                      {isUser ? 'User' : 'AI Assistant'}
                    </span>
                    {message.timestamp && (
                      <span className="text-xs text-gray-500">
                        {new Date(message.timestamp).toLocaleTimeString()}
                      </span>
                    )}
                  </div>

                  {/* Content Preview */}
                  <div className="text-xs sm:text-sm text-gray-700">
                    {isUser ? (
                      <p className="line-clamp-3 whitespace-pre-wrap break-words">{message.content}</p>
                    ) : (
                      <div className="line-clamp-3">
                        {message.content.length > 200 ? (
                          <p className="whitespace-pre-wrap break-words">
                            {message.content.substring(0, 200)}...
                          </p>
                        ) : (
                          <div className="prose prose-sm max-w-none">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>
                              {message.content}
                            </ReactMarkdown>
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Sources indicator */}
                  {!isUser && message.metadata?.sources && message.metadata.sources.length > 0 && (
                    <div className="mt-2 text-xs text-gray-500">
                      📚 {message.metadata.sources.length} source(s)
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div className="px-4 sm:px-6 py-3 sm:py-4 border-t border-gray-200 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 bg-gray-50 flex-shrink-0">
          <p className="text-xs sm:text-sm text-gray-600 text-center sm:text-left">
            {selectedMessages.length > 0 ? (
              <>
                Ready to export <span className="font-semibold">{selectedMessages.length}</span> message
                {selectedMessages.length !== 1 ? 's' : ''}
              </>
            ) : (
              'Select messages to export'
            )}
          </p>

          <div className="flex flex-col-reverse sm:flex-row gap-2 sm:gap-3">
            <button
              onClick={onClose}
              disabled={isExporting}
              className="w-full sm:w-auto px-4 py-2.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors disabled:opacity-50 min-touch-target"
            >
              Cancel
            </button>
            <button
              onClick={handleExport}
              disabled={selectedMessages.length === 0 || isExporting}
              className="w-full sm:w-auto px-6 py-2.5 text-sm font-medium text-white bg-blue-600 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 min-touch-target"
            >
              {isExporting ? (
                <>
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  Export {exportFormat.toUpperCase()}
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}