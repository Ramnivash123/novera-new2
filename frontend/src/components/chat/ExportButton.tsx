import { useState } from 'react';
import { Download, FileText, FileJson, FileType, Loader2, CheckSquare } from 'lucide-react';
import { toast } from '../../utils/toast';

interface ExportButtonProps {
  conversationId: string;
  messages?: Array<{ role: string; content: string; timestamp?: string }>;
  onSelectiveExport?: () => void;
}

export default function ExportButton({ conversationId, messages, onSelectiveExport }: ExportButtonProps) {
  const [isExporting, setIsExporting] = useState(false);
  const [showMenu, setShowMenu] = useState(false);

  const handleLocalExport = (format: 'markdown' | 'json') => {
    if (!messages || messages.length === 0) {
      toast.error('No messages to export');
      return;
    }

    let content: string;
    let mimeType: string;
    let extension: string;

    if (format === 'markdown') {
      content = messages
        .map((m) => {
          const label = m.role === 'user' ? '**You**' : '**Novera**';
          const time = m.timestamp
            ? new Date(m.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
            : '';
          return `${label}${time ? ` *(${time})*` : ''}\n\n${m.content}`;
        })
        .join('\n\n---\n\n');
      mimeType = 'text/markdown';
      extension = 'md';
    } else {
      content = JSON.stringify(
        {
          exported_at: new Date().toISOString(),
          messages: messages.map((m) => ({
            role: m.role,
            content: m.content,
            timestamp: m.timestamp,
          })),
        },
        null,
        2
      );
      mimeType = 'application/json';
      extension = 'json';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation_local.${extension}`;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    toast.success(`Exported as ${format.toUpperCase()}`);
  };

  const handleExport = async (format: 'markdown' | 'json' | 'pdf') => {
    setIsExporting(true);
    setShowMenu(false);

    // No persisted conversation yet — export locally from in-memory messages
    if (conversationId === 'local') {
      if (format !== 'pdf') {
        handleLocalExport(format);
      } else {
        // PDF requires a backend conversation — fall back to markdown
        toast.info('PDF export requires a saved conversation. Exporting as Markdown instead.');
        handleLocalExport('markdown');
      }
      setIsExporting(false);
      return;
    }

    try {
      const token = localStorage.getItem('access_token');

      const response = await fetch(
        `/api/v1/chat/conversations/${conversationId}/export?format=${format}`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) {
        // PDF may fail if reportlab not installed — fall back to markdown
        if (format === 'pdf' && response.status === 500) {
          toast.info('PDF generation unavailable. Downloading as Markdown instead.');
          const mdResponse = await fetch(
            `/api/v1/chat/conversations/${conversationId}/export?format=markdown`,
            { headers: { Authorization: `Bearer ${token}` } }
          );
          if (!mdResponse.ok) throw new Error('Export failed');
          const blob = await mdResponse.blob();
          const url = window.URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `conversation_${conversationId.slice(0, 8)}.md`;
          document.body.appendChild(a);
          a.click();
          window.URL.revokeObjectURL(url);
          document.body.removeChild(a);
          return;
        }
        throw new Error('Export failed');
      }

      const contentDisposition = response.headers.get('Content-Disposition');
      const filenameMatch = contentDisposition?.match(/filename="?([^"]+)"?/);
      const filename =
        filenameMatch?.[1] || `conversation_${conversationId.slice(0, 8)}.${format === 'pdf' ? 'pdf' : format === 'json' ? 'json' : 'md'}`;

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      toast.success(`Exported as ${format.toUpperCase()}`);
    } catch (error) {
      console.error('Export error:', error);
      toast.error('Export failed. Please try again.');
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="relative">
      <button
        onClick={() => setShowMenu(!showMenu)}
        disabled={isExporting}
        className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 border border-gray-300 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-touch-target"
        title="Export conversation"
      >
        {isExporting ? (
          <Loader2 className="w-4 h-4 animate-spin" />
        ) : (
          <Download className="w-4 h-4" />
        )}
        <span className="hidden sm:inline">Export</span>
      </button>

      {showMenu && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setShowMenu(false)} />

          <div className="absolute right-0 mt-2 w-56 bg-white rounded-lg shadow-lg border border-gray-200 py-1 z-20">
            {onSelectiveExport && conversationId !== 'local' && (
              <>
                <button
                  onClick={() => {
                    setShowMenu(false);
                    onSelectiveExport();
                  }}
                  disabled={isExporting}
                  className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-gray-700 hover:bg-purple-50 transition-colors disabled:opacity-50"
                >
                  <CheckSquare className="w-4 h-4 text-purple-600" />
                  <div className="text-left">
                    <p className="font-medium">Select Messages</p>
                    <p className="text-xs text-gray-500">Choose specific messages</p>
                  </div>
                </button>
                <div className="border-t border-gray-200 my-1">
                  <div className="px-4 py-2">
                    <p className="text-xs font-medium text-gray-500 uppercase">Export All</p>
                  </div>
                </div>
              </>
            )}

            <button
              onClick={() => handleExport('pdf')}
              disabled={isExporting}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-gray-700 hover:bg-red-50 transition-colors disabled:opacity-50"
            >
              <FileType className="w-4 h-4 text-red-600" />
              <div className="text-left">
                <p className="font-medium">Export as PDF</p>
                {conversationId === 'local' && (
                  <p className="text-xs text-gray-400">Falls back to Markdown</p>
                )}
              </div>
            </button>

            <div className="border-t border-gray-100 my-1" />

            <button
              onClick={() => handleExport('markdown')}
              disabled={isExporting}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-gray-700 hover:bg-blue-50 transition-colors disabled:opacity-50"
            >
              <FileText className="w-4 h-4 text-blue-600" />
              <span className="font-medium">Export as Markdown</span>
            </button>

            <div className="border-t border-gray-100 my-1" />

            <button
              onClick={() => handleExport('json')}
              disabled={isExporting}
              className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-gray-700 hover:bg-green-50 transition-colors disabled:opacity-50"
            >
              <FileJson className="w-4 h-4 text-green-600" />
              <span className="font-medium">Export as JSON</span>
            </button>
          </div>
        </>
      )}
    </div>
  );
}