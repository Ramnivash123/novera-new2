import { useState, useEffect } from 'react';
import { X, Download, FileText, Loader2, AlertCircle } from 'lucide-react';
import api, { DocumentInfo } from '../../services/api';

interface DocumentViewerProps {
  documentId: string;
  onClose: () => void;
}

export default function DocumentViewer({ documentId, onClose }: DocumentViewerProps) {
  const [docInfo, setDocInfo] = useState<DocumentInfo | null>(null);
  const [preview, setPreview] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadDocument();
  }, [documentId]);

  const loadDocument = async () => {
    setLoading(true);
    setError(null);

    try {
      const info = await api.getDocumentInfo(documentId);
      setDocInfo(info);

      if (info.preview_type === 'text') {
        const textPreview = await api.getTextPreview(documentId);
        setPreview(textPreview);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load document');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = async () => {
    try {
      const blob = await api.downloadDocument(documentId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = docInfo?.original_filename || 'document';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err) {
      console.error('Download failed:', err);
      alert('Failed to download document');
    }
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto bg-black/50 backdrop-blur-sm">
      <div className="flex items-center justify-center min-h-screen p-2 sm:p-4 md:p-8">
        <div className="relative bg-white rounded-lg sm:rounded-xl shadow-xl w-full max-w-6xl max-h-[95vh] sm:max-h-[90vh] flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between px-4 sm:px-6 py-3 sm:py-4 border-b border-gray-200 flex-shrink-0">
            <div className="flex-1 min-w-0 pr-2">
              <h3 className="text-base sm:text-lg font-semibold text-gray-900 truncate">
                {docInfo?.original_filename || 'Document Preview'}
              </h3>
              {docInfo && (
                <p className="text-xs sm:text-sm text-gray-600 mt-0.5 sm:mt-1 truncate">
                  {docInfo.doc_type} | {docInfo.file_size_mb} MB | {docInfo.total_pages} pages | {docInfo.total_chunks} chunks
                </p>
              )}
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              {docInfo && (
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-2 text-xs sm:text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors min-touch-target"
                >
                  <Download className="w-4 h-4" />
                  <span className="hidden xs:inline">Download</span>
                </button>
              )}
              <button
                onClick={onClose}
                className="text-gray-400 hover:text-gray-600 transition-colors p-2 -mr-2 min-touch-target"
                aria-label="Close"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-3 sm:p-6 scroll-smooth-touch">
            {loading ? (
              <div className="flex items-center justify-center h-64">
                <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
              </div>
            ) : error ? (
              <div className="flex flex-col items-center justify-center h-64 text-center px-4">
                <AlertCircle className="w-12 h-12 sm:w-16 sm:h-16 text-red-400 mb-4" />
                <p className="text-sm sm:text-base text-red-600">{error}</p>
              </div>
            ) : docInfo?.preview_type === 'text' && preview ? (
              <div className="bg-gray-50 rounded-lg p-3 sm:p-6 border border-gray-200">
                <pre className="text-xs sm:text-sm text-gray-800 whitespace-pre-wrap font-mono break-words">
                  {preview.content}
                </pre>
                {preview.is_truncated && (
                  <p className="mt-4 text-xs sm:text-sm text-gray-600 italic">
                    Preview truncated. Download the full document to see all content.
                  </p>
                )}
              </div>
            ) : docInfo?.preview_type === 'pdf' ? (
              <div className="flex flex-col items-center justify-center h-64 text-center px-4">
                <FileText className="w-12 h-12 sm:w-16 sm:h-16 text-gray-300 mb-4" />
                <p className="text-sm sm:text-base text-gray-700 mb-4">PDF Preview</p>
                <p className="text-xs sm:text-sm text-gray-600 mb-6">
                  PDF preview not available. Download the document to view it.
                </p>
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-2 px-4 py-2.5 text-white bg-primary-600 rounded-lg hover:bg-primary-700 transition-colors text-sm sm:text-base min-touch-target"
                >
                  <Download className="w-4 h-4" />
                  Download PDF
                </button>
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-64 text-center px-4">
                <FileText className="w-12 h-12 sm:w-16 sm:h-16 text-gray-300 mb-4" />
                <p className="text-sm sm:text-base text-gray-700 mb-4">Preview not available for this file type</p>
                <button
                  onClick={handleDownload}
                  className="flex items-center gap-2 px-4 py-2.5 text-white bg-primary-600 rounded-lg hover:bg-primary-700 transition-colors text-sm sm:text-base min-touch-target"
                >
                  <Download className="w-4 h-4" />
                  Download Document
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}