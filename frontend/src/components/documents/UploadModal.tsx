import { useState, useCallback } from 'react';
import { X, Upload, File, Loader2, CheckCircle, AlertCircle, Plus } from 'lucide-react';
import api from '../../services/api';

interface UploadModalProps {
  onClose: () => void;
  onSuccess: () => void;
}

interface FileEntry {
  file: File;
  id: string;
  error?: string;
}

interface UploadResult {
  filename: string;
  status: string;
  message: string;
  document_id?: string;
  error?: string;
}

const ALLOWED_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.xlsx', '.xls'];
const MAX_SIZE_BYTES = 50 * 1024 * 1024;

function validateFile(file: File): string | null {
  if (file.size > MAX_SIZE_BYTES) return 'File exceeds 50MB limit';
  const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
  if (!ALLOWED_EXTENSIONS.includes(ext))
    return `${ext} not supported. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}`;
  return null;
}

export default function UploadModal({ onClose, onSuccess }: UploadModalProps) {
  const [files, setFiles] = useState<FileEntry[]>([]);
  const [docType, setDocType] = useState('finance');
  const [department, setDepartment] = useState('');
  const [uploading, setUploading] = useState(false);
  const [results, setResults] = useState<UploadResult[] | null>(null);
  const [globalError, setGlobalError] = useState<string | null>(null);

  const addFiles = useCallback((incoming: FileList | File[]) => {
    const arr = Array.from(incoming);
    const entries: FileEntry[] = arr.map((f) => ({
      file: f,
      id: `${f.name}-${f.size}-${Date.now()}-${Math.random()}`,
      error: validateFile(f) ?? undefined,
    }));
    setFiles((prev) => {
      const existingNames = new Set(prev.map((e) => e.file.name));
      const unique = entries.filter((e) => !existingNames.has(e.file.name));
      return [...prev, ...unique];
    });
    setGlobalError(null);
    setResults(null);
  }, []);

  const removeFile = (id: string) =>
    setFiles((prev) => prev.filter((e) => e.id !== id));

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.length) addFiles(e.target.files);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
  };

  const validFiles = files.filter((e) => !e.error);
  const hasErrors = files.some((e) => e.error);

  const handleUpload = async () => {
    if (validFiles.length === 0) return;

    setUploading(true);
    setGlobalError(null);
    setResults(null);

    try {
      const raw = validFiles.map((e) => e.file);

      let response: any;
      if (raw.length === 1) {
        const single = await api.uploadDocument(raw[0], docType, department || undefined);
        response = {
          total: 1,
          succeeded: single.status !== 'failed' ? 1 : 0,
          failed: single.status === 'failed' ? 1 : 0,
          duplicates: single.status === 'duplicate' ? 1 : 0,
          results: [single],
        };
      } else {
        response = await api.uploadDocuments(raw, docType, department || undefined);
      }

      setResults(response.results ?? []);

      const allDone =
        response.failed === 0 &&
        (response.succeeded > 0 || response.duplicates > 0);

      if (allDone) {
        setTimeout(() => onSuccess(), 2000);
      }
    } catch (err: any) {
      setGlobalError(err.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const statusIcon = (status: string) => {
    if (status === 'processing') return <CheckCircle className="w-4 h-4 text-green-500 flex-shrink-0" />;
    if (status === 'duplicate') return <AlertCircle className="w-4 h-4 text-yellow-500 flex-shrink-0" />;
    return <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />;
  };

  const statusColor = (status: string) => {
    if (status === 'processing') return 'text-green-700';
    if (status === 'duplicate') return 'text-yellow-700';
    return 'text-red-700';
  };

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen px-3 sm:px-4 pt-4 pb-20 text-center sm:p-0">
        <div className="fixed inset-0 bg-gray-500 bg-opacity-75 transition-opacity" onClick={onClose} />

        <div className="relative inline-block align-bottom bg-white rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg w-full mx-3 sm:mx-0">
          {/* Header */}
          <div className="flex items-center justify-between px-4 sm:px-6 py-3 sm:py-4 border-b border-gray-200">
            <h3 className="text-base sm:text-lg font-semibold text-gray-900">Upload Documents</h3>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-500 transition-colors p-2 -mr-2"
              disabled={uploading}
              aria-label="Close"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Body */}
          <div className="px-4 sm:px-6 py-4 space-y-4">
            {/* Drop Zone */}
            <div
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-4 sm:p-6 text-center transition-colors ${
                files.length > 0
                  ? 'border-primary-300 bg-primary-50'
                  : 'border-gray-300 hover:border-primary-400 bg-gray-50'
              }`}
            >
              <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <p className="text-sm text-gray-600 mb-2">
                Drag and drop files here, or
              </p>
              <label className="inline-flex items-center gap-1.5 px-3 py-2 bg-white border border-gray-300 rounded-lg text-xs sm:text-sm font-medium text-gray-700 hover:bg-gray-50 cursor-pointer transition-colors">
                <Plus className="w-3.5 h-3.5" />
                Add Files
                <input
                  type="file"
                  multiple
                  onChange={handleInputChange}
                  accept=".pdf,.docx,.doc,.txt,.xlsx,.xls"
                  className="sr-only"
                  disabled={uploading}
                />
              </label>
              <p className="text-xs text-gray-400 mt-2">
                PDF, DOCX, TXT, XLSX — max 50MB each — up to 20 files
              </p>
            </div>

            {/* File List */}
            {files.length > 0 && (
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {files.map((entry) => (
                  <div
                    key={entry.id}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg border text-sm ${
                      entry.error
                        ? 'border-red-200 bg-red-50'
                        : 'border-gray-200 bg-gray-50'
                    }`}
                  >
                    <File className={`w-4 h-4 flex-shrink-0 ${entry.error ? 'text-red-400' : 'text-primary-500'}`} />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-gray-800 truncate">{entry.file.name}</p>
                      {entry.error ? (
                        <p className="text-xs text-red-600">{entry.error}</p>
                      ) : (
                        <p className="text-xs text-gray-500">
                          {(entry.file.size / 1024 / 1024).toFixed(2)} MB
                        </p>
                      )}
                    </div>
                    {!uploading && (
                      <button
                        onClick={() => removeFile(entry.id)}
                        className="text-gray-400 hover:text-red-500 transition-colors p-1"
                        aria-label="Remove"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Summary line */}
            {files.length > 0 && (
              <p className="text-xs text-gray-500">
                {validFiles.length} valid file{validFiles.length !== 1 ? 's' : ''}
                {hasErrors ? ` · ${files.length - validFiles.length} invalid (will be skipped)` : ''}
              </p>
            )}

            {/* Upload indicator */}
            {uploading && (
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Loader2 className="w-4 h-4 animate-spin text-primary-500" />
                Uploading {validFiles.length} file{validFiles.length !== 1 ? 's' : ''}...
              </div>
            )}

            {/* Results */}
            {results && (
              <div className="space-y-1.5 max-h-40 overflow-y-auto">
                {results.map((r, i) => (
                  <div key={i} className="flex items-start gap-2 text-sm">
                    {statusIcon(r.status)}
                    <div className="flex-1 min-w-0">
                      <span className="font-medium text-gray-800 truncate block">{r.filename}</span>
                      <span className={`text-xs ${statusColor(r.status)}`}>{r.message}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Global error */}
            {globalError && (
              <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                <p className="text-xs sm:text-sm text-red-800">{globalError}</p>
              </div>
            )}

            {/* Form fields */}
            <div className="space-y-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Document Type *
                </label>
                <select
                  value={docType}
                  onChange={(e) => setDocType(e.target.value)}
                  disabled={uploading}
                  className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm"
                >
                  <option value="finance">Finance</option>
                  <option value="hrms">HRMS</option>
                  <option value="policy">Policy</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Department (Optional)
                </label>
                <input
                  type="text"
                  value={department}
                  onChange={(e) => setDepartment(e.target.value)}
                  placeholder="e.g., Finance, HR, Operations"
                  disabled={uploading}
                  className="w-full px-3 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm"
                />
              </div>
            </div>
          </div>

          {/* Footer */}
          <div className="px-4 sm:px-6 py-3 sm:py-4 bg-gray-50 border-t border-gray-200 flex flex-col-reverse sm:flex-row sm:items-center sm:justify-between gap-2">
            <button
              onClick={onClose}
              disabled={uploading}
              className="w-full sm:w-auto px-4 py-2.5 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 transition-colors"
            >
              {results ? 'Close' : 'Cancel'}
            </button>
            <button
              onClick={handleUpload}
              disabled={validFiles.length === 0 || uploading}
              className="w-full sm:w-auto px-4 py-2.5 text-sm font-medium text-white bg-gradient-to-r from-primary-500 to-primary-600 rounded-lg hover:from-primary-600 hover:to-primary-700 disabled:opacity-50 transition-all shadow-sm"
            >
              {uploading ? (
                <span className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Uploading...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  <Upload className="w-4 h-4" />
                  {validFiles.length > 1
                    ? `Upload ${validFiles.length} Files`
                    : 'Upload'}
                </span>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}