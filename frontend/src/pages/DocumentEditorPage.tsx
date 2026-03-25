import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft,
  FileText,
  Edit,
  Eye,
  BarChart3,
  Loader2,
  AlertCircle,
  Info,
  Sparkles
} from 'lucide-react';
import api, { DocumentInfo, ChunkData, DocumentEditStats } from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import { toast } from '../utils/toast';
import DocumentViewer from '../components/documents/DocumentViewer';
import ChunkList from '../components/documents/ChunkList';
import ChunkEditor from '../components/documents/ChunkEditor';
import EditHistoryModal from '../components/documents/EditHistoryModal';
import DocumentMetadataEditor from '../components/documents/DocumentMetadataEditor';

export default function DocumentEditorPage() {
  const { documentId } = useParams<{ documentId: string }>();
  const navigate = useNavigate();
  const { isAdmin } = useAuth();

  const [docInfo, setDocInfo] = useState<DocumentInfo | null>(null);
  const [chunks, setChunks] = useState<ChunkData[]>([]);
  const [stats, setStats] = useState<DocumentEditStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [showDocViewer, setShowDocViewer] = useState(false);
  const [showMetadataEditor, setShowMetadataEditor] = useState(false);
  const [selectedChunk, setSelectedChunk] = useState<ChunkData | null>(null);
  const [historyChunkId, setHistoryChunkId] = useState<string | null>(null);
  
  // Title generation state
  const [generatingTitles, setGeneratingTitles] = useState(false);

  useEffect(() => {
    if (documentId) {
      loadDocument();
    }
  }, [documentId]);

  const loadDocument = async () => {
    if (!documentId) return;

    setLoading(true);
    setError(null);

    try {
      const [info, chunksData, statsData] = await Promise.all([
        api.getDocumentInfo(documentId),
        api.getDocumentChunks(documentId),
        api.getDocumentEditStats(documentId)
      ]);

      setDocInfo(info);
      setChunks(chunksData);
      setStats(statsData);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to load document');
      toast.error('Failed to load document');
    } finally {
      setLoading(false);
    }
  };

  const handleSaveChunk = async (chunkId: string, newContent: string) => {
    try {
      const updatedChunk = await api.editChunk(chunkId, newContent);
      
      setChunks(prev => prev.map(c => c.id === chunkId ? updatedChunk : c));
      
      loadDocument();
      
      toast.success('Chunk updated successfully. Embedding regenerated.');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to save chunk');
      throw err;
    }
  };

  const handleRevertChunk = async (chunkId: string) => {
    try {
      await api.revertChunk(chunkId);
      
      loadDocument();
      
      toast.success('Chunk reverted to original content');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to revert chunk');
      throw err;
    }
  };

  const handleDeleteChunk = async (chunkId: string) => {
    try {
      await api.deleteChunk(chunkId);
      
      setChunks(prev => prev.filter(c => c.id !== chunkId));
      
      loadDocument();
      
      toast.success('Chunk deleted successfully');
    } catch (err: any) {
      toast.error(err.response?.data?.detail || 'Failed to delete chunk');
      throw err;
    }
  };

  const handleGenerateAllTitles = async () => {
    if (!documentId) return;
    
    if (!confirm(`Generate AI titles for all ${chunks.length} chunks? This may take a moment.`)) {
      return;
    }
    
    setGeneratingTitles(true);
    try {
      const result = await api.generateAllTitles(documentId);
      
      toast.success(`Generated ${result.success} titles successfully!`);
      
      // Reload document to show new titles
      await loadDocument();
      
    } catch (err: any) {
      console.error('Title generation failed:', err);
      toast.error(err.response?.data?.detail || 'Failed to generate titles');
    } finally {
      setGeneratingTitles(false);
    }
  };

  const handleSaveMetadata = async (metadata: any) => {
    toast.info('Metadata update feature coming soon');
  };

  if (loading) {
    return (
      <div className="h-full flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-primary-500 animate-spin" />
      </div>
    );
  }

  if (error || !docInfo) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center px-4">
        <AlertCircle className="w-12 h-12 sm:w-16 sm:h-16 text-red-400 mb-4" />
        <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mb-2">Failed to Load Document</h2>
        <p className="text-sm sm:text-base text-gray-600 mb-6">{error}</p>
        <button
          onClick={() => navigate('/documents')}
          className="flex items-center gap-2 px-4 py-2.5 text-white bg-primary-600 rounded-lg hover:bg-primary-700 transition-colors min-touch-target"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to Documents
        </button>
      </div>
    );
  }

  // Count chunks with AI titles
  const chunksWithTitles = chunks.filter(c => c.title).length;
  const titleCoverage = chunks.length > 0 ? Math.round((chunksWithTitles / chunks.length) * 100) : 0;

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-3 sm:px-6 py-3 sm:py-4 flex-shrink-0">
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3 sm:gap-4 mb-3 sm:mb-4">
          <div className="flex items-start gap-2 sm:gap-4 min-w-0 flex-1">
            <button
              onClick={() => navigate('/documents')}
              className="p-2 -ml-2 text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors min-touch-target flex-shrink-0"
              aria-label="Back to documents"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div className="min-w-0 flex-1">
              <h1 className="text-lg sm:text-2xl font-bold text-gray-900 break-words">
                {docInfo.original_filename}
              </h1>
              <p className="text-xs sm:text-sm text-gray-600 mt-1 flex flex-wrap gap-x-2 gap-y-1">
                <span>{docInfo.doc_type}</span>
                {docInfo.department && (
                  <>
                    <span className="hidden sm:inline">|</span>
                    <span>{docInfo.department}</span>
                  </>
                )}
                <span className="hidden sm:inline">|</span>
                <span>{docInfo.file_size_mb} MB</span>
                <span className="hidden sm:inline">|</span>
                <span>{docInfo.total_pages} pages</span>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-shrink-0 flex-wrap">
            <button
              onClick={() => setShowDocViewer(true)}
              className="flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-2 text-xs sm:text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors min-touch-target"
            >
              <Eye className="w-4 h-4" />
              <span className="hidden xs:inline">Preview</span>
            </button>
            
            <button
              onClick={() => setShowMetadataEditor(true)}
              className="flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-2 text-xs sm:text-sm font-medium text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors min-touch-target"
            >
              <Edit className="w-4 h-4" />
              <span className="hidden xs:inline">Metadata</span>
            </button>

            {isAdmin && (
              <button
                onClick={handleGenerateAllTitles}
                disabled={generatingTitles}
                className="flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-2 text-xs sm:text-sm font-medium text-white bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors min-touch-target disabled:opacity-50 disabled:cursor-not-allowed"
                title={`${titleCoverage}% chunks have AI titles`}
              >
                {generatingTitles ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span className="hidden xs:inline">Generating...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-4 h-4" />
                    <span className="hidden xs:inline">AI Titles</span>
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Access Level Info */}
        {!isAdmin && (
          <div className="mb-3 sm:mb-4 p-2.5 sm:p-3 bg-yellow-50 border border-yellow-200 rounded-lg flex items-start gap-2">
            <Info className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
            <p className="text-xs sm:text-sm text-yellow-800">
              You are viewing this document in read-only mode. Contact an administrator to make edits.
            </p>
          </div>
        )}

        {/* Title Coverage Info */}
        {isAdmin && titleCoverage < 100 && (
          <div className="mb-3 sm:mb-4 p-2.5 sm:p-3 bg-purple-50 border border-purple-200 rounded-lg flex items-start gap-2">
            <Sparkles className="w-4 h-4 sm:w-5 sm:h-5 text-purple-600 flex-shrink-0 mt-0.5" />
            <div className="flex-1">
              <p className="text-xs sm:text-sm text-purple-900 font-medium">
                {titleCoverage}% of chunks have AI-generated titles ({chunksWithTitles}/{chunks.length})
              </p>
              <p className="text-xs text-purple-700 mt-1">
                Click "AI Titles" to generate intelligent titles for all chunks
              </p>
            </div>
          </div>
        )}

        {/* Statistics - Responsive Grid */}
        {stats && (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2 sm:gap-4">
            <div className="bg-gray-50 rounded-lg p-2.5 sm:p-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-gray-600 truncate">Total Chunks</span>
                <FileText className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-gray-400 flex-shrink-0" />
              </div>
              <p className="text-lg sm:text-xl font-bold text-gray-900">{stats.total_chunks}</p>
            </div>

            <div className="bg-green-50 rounded-lg p-2.5 sm:p-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-green-600 truncate">Edited</span>
                <BarChart3 className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-green-500 flex-shrink-0" />
              </div>
              <p className="text-lg sm:text-xl font-bold text-green-700">{stats.edited_chunks}</p>
            </div>

            <div className="bg-blue-50 rounded-lg p-2.5 sm:p-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-blue-600 truncate">Original</span>
                <FileText className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-blue-500 flex-shrink-0" />
              </div>
              <p className="text-lg sm:text-xl font-bold text-blue-700">{stats.unedited_chunks}</p>
            </div>

            <div className="bg-purple-50 rounded-lg p-2.5 sm:p-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-purple-600 truncate">Total Edits</span>
                <Edit className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-purple-500 flex-shrink-0" />
              </div>
              <p className="text-lg sm:text-xl font-bold text-purple-700">{stats.total_edits}</p>
            </div>

            <div className="bg-yellow-50 rounded-lg p-2.5 sm:p-3 col-span-2 sm:col-span-1">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-yellow-600 truncate">Edited %</span>
                <BarChart3 className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-yellow-500 flex-shrink-0" />
              </div>
              <p className="text-lg sm:text-xl font-bold text-yellow-700">{stats.edit_percentage}%</p>
            </div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden">
        <ChunkList
          chunks={chunks}
          onEditChunk={(chunk) => setSelectedChunk(chunk)}
        />
      </div>

      {/* Modals */}
      {showDocViewer && documentId && (
        <DocumentViewer
          documentId={documentId}
          onClose={() => setShowDocViewer(false)}
        />
      )}

      {showMetadataEditor && (
        <DocumentMetadataEditor
          docInfo={docInfo}
          onSave={handleSaveMetadata}
          onClose={() => setShowMetadataEditor(false)}
        />
      )}

      {selectedChunk && (
        <ChunkEditor
          chunk={selectedChunk}
          onSave={handleSaveChunk}
          onRevert={handleRevertChunk}
          onDelete={handleDeleteChunk}
          onViewHistory={(chunkId) => {
            setHistoryChunkId(chunkId);
            setSelectedChunk(null);
          }}
          onClose={() => setSelectedChunk(null)}
        />
      )}

      {historyChunkId && (
        <EditHistoryModal
          chunkId={historyChunkId}
          onClose={() => setHistoryChunkId(null)}
        />
      )}
    </div>
  );
}