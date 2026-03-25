import { useState, useEffect } from 'react';
import { X, Save, RotateCcw, Trash2, History, AlertCircle, Sparkles, Loader2, Edit2 } from 'lucide-react';
import { ChunkData } from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';
import api from '../../services/api';
import { toast } from '../../utils/toast';

interface ChunkEditorProps {
  chunk: ChunkData;
  onSave: (chunkId: string, newContent: string) => Promise<void>;
  onRevert: (chunkId: string) => Promise<void>;
  onDelete: (chunkId: string) => Promise<void>;
  onViewHistory: (chunkId: string) => void;
  onClose: () => void;
}

export default function ChunkEditor({
  chunk,
  onSave,
  onRevert,
  onDelete,
  onViewHistory,
  onClose
}: ChunkEditorProps) {
  const { isAdmin } = useAuth();
  const [content, setContent] = useState(chunk.content);
  const [saving, setSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  
  // Title editing states
  const [titleValue, setTitleValue] = useState(chunk.title || '');
  const [editingTitle, setEditingTitle] = useState(false);
  const [generatingTitle, setGeneratingTitle] = useState(false);

  useEffect(() => {
    setHasChanges(content !== chunk.content);
  }, [content, chunk.content]);

  const handleSave = async () => {
    if (!hasChanges) {
      toast.info('No changes to save');
      return;
    }

    setSaving(true);
    try {
      await onSave(chunk.id, content);
      onClose();
    } catch (error) {
      console.error('Save failed:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleRevert = async () => {
    if (!confirm('Revert to original content? This will discard all edits.')) return;

    try {
      await onRevert(chunk.id);
      onClose();
    } catch (error) {
      console.error('Revert failed:', error);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Delete this chunk permanently? This action cannot be undone.')) return;

    try {
      await onDelete(chunk.id);
      onClose();
    } catch (error) {
      console.error('Delete failed:', error);
    }
  };

  const handleRegenerateTitle = async () => {
    setGeneratingTitle(true);
    try {
      const result = await api.generateChunkTitle(chunk.id);
      setTitleValue(result.title);
      toast.success('Title generated successfully!');
      // Reload to show new title
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    } catch (err: any) {
      console.error('Title generation failed:', err);
      toast.error(err.response?.data?.detail || 'Failed to generate title');
    } finally {
      setGeneratingTitle(false);
    }
  };

  const handleSaveTitle = async () => {
    if (!titleValue.trim()) {
      toast.error('Title cannot be empty');
      return;
    }

    try {
      await api.updateChunkTitle(chunk.id, titleValue.trim());
      setEditingTitle(false);
      toast.success('Title updated successfully!');
      // Reload to show new title
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    } catch (err: any) {
      console.error('Title update failed:', err);
      toast.error(err.response?.data?.detail || 'Failed to update title');
    }
  };

  const handleCancelTitleEdit = () => {
    setEditingTitle(false);
    setTitleValue(chunk.title || '');
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-2 sm:p-4">
      <div className="bg-white rounded-xl shadow-2xl w-full max-w-4xl max-h-[90vh] flex flex-col">
        {/* Header */}
        <div className="px-4 sm:px-6 py-4 border-b border-gray-200 flex items-center justify-between flex-shrink-0">
          <div className="flex-1 min-w-0 mr-4">
            <h2 className="text-lg sm:text-xl font-bold text-gray-900 truncate">
              {isAdmin ? 'Edit Chunk' : 'View Chunk'}
            </h2>
            <div className="flex flex-wrap items-center gap-x-2 gap-y-1 mt-1 text-xs sm:text-sm text-gray-600">
              <span>Index: {chunk.chunk_index + 1}</span>
              <span className="hidden xs:inline">•</span>
              <span>Type: {chunk.chunk_type}</span>
              <span className="hidden sm:inline">•</span>
              <span className="hidden sm:inline">Pages: {chunk.page_numbers.join(', ')}</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors flex-shrink-0"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4 scroll-smooth-touch">
          {/* Read-only notice */}
          {!isAdmin && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3 flex items-start gap-2">
              <AlertCircle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-yellow-800">
                You are viewing this chunk in read-only mode. Only administrators can make edits.
              </p>
            </div>
          )}

          {/* Edit status */}
          {chunk.is_edited && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <div className="flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1 text-sm text-blue-800">
                  <p className="font-medium">This chunk has been edited {chunk.edit_count} time(s)</p>
                  {chunk.edited_at && (
                    <p className="mt-1">Last edited: {new Date(chunk.edited_at).toLocaleString()}</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Title Section */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="block text-sm font-medium text-gray-700 flex items-center gap-2">
                Chunk Title
                {chunk.title && <Sparkles className="w-4 h-4 text-purple-500" title="AI-generated" />}
              </label>
              {isAdmin && (
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleRegenerateTitle}
                    disabled={generatingTitle}
                    className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-purple-600 hover:bg-purple-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                    title="Generate new AI title"
                  >
                    {generatingTitle ? (
                      <>
                        <Loader2 className="w-3.5 h-3.5 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Sparkles className="w-3.5 h-3.5" />
                        Regenerate
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>
            
            {editingTitle && isAdmin ? (
              <div className="space-y-2">
                <input
                  type="text"
                  value={titleValue}
                  onChange={(e) => setTitleValue(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
                  placeholder="Enter custom title (max 200 characters)..."
                  maxLength={200}
                />
                <div className="flex gap-2">
                  <button
                    onClick={handleSaveTitle}
                    className="px-3 py-1.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors"
                  >
                    Save Title
                  </button>
                  <button
                    onClick={handleCancelTitleEdit}
                    className="px-3 py-1.5 bg-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-300 transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              </div>
            ) : (
              <div
                onClick={() => isAdmin && setEditingTitle(true)}
                className={`px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm flex items-center justify-between ${
                  isAdmin ? 'cursor-pointer hover:bg-gray-100 hover:border-gray-300' : ''
                }`}
              >
                <span className="flex-1 truncate">
                  {titleValue || chunk.section_title || `Chunk #${chunk.chunk_index + 1}`}
                </span>
                {isAdmin && <Edit2 className="w-4 h-4 text-gray-400 ml-2 flex-shrink-0" />}
              </div>
            )}
          </div>

          {/* Section Title (if exists) */}
          {chunk.section_title && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Section
              </label>
              <div className="px-3 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm text-gray-700">
                {chunk.section_title}
              </div>
            </div>
          )}

          {/* Content Editor */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Content
            </label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              disabled={!isAdmin}
              className={`w-full h-64 sm:h-80 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 font-mono text-xs sm:text-sm resize-none ${
                !isAdmin ? 'bg-gray-50 cursor-not-allowed' : ''
              }`}
              placeholder="Chunk content..."
            />
            <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
              <span>Characters: {content.length}</span>
              <span>Tokens: ~{Math.ceil(content.length / 4)}</span>
            </div>
          </div>

          {/* Metadata */}
          {chunk.metadata && Object.keys(chunk.metadata).length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Metadata
              </label>
              <div className="bg-gray-50 rounded-lg p-3 border border-gray-200">
                <pre className="text-xs text-gray-700 overflow-x-auto">
                  {JSON.stringify(chunk.metadata, null, 2)}
                </pre>
              </div>
            </div>
          )}
        </div>

        {/* Footer Actions */}
        <div className="px-4 sm:px-6 py-4 border-t border-gray-200 flex flex-col sm:flex-row items-stretch sm:items-center gap-2 sm:gap-3 flex-shrink-0">
          {isAdmin ? (
            <>
              <button
                onClick={handleSave}
                disabled={!hasChanges || saving}
                className="flex items-center justify-center gap-2 px-4 py-2.5 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium text-sm sm:text-base order-1 sm:order-none"
              >
                {saving ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4" />
                    Save Changes
                  </>
                )}
              </button>

              {chunk.is_edited && (
                <button
                  onClick={handleRevert}
                  className="flex items-center justify-center gap-2 px-4 py-2.5 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors font-medium text-sm sm:text-base order-2 sm:order-none"
                >
                  <RotateCcw className="w-4 h-4" />
                  <span className="hidden xs:inline">Revert to Original</span>
                  <span className="xs:hidden">Revert</span>
                </button>
              )}

              <button
                onClick={() => onViewHistory(chunk.id)}
                className="flex items-center justify-center gap-2 px-4 py-2.5 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors font-medium text-sm sm:text-base order-3 sm:order-none"
              >
                <History className="w-4 h-4" />
                <span className="hidden sm:inline">View History</span>
                <span className="sm:hidden">History</span>
              </button>

              <button
                onClick={handleDelete}
                className="flex items-center justify-center gap-2 px-4 py-2.5 text-red-600 hover:bg-red-50 rounded-lg transition-colors font-medium text-sm sm:text-base order-4 sm:order-none sm:ml-auto"
              >
                <Trash2 className="w-4 h-4" />
                <span className="hidden xs:inline">Delete Chunk</span>
                <span className="xs:hidden">Delete</span>
              </button>
            </>
          ) : (
            <button
              onClick={onClose}
              className="flex items-center justify-center gap-2 px-6 py-2.5 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium text-sm sm:text-base"
            >
              Close
            </button>
          )}
        </div>
      </div>
    </div>
  );
}