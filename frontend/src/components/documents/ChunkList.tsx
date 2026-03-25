import { useState } from 'react';
import { Edit, FileText, CheckCircle, Search, Filter, X, Sparkles } from 'lucide-react';
import { ChunkData } from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

interface ChunkListProps {
  chunks: ChunkData[];
  onEditChunk: (chunk: ChunkData) => void;
}

export default function ChunkList({ chunks, onEditChunk }: ChunkListProps) {
  const { isAdmin } = useAuth();
  const [searchQuery, setSearchQuery] = useState('');
  const [filterEdited, setFilterEdited] = useState<'all' | 'edited' | 'original'>('all');
  const [showFilters, setShowFilters] = useState(false);

  // Helper function to get display title
  const getChunkTitle = (chunk: ChunkData): string => {
    if (chunk.title) {
      return chunk.title;
    }
    
    // Fallback: Use section title
    if (chunk.section_title) {
      return chunk.section_title;
    }
    
    // Last resort: Generic title
    return `Chunk #${chunk.chunk_index + 1}`;
  };

  const filteredChunks = chunks.filter(chunk => {
    const title = getChunkTitle(chunk);
    const matchesSearch = chunk.content.toLowerCase().includes(searchQuery.toLowerCase()) ||
                         title.toLowerCase().includes(searchQuery.toLowerCase());
    
    const matchesFilter = filterEdited === 'all' ? true :
                         filterEdited === 'edited' ? chunk.is_edited :
                         !chunk.is_edited;
    
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="px-3 sm:px-6 py-3 sm:py-4 border-b border-gray-200 bg-white flex-shrink-0">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 sm:gap-4 mb-3 sm:mb-4">
          <h2 className="text-base sm:text-lg font-semibold text-gray-900">
            Document Chunks ({chunks.length})
          </h2>
          
          {/* Desktop Filter Buttons */}
          <div className="hidden sm:flex items-center gap-2">
            <button
              onClick={() => setFilterEdited('all')}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                filterEdited === 'all'
                  ? 'bg-primary-100 text-primary-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              All
            </button>
            <button
              onClick={() => setFilterEdited('edited')}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                filterEdited === 'edited'
                  ? 'bg-primary-100 text-primary-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Edited
            </button>
            <button
              onClick={() => setFilterEdited('original')}
              className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                filterEdited === 'original'
                  ? 'bg-primary-100 text-primary-700'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              Original
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
            <input
              type="text"
              placeholder="Search chunks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm sm:text-base"
            />
          </div>

          {/* Mobile Filter Toggle */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="sm:hidden flex items-center justify-center gap-2 px-3 py-2.5 border border-gray-300 rounded-lg bg-white hover:bg-gray-50 transition-colors min-touch-target"
          >
            <Filter className="w-4 h-4 text-gray-600" />
          </button>
        </div>

        {/* Mobile Filter Panel */}
        {showFilters && (
          <div className="sm:hidden mt-3 p-3 bg-gray-50 border border-gray-200 rounded-lg animate-slideUp">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm font-medium text-gray-700">Filter by status:</p>
              <button
                onClick={() => setShowFilters(false)}
                className="p-1 hover:bg-gray-200 rounded"
              >
                <X className="w-4 h-4 text-gray-500" />
              </button>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {['all', 'edited', 'original'].map((status) => (
                <button
                  key={status}
                  onClick={() => {
                    setFilterEdited(status as 'all' | 'edited' | 'original');
                    setShowFilters(false);
                  }}
                  className={`px-3 py-2 rounded-lg text-sm transition-colors min-touch-target ${
                    filterEdited === status
                      ? 'bg-primary-100 text-primary-700 font-medium'
                      : 'bg-white text-gray-700 hover:bg-gray-100 border border-gray-300'
                  }`}
                >
                  {status.charAt(0).toUpperCase() + status.slice(1)}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Chunk List */}
      <div className="flex-1 overflow-y-auto scroll-smooth-touch">
        {filteredChunks.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-64 text-center px-4">
            <FileText className="w-12 h-12 sm:w-16 sm:h-16 text-gray-300 mb-4" />
            <p className="text-sm sm:text-base text-gray-600">
              {searchQuery ? 'No chunks found matching your search' : 'No chunks available'}
            </p>
          </div>
        ) : (
          <div className="p-3 sm:p-6 space-y-3">
            {filteredChunks.map((chunk) => {
              const displayTitle = getChunkTitle(chunk);
              const hasAITitle = !!chunk.title;
              
              return (
                <div
                  key={chunk.id}
                  className="bg-white border border-gray-200 rounded-lg hover:shadow-md transition-shadow overflow-hidden"
                >
                  <div className="p-3 sm:p-4">
                    {/* Chunk Header */}
                    <div className="flex items-start justify-between gap-2 mb-2 sm:mb-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start gap-2 mb-1">
                          {hasAITitle && (
                            <Sparkles className="w-4 h-4 text-purple-500 flex-shrink-0 mt-0.5" title="AI-generated title" />
                          )}
                          <h3 className="text-sm sm:text-base font-semibold text-gray-900 break-words flex-1">
                            {displayTitle}
                          </h3>
                        </div>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-xs text-gray-500">#{chunk.chunk_index + 1}</span>
                          {chunk.is_edited && (
                            <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-yellow-100 text-yellow-800">
                              <CheckCircle className="w-3 h-3 mr-1" />
                              Edited ({chunk.edit_count}x)
                            </span>
                          )}
                        </div>
                        <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-gray-500 mt-1">
                          <span>Type: {chunk.chunk_type}</span>
                          <span className="hidden xs:inline">•</span>
                          <span>Pages: {chunk.page_numbers.join(', ')}</span>
                        </div>
                      </div>
                      <button
                        onClick={() => onEditChunk(chunk)}
                        className="flex items-center gap-1.5 px-2.5 sm:px-3 py-1.5 text-xs sm:text-sm font-medium text-primary-700 bg-primary-50 rounded-lg hover:bg-primary-100 transition-colors min-touch-target flex-shrink-0"
                      >
                        <Edit className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                        <span className="hidden xs:inline">{isAdmin ? 'Edit' : 'View'}</span>
                      </button>
                    </div>

                    {/* Chunk Preview */}
                    <div className="bg-gray-50 rounded-lg p-2.5 sm:p-3 border border-gray-200">
                      <p className="text-xs sm:text-sm text-gray-700 line-clamp-3 break-words">
                        {chunk.content}
                      </p>
                    </div>

                    {/* Chunk Stats */}
                    <div className="mt-2 sm:mt-3 flex flex-wrap items-center justify-between gap-2 text-xs text-gray-500">
                      <span>Tokens: {chunk.token_count}</span>
                      <span>Length: {chunk.content.length} chars</span>
                      {chunk.edited_at && (
                        <span className="hidden sm:inline">Last edited: {new Date(chunk.edited_at).toLocaleDateString()}</span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}