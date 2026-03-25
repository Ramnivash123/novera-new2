import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  MessageSquare,
  Trash2,
  ExternalLink,
  Search,
  Calendar,
  TrendingUp,
  Filter,
  Download,
  BarChart3,
  Clock,
  ChevronDown,
  X,
  AlertCircle,
  Plus,
  CheckSquare,
  Square,
  Sparkles
} from 'lucide-react';
import { formatDistanceToNow, format, isToday, isThisWeek, isThisMonth } from 'date-fns';
import api, { Conversation } from '../services/api';
import { ConversationCardSkeleton } from '../components/common/Skeletons';
import { toast } from '../utils/toast';

type SortOption = 'recent' | 'oldest' | 'messages' | 'duration';
type FilterOption = 'all' | 'today' | 'week' | 'month';

interface ConversationStats {
  total: number;
  today: number;
  week: number;
  totalMessages: number;
  avgMessagesPerConversation: number;
}

export default function ConversationsPage() {
  const navigate = useNavigate();
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<SortOption>('recent');
  const [filterBy, setFilterBy] = useState<FilterOption>('all');
  const [showFilters, setShowFilters] = useState(false);
  const [selectedConversations, setSelectedConversations] = useState<Set<string>>(new Set());
  const [bulkActionMode, setBulkActionMode] = useState(false);
  const [deleteLoading, setDeleteLoading] = useState<string | null>(null);
  const [bulkDeleteLoading, setBulkDeleteLoading] = useState(false);

  useEffect(() => {
    loadConversations();
  }, []);

  const loadConversations = async () => {
    setLoading(true);
    try {
      const data = await api.getConversations(50);
      setConversations(data);
    } catch (error) {
      console.error('Failed to load conversations:', error);
      toast.error('Failed to load conversations');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteConversation = async (conversationId: string, event?: React.MouseEvent) => {
    if (event) {
      event.stopPropagation();
    }

    if (!confirm('Are you sure you want to delete this conversation? This action cannot be undone.')) {
      return;
    }

    setDeleteLoading(conversationId);
    try {
      await api.deleteConversation(conversationId);
      setConversations(convs => convs.filter(c => c.id !== conversationId));
      setSelectedConversations(prev => {
        const next = new Set(prev);
        next.delete(conversationId);
        return next;
      });
      toast.success('Conversation deleted successfully');
    } catch (error) {
      console.error('Failed to delete conversation:', error);
      toast.error('Failed to delete conversation');
    } finally {
      setDeleteLoading(null);
    }
  };

  const handleBulkDelete = async () => {
    if (selectedConversations.size === 0) return;

    const confirmMsg = `Delete ${selectedConversations.size} conversation${selectedConversations.size > 1 ? 's' : ''}? This action cannot be undone.`;
    if (!confirm(confirmMsg)) return;

    setBulkDeleteLoading(true);
    const deleteIds = Array.from(selectedConversations);
    let successCount = 0;
    let failCount = 0;

    try {
      for (const id of deleteIds) {
        try {
          await api.deleteConversation(id);
          successCount++;
          setConversations(convs => convs.filter(c => c.id !== id));
        } catch (error) {
          console.error(`Failed to delete conversation ${id}:`, error);
          failCount++;
        }
      }

      setSelectedConversations(new Set());
      setBulkActionMode(false);

      if (failCount === 0) {
        toast.success(`${successCount} conversation${successCount > 1 ? 's' : ''} deleted successfully`);
      } else {
        toast.warning(`${successCount} deleted, ${failCount} failed`);
      }
    } catch (error) {
      console.error('Bulk delete failed:', error);
      toast.error('Bulk delete operation failed');
    } finally {
      setBulkDeleteLoading(false);
    }
  };

  const toggleConversationSelection = (id: string, event?: React.MouseEvent) => {
    if (event) {
      event.stopPropagation();
    }
    setSelectedConversations(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleSelectAll = () => {
    if (selectedConversations.size === processedConversations.length) {
      setSelectedConversations(new Set());
    } else {
      setSelectedConversations(new Set(processedConversations.map(c => c.id)));
    }
  };

  const handleExportConversation = async (conversationId: string, event?: React.MouseEvent) => {
    if (event) {
      event.stopPropagation();
    }

    try {
      const blob = await api.exportConversation(conversationId, 'markdown');
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `conversation_${conversationId.slice(0, 8)}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      toast.success('Conversation exported successfully');
    } catch (error) {
      console.error('Failed to export conversation:', error);
      toast.error('Failed to export conversation');
    }
  };

  const getConversationPreview = (conv: Conversation): string => {
    const userMessages = conv.messages.filter(m => m.role === 'user');
    if (userMessages.length === 0) return 'New conversation';
    const firstMessage = userMessages[0].content;
    return firstMessage.substring(0, 150) + (firstMessage.length > 150 ? '...' : '');
  };

  const getConversationTitle = (conv: Conversation): string => {
    const userMessages = conv.messages.filter(m => m.role === 'user');
    if (userMessages.length === 0) return 'New conversation';
    const firstMessage = userMessages[0].content;
    return firstMessage.substring(0, 60) + (firstMessage.length > 60 ? '...' : '');
  };

  const calculateStats = (): ConversationStats => {
    const today = conversations.filter(c => isToday(new Date(c.created_at))).length;
    const week = conversations.filter(c => isThisWeek(new Date(c.created_at))).length;
    const totalMessages = conversations.reduce((sum, c) => sum + c.messages.length, 0);

    return {
      total: conversations.length,
      today,
      week,
      totalMessages,
      avgMessagesPerConversation: conversations.length > 0
        ? Math.round(totalMessages / conversations.length)
        : 0
    };
  };

  const filterConversations = (convs: Conversation[]): Conversation[] => {
    switch (filterBy) {
      case 'today':
        return convs.filter(c => isToday(new Date(c.created_at)));
      case 'week':
        return convs.filter(c => isThisWeek(new Date(c.created_at)));
      case 'month':
        return convs.filter(c => isThisMonth(new Date(c.created_at)));
      default:
        return convs;
    }
  };

  const sortConversations = (convs: Conversation[]): Conversation[] => {
    const sorted = [...convs];

    switch (sortBy) {
      case 'recent':
        return sorted.sort((a, b) =>
          new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
        );
      case 'oldest':
        return sorted.sort((a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime()
        );
      case 'messages':
        return sorted.sort((a, b) => b.messages.length - a.messages.length);
      case 'duration':
        return sorted.sort((a, b) => {
          const durationA = new Date(a.updated_at).getTime() - new Date(a.created_at).getTime();
          const durationB = new Date(b.updated_at).getTime() - new Date(b.created_at).getTime();
          return durationB - durationA;
        });
      default:
        return sorted;
    }
  };

  const processedConversations = sortConversations(
    filterConversations(
      conversations.filter(conv =>
        getConversationPreview(conv).toLowerCase().includes(searchQuery.toLowerCase())
      )
    )
  );

  const stats = calculateStats();

  return (
    <div className="h-full flex flex-col bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 flex-shrink-0">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Title Section */}
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">
                Conversation History
              </h1>
              <p className="text-sm text-gray-600 mt-1">
                View and manage your chat conversations
              </p>
            </div>

            <button
              onClick={() => navigate('/chat')}
              className="inline-flex items-center gap-2 px-4 py-2.5 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-medium shadow-sm"
            >
              <Plus className="w-4 h-4" />
              <span className="hidden sm:inline">New Chat</span>
            </button>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-blue-900">Total</p>
                  <p className="text-2xl font-bold text-blue-900 mt-1">{stats.total}</p>
                </div>
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                  <MessageSquare className="w-6 h-6 text-blue-600" />
                </div>
              </div>
              <p className="text-xs text-blue-700 mt-2">conversations</p>
            </div>

            <div className="bg-gradient-to-br from-green-50 to-green-100 border border-green-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-green-900">Today</p>
                  <p className="text-2xl font-bold text-green-900 mt-1">{stats.today}</p>
                </div>
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                  <Sparkles className="w-6 h-6 text-green-600" />
                </div>
              </div>
              <p className="text-xs text-green-700 mt-2">new chats</p>
            </div>

            <div className="bg-gradient-to-br from-purple-50 to-purple-100 border border-purple-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-purple-900">This Week</p>
                  <p className="text-2xl font-bold text-purple-900 mt-1">{stats.week}</p>
                </div>
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                  <Calendar className="w-6 h-6 text-purple-600" />
                </div>
              </div>
              <p className="text-xs text-purple-700 mt-2">conversations</p>
            </div>

            <div className="bg-gradient-to-br from-orange-50 to-orange-100 border border-orange-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-orange-900">Avg Messages</p>
                  <p className="text-2xl font-bold text-orange-900 mt-1">{stats.avgMessagesPerConversation}</p>
                </div>
                <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-orange-600" />
                </div>
              </div>
              <p className="text-xs text-orange-700 mt-2">per chat</p>
            </div>
          </div>

          {/* Controls */}
          <div className="flex flex-col sm:flex-row gap-3">
            {/* Search */}
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search conversations..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-10 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-primary-500 text-sm"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
            </div>

            {/* Filter & Sort */}
            <div className="flex gap-2">
              <div className="relative">
                <button
                  onClick={() => setShowFilters(!showFilters)}
                  className="inline-flex items-center gap-2 px-4 py-2.5 border border-gray-300 bg-white rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium"
                >
                  <Filter className="w-4 h-4" />
                  <span>Filters</span>
                  <ChevronDown className={`w-4 h-4 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
                </button>

                {showFilters && (
                  <>
                    <div className="fixed inset-0 z-10" onClick={() => setShowFilters(false)} />
                    <div className="absolute right-0 mt-2 w-64 bg-white rounded-lg shadow-lg border border-gray-200 py-2 z-20">
                      <div className="px-3 py-2 border-b border-gray-100">
                        <p className="text-xs font-semibold text-gray-700 uppercase">Time Period</p>
                      </div>
                      {(['all', 'today', 'week', 'month'] as FilterOption[]).map((option) => (
                        <button
                          key={option}
                          onClick={() => {
                            setFilterBy(option);
                            setShowFilters(false);
                          }}
                          className={`w-full px-4 py-2.5 text-left text-sm transition-colors ${
                            filterBy === option
                              ? 'bg-primary-50 text-primary-700 font-medium'
                              : 'text-gray-700 hover:bg-gray-50'
                          }`}
                        >
                          {option.charAt(0).toUpperCase() + option.slice(1)}
                        </button>
                      ))}

                      <div className="px-3 py-2 border-t border-b border-gray-100 mt-1">
                        <p className="text-xs font-semibold text-gray-700 uppercase">Sort By</p>
                      </div>
                      {([
                        { value: 'recent' as const, label: 'Most Recent', icon: Clock },
                        { value: 'oldest' as const, label: 'Oldest First', icon: Calendar },
                        { value: 'messages' as const, label: 'Most Messages', icon: MessageSquare },
                        { value: 'duration' as const, label: 'Longest Duration', icon: TrendingUp }
                      ]).map(({ value, label, icon: Icon }) => (
                        <button
                          key={value}
                          onClick={() => {
                            setSortBy(value);
                            setShowFilters(false);
                          }}
                          className={`w-full px-4 py-2.5 text-left text-sm transition-colors flex items-center gap-2 ${
                            sortBy === value
                              ? 'bg-primary-50 text-primary-700 font-medium'
                              : 'text-gray-700 hover:bg-gray-50'
                          }`}
                        >
                          <Icon className="w-4 h-4" />
                          {label}
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>

              {bulkActionMode ? (
                <div className="flex items-center gap-2 px-4 py-2.5 bg-primary-50 border border-primary-200 rounded-lg">
                  <span className="text-sm font-medium text-primary-700">
                    {selectedConversations.size} selected
                  </span>
                  {selectedConversations.size > 0 && (
                    <button
                      onClick={handleBulkDelete}
                      disabled={bulkDeleteLoading}
                      className="ml-2 px-3 py-1 bg-red-600 text-white rounded text-sm font-medium hover:bg-red-700 disabled:opacity-50"
                    >
                      {bulkDeleteLoading ? 'Deleting...' : 'Delete'}
                    </button>
                  )}
                  <button
                    onClick={() => {
                      setBulkActionMode(false);
                      setSelectedConversations(new Set());
                    }}
                    className="ml-1 p-1 hover:bg-primary-100 rounded"
                  >
                    <X className="w-4 h-4 text-primary-700" />
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setBulkActionMode(true)}
                  className="inline-flex items-center gap-2 px-4 py-2.5 border border-gray-300 bg-white rounded-lg hover:bg-gray-50 transition-colors text-sm font-medium"
                >
                  <CheckSquare className="w-4 h-4" />
                  <span className="hidden sm:inline">Select</span>
                </button>
              )}
            </div>
          </div>

          {/* Active Filters */}
          {(filterBy !== 'all' || sortBy !== 'recent') && (
            <div className="flex flex-wrap items-center gap-2 mt-4">
              <span className="text-sm text-gray-600">Active:</span>
              {filterBy !== 'all' && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-primary-100 text-primary-700 rounded-full text-sm font-medium">
                  {filterBy.charAt(0).toUpperCase() + filterBy.slice(1)}
                  <button onClick={() => setFilterBy('all')} className="hover:text-primary-900">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </span>
              )}
              {sortBy !== 'recent' && (
                <span className="inline-flex items-center gap-1.5 px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
                  Sort: {sortBy}
                  <button onClick={() => setSortBy('recent')} className="hover:text-blue-900">
                    <X className="w-3.5 h-3.5" />
                  </button>
                </span>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {loading ? (
            <div className="space-y-4">
              {Array.from({ length: 5 }).map((_, i) => (
                <ConversationCardSkeleton key={i} />
              ))}
            </div>
          ) : processedConversations.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 px-4">
              <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mb-4">
                {searchQuery || filterBy !== 'all' ? (
                  <AlertCircle className="w-10 h-10 text-gray-400" />
                ) : (
                  <MessageSquare className="w-10 h-10 text-gray-400" />
                )}
              </div>
              <h3 className="text-lg font-semibold text-gray-900 mb-2">
                {searchQuery || filterBy !== 'all' ? 'No conversations found' : 'No conversations yet'}
              </h3>
              <p className="text-gray-600 text-center max-w-md mb-6">
                {searchQuery || filterBy !== 'all'
                  ? 'Try adjusting your search or filter criteria'
                  : 'Start a new conversation to begin chatting'
                }
              </p>
              {!searchQuery && filterBy === 'all' && (
                <button
                  onClick={() => navigate('/chat')}
                  className="inline-flex items-center gap-2 px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors font-medium"
                >
                  <Plus className="w-5 h-5" />
                  Start Chatting
                </button>
              )}
            </div>
          ) : (
            <>
              {bulkActionMode && (
                <div className="mb-4 flex items-center justify-between">
                  <p className="text-sm text-gray-600">
                    {processedConversations.length} conversation{processedConversations.length !== 1 ? 's' : ''}
                  </p>
                  <button
                    onClick={toggleSelectAll}
                    className="text-sm text-primary-600 hover:text-primary-700 font-medium"
                  >
                    {selectedConversations.size === processedConversations.length ? 'Deselect All' : 'Select All'}
                  </button>
                </div>
              )}

              <div className="space-y-3">
                {processedConversations.map((conv) => {
                  const isSelected = selectedConversations.has(conv.id);
                  const isDeleting = deleteLoading === conv.id;

                  return (
                    <div
                      key={conv.id}
                      onClick={() => !bulkActionMode && navigate(`/chat/${conv.id}`)}
                      className={`group bg-white border rounded-lg transition-all ${
                        isSelected
                          ? 'border-primary-300 ring-2 ring-primary-100 shadow-sm'
                          : 'border-gray-200 hover:border-gray-300 hover:shadow-sm'
                      } ${isDeleting ? 'opacity-50' : ''} ${!bulkActionMode ? 'cursor-pointer' : ''}`}
                    >
                      <div className="p-4 sm:p-5">
                        <div className="flex gap-3 sm:gap-4">
                          {/* Checkbox */}
                          {bulkActionMode && (
                            <div className="flex-shrink-0 pt-1">
                              <button
                                onClick={(e) => toggleConversationSelection(conv.id, e)}
                                className="w-5 h-5"
                              >
                                {isSelected ? (
                                  <CheckSquare className="w-5 h-5 text-primary-600" />
                                ) : (
                                  <Square className="w-5 h-5 text-gray-400 hover:text-gray-600" />
                                )}
                              </button>
                            </div>
                          )}

                          {/* Content */}
                          <div className="flex-1 min-w-0">
                            {/* Title and Preview */}
                            <div className="mb-3">
                              <h3 className="text-base font-semibold text-gray-900 mb-1.5 line-clamp-1 group-hover:text-primary-600 transition-colors">
                                {getConversationTitle(conv)}
                              </h3>
                              <p className="text-sm text-gray-600 line-clamp-2 leading-relaxed">
                                {getConversationPreview(conv)}
                              </p>
                            </div>

                            {/* Metadata */}
                            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-gray-500 mb-3 pb-3 border-b border-gray-100">
                              <div className="flex items-center gap-1.5">
                                <Clock className="w-3.5 h-3.5" />
                                <span>{formatDistanceToNow(new Date(conv.created_at), { addSuffix: true })}</span>
                              </div>
                              <div className="flex items-center gap-1.5">
                                <MessageSquare className="w-3.5 h-3.5" />
                                <span>{conv.messages.length} messages</span>
                              </div>
                              <div className="flex items-center gap-1.5">
                                <Calendar className="w-3.5 h-3.5" />
                                <span>{format(new Date(conv.created_at), 'MMM d, yyyy')}</span>
                              </div>
                            </div>

                            {/* Actions */}
                            <div className="flex flex-wrap items-center gap-2">
                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  navigate(`/chat/${conv.id}`);
                                }}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-primary-600 text-white rounded-md hover:bg-primary-700 transition-colors text-xs font-medium"
                              >
                                <ExternalLink className="w-3.5 h-3.5" />
                                Continue
                              </button>

                              <button
                                onClick={(e) => {
                                  e.stopPropagation();
                                  navigate(`/chat/${conv.id}?view=analytics`);
                                }}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors text-xs font-medium"
                              >
                                <BarChart3 className="w-3.5 h-3.5" />
                                <span className="hidden sm:inline">Analytics</span>
                              </button>

                              <button
                                onClick={(e) => handleExportConversation(conv.id, e)}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 border border-gray-300 text-gray-700 rounded-md hover:bg-gray-50 transition-colors text-xs font-medium"
                              >
                                <Download className="w-3.5 h-3.5" />
                                <span className="hidden sm:inline">Export</span>
                              </button>

                              <button
                                onClick={(e) => handleDeleteConversation(conv.id, e)}
                                disabled={isDeleting}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-red-600 hover:bg-red-50 rounded-md transition-colors text-xs font-medium ml-auto disabled:opacity-50"
                              >
                                {isDeleting ? (
                                  <>
                                    <div className="w-3.5 h-3.5 border-2 border-red-600 border-t-transparent rounded-full animate-spin" />
                                    <span className="hidden sm:inline">Deleting...</span>
                                  </>
                                ) : (
                                  <>
                                    <Trash2 className="w-3.5 h-3.5" />
                                    <span className="hidden sm:inline">Delete</span>
                                  </>
                                )}
                              </button>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}