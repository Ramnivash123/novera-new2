import { useState, useEffect, useRef, useMemo, useCallback, memo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Send, Loader2, FileText, AlertCircle, Plus, Sparkles, BarChart3, X, ChevronDown, ChevronUp } from 'lucide-react';
import api, { ChatMessage, ChatResponse, Source } from '../services/api';
import MessageBubble from '../components/chat/MessageBubble';
import SourceCard from '../components/chat/SourceCard';
import ContextIndicator from '../components/chat/ContextIndicator';
import ExportButton from '../components/chat/ExportButton';
import AnalyticsModal from '../components/chat/AnalyticsModal';
import { useConversation } from '../contexts/ConversationContext';
import SelectiveExportModal from '../components/chat/SelectiveExportModal';
import { useCustomization } from '../contexts/CustomizationContext';
import { useAuth } from '../contexts/AuthContext';

const MemoizedMessageBubble = memo(MessageBubble);
const MemoizedSourceCard = memo(SourceCard);

function getFullImageUrl(path: string | null): string {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  return `${API_BASE}${path}`;
}

export default function ChatPage() {
  const { conversationId } = useParams();
  const navigate = useNavigate();
  const { setCurrentConversationId } = useConversation();
  const { customization } = useCustomization();
  const { isAdmin } = useAuth();

  // State
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentConversationIdLocal, setCurrentConversationIdLocal] = useState<string | null>(
    conversationId || null
  );
  const [sources, setSources] = useState<Source[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [contextSummary, setContextSummary] = useState<any>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [showSelectiveExport, setShowSelectiveExport] = useState(false);
  const [showSourcesDrawer, setShowSourcesDrawer] = useState(false);
  const [showSourcesPanel, setShowSourcesPanel] = useState(true);
  const [isDarkMode, setIsDarkMode] = useState(
    () => document.documentElement.classList.contains('dark')
  );

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Sync dark mode with the html class (set by CustomizationContext)
  useEffect(() => {
    const observer = new MutationObserver(() => {
      setIsDarkMode(document.documentElement.classList.contains('dark'));
    });
    observer.observe(document.documentElement, { attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  // Sync global conversation context
  useEffect(() => {
    setCurrentConversationId(currentConversationIdLocal);
    return () => setCurrentConversationId(null);
  }, [currentConversationIdLocal, setCurrentConversationId]);

  // Load conversation from URL param
  useEffect(() => {
    if (conversationId) loadConversation(conversationId);
  }, [conversationId]);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  // Keyboard shortcut: Ctrl/Cmd+A → analytics
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'a' && currentConversationIdLocal) {
        e.preventDefault();
        setShowAnalytics(true);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [currentConversationIdLocal]);

  // Close mobile sources drawer on outside click
  useEffect(() => {
    const onClickOutside = (e: MouseEvent) => {
      if (window.innerWidth >= 1024) return;
      const drawer = document.getElementById('sources-drawer');
      const btn = document.getElementById('sources-drawer-button');
      if (showSourcesDrawer && drawer && !drawer.contains(e.target as Node) && btn && !btn.contains(e.target as Node)) {
        setShowSourcesDrawer(false);
      }
    };
    document.addEventListener('mousedown', onClickOutside);
    return () => document.removeEventListener('mousedown', onClickOutside);
  }, [showSourcesDrawer]);

  // ── Helpers ───────────────────────────────────────────────────────────────

  const loadConversation = async (convId: string) => {
    try {
      const conversation = await api.getConversation(convId);
      setMessages(conversation.messages);
      setCurrentConversationIdLocal(convId);
    } catch (err: any) {
      const status = err?.response?.status;
      if (status === 403 || status === 404) {
        setCurrentConversationIdLocal(null);
        navigate('/chat', { replace: true });
      }
    }
  };

  const handleChatResponse = useCallback((response: ChatResponse) => {
    setSources(response.sources.map(src => ({ ...src })));
    setTimeout(() => {
      if (response.suggestions && Array.isArray(response.suggestions)) {
        setSuggestions(response.suggestions);
        setShowSuggestions(true);
      } else {
        setSuggestions([]);
      }
    }, 0);
    if (response.metadata?.context_summary) {
      setContextSummary(response.metadata.context_summary);
    }
  }, []);

  const handleSendMessage = async () => {
    if (!input.trim() || loading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);
    setError(null);

    try {
      const response: ChatResponse = await api.sendChatMessage({
        query: input.trim(),
        conversation_id: currentConversationIdLocal,
      });

      if (!currentConversationIdLocal && response.conversation_id) {
        setCurrentConversationIdLocal(response.conversation_id);
        navigate(`/chat/${response.conversation_id}`, { replace: true });
      }

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.answer,
        timestamp: new Date().toISOString(),
        metadata: {
          sources: response.sources,
          confidence: response.confidence,
          citations: response.citations,
          reformulated_query: response.metadata?.query_reformulated ? input.trim() : undefined,
        },
      };

      setMessages(prev => [...prev, assistantMessage]);
      handleChatResponse(response);
      setError(null);
    } catch (err: any) {
      console.error('Chat error:', err);
      const errorMessage = err.response?.data?.detail || 'Failed to send message. Please try again.';
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `I apologize, but I encountered an error: ${errorMessage}\n\nPlease try rephrasing your question or try again in a moment.`,
        timestamp: new Date().toISOString(),
        metadata: { error: true },
      }]);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const startNewChat = useCallback(() => {
    setMessages([]);
    setCurrentConversationIdLocal(null);
    setSources([]);
    setError(null);
    setContextSummary(null);
    setSuggestions([]);
    navigate('/chat');
    inputRef.current?.focus();
  }, [navigate]);

  const clearContext = useCallback(() => {
    setContextSummary(null);
    setSuggestions([]);
  }, []);

  const handleSuggestionClick = useCallback((suggestion: string) => {
    setInput(suggestion);
    setSuggestions([]);
  }, []);

  const exampleQuestions = useMemo(() => [
    'What is this document about?',
    'Explain the leave policy',
    'Show me expense reimbursement details',
    'What was the Q4 revenue?',
  ], []);

  const currentLogo = useMemo(() => {
    if (isDarkMode && customization?.branding?.logo_dark_url) {
      return getFullImageUrl(customization.branding.logo_dark_url);
    }
    if (customization?.branding?.logo_url) {
      return getFullImageUrl(customization.branding.logo_url);
    }
    return null;
  }, [isDarkMode, customization]);

  const appName = customization?.branding?.app_name || 'Novera AI';

  // Only show sources UI to admin users
  const canSeeSources = isAdmin;

  return (
    <div className="flex h-full bg-gray-50 relative">
      {/* Mobile Sources Drawer Overlay — admin only */}
      {canSeeSources && showSourcesDrawer && sources.length > 0 && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden"
          onClick={() => setShowSourcesDrawer(false)}
        />
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">

        {/* ── Header ────────────────────────────────────────────────────────── */}
        <div className="h-14 sm:h-16 border-b border-gray-200 bg-white px-3 sm:px-6 flex items-center justify-between shadow-sm flex-shrink-0">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0 flex-1">
            <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-md flex-shrink-0">
              <Sparkles className="w-4 h-4 sm:w-6 sm:h-6 text-white" />
            </div>
            <div className="min-w-0 flex-1">
              <h1 className="text-sm sm:text-lg font-semibold text-gray-900 truncate">
                {currentConversationIdLocal ? `Chat with ${appName}` : 'New Conversation'}
              </h1>
              <p className="text-xs text-gray-500 hidden sm:block">AI-powered document assistant</p>
            </div>
          </div>

          {/* Header action buttons — only shown when there are messages */}
          {messages.length > 0 && (
            <div className="flex items-center gap-1 sm:gap-2">
              {/* Mobile sources button — admin only */}
              {canSeeSources && sources.length > 0 && (
                <button
                  id="sources-drawer-button"
                  onClick={() => setShowSourcesDrawer(!showSourcesDrawer)}
                  className="lg:hidden flex items-center gap-1 px-2 sm:px-3 py-2 text-xs sm:text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors min-touch-target relative"
                >
                  <FileText className="w-4 h-4" />
                  <span className="hidden xs:inline">Sources</span>
                  <span className="absolute -top-1 -right-1 w-5 h-5 bg-blue-600 text-white text-xs rounded-full flex items-center justify-center">
                    {sources.length}
                  </span>
                </button>
              )}

              {/* Desktop sources toggle — admin only */}
              {canSeeSources && sources.length > 0 && (
                <button
                  onClick={() => setShowSourcesPanel(!showSourcesPanel)}
                  className="hidden lg:flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors min-touch-target"
                  title={showSourcesPanel ? 'Hide sources panel' : 'Show sources panel'}
                >
                  <FileText className="w-4 h-4" />
                  <span>{showSourcesPanel ? 'Hide' : 'Show'} Sources</span>
                  <span className="ml-1 px-1.5 py-0.5 bg-blue-100 text-blue-700 text-xs rounded-full font-semibold">
                    {sources.length}
                  </span>
                </button>
              )}

              {/* Analytics — only when conversation is persisted */}
              {currentConversationIdLocal && (
                <button
                  onClick={() => setShowAnalytics(true)}
                  className="hidden sm:flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 border border-gray-300 rounded-lg transition-colors min-touch-target"
                  title="View analytics"
                >
                  <BarChart3 className="w-4 h-4" />
                  <span className="hidden md:inline">Analytics</span>
                </button>
              )}

              {/* Export — always available when messages exist */}
              <ExportButton
                conversationId={currentConversationIdLocal || 'local'}
                messages={messages}
                onSelectiveExport={() => setShowSelectiveExport(true)}
              />

              {/* New chat */}
              <button
                onClick={startNewChat}
                className="flex items-center gap-1 sm:gap-2 px-2 sm:px-4 py-2 text-xs sm:text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors min-touch-target"
              >
                <Plus className="w-4 h-4" />
                <span className="hidden xs:inline">New</span>
              </button>
            </div>
          )}
        </div>

        {/* ── Messages ──────────────────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto scroll-smooth-touch">
          {/* Sticky context indicator */}
          {messages.length > 0 && contextSummary && (
            <div className="sticky top-0 z-10 pt-2 sm:pt-4 bg-gray-50">
              <ContextIndicator
                contextSummary={contextSummary}
                onClearContext={clearContext}
              />
            </div>
          )}

          <div className="max-w-6xl mx-auto px-3 sm:px-8 py-4 sm:py-8 space-y-4 sm:space-y-6">
            {/* Empty state */}
            {messages.length === 0 && (
              <div className="h-full flex items-center justify-center py-8 sm:py-12">
                <div className="text-center max-w-2xl px-4">
                  {currentLogo ? (
                    <div className="mx-auto mb-4 sm:mb-6 flex items-center justify-center">
                      <img
                        src={currentLogo}
                        alt={appName}
                        className="h-16 sm:h-20 w-auto object-contain max-w-[200px] sm:max-w-[280px]"
                        onError={e => {
                          e.currentTarget.style.display = 'none';
                          const parent = e.currentTarget.parentElement;
                          if (parent && !parent.querySelector('.logo-fallback')) {
                            const fb = document.createElement('div');
                            fb.className = 'logo-fallback w-16 h-16 sm:w-20 sm:h-20 bg-gradient-to-br from-blue-100 to-purple-100 rounded-2xl mx-auto flex items-center justify-center shadow-lg';
                            fb.innerHTML = `<span class="text-4xl sm:text-5xl">${appName.charAt(0)}</span>`;
                            parent.appendChild(fb);
                          }
                        }}
                      />
                    </div>
                  ) : (
                    <div className="w-16 h-16 sm:w-20 sm:h-20 bg-gradient-to-br from-blue-100 to-purple-100 rounded-2xl mx-auto mb-4 sm:mb-6 flex items-center justify-center shadow-lg">
                      <Sparkles className="w-8 h-8 sm:w-10 sm:h-10 text-blue-600" />
                    </div>
                  )}

                  <h2 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2 sm:mb-3">
                    Welcome to {appName}
                  </h2>
                  <p className="text-sm sm:text-base text-gray-600 mb-6 sm:mb-8 leading-relaxed">
                    {customization?.branding?.app_tagline ||
                      "I'm your AI assistant specialized in Finance and HRMS documents. Ask me anything about your uploaded documents, policies, or just have a conversation!"}
                  </p>

                  <div className="space-y-3">
                    <p className="text-xs sm:text-sm font-medium text-gray-700">Try asking:</p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      {exampleQuestions.map((example, idx) => (
                        <button
                          key={idx}
                          onClick={() => setInput(example)}
                          className="px-3 sm:px-4 py-2.5 sm:py-3 text-xs sm:text-sm text-left text-gray-700 bg-white hover:bg-gray-50 border border-gray-200 hover:border-blue-300 rounded-lg transition-all shadow-sm hover:shadow min-touch-target"
                        >
                          <FileText className="w-4 h-4 inline mr-2 text-gray-400" />
                          {example}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Message list */}
            {messages.map((message, index) => (
              <MemoizedMessageBubble key={`${message.timestamp}-${index}`} message={message} />
            ))}

            {/* Loading indicator */}
            {loading && (
              <div className="flex items-start gap-2 sm:gap-3">
                <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0 shadow-md">
                  <Sparkles className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
                </div>
                <div className="flex-1 bg-white rounded-2xl p-3 sm:p-4 shadow-sm border border-gray-200">
                  <div className="flex items-center gap-3 text-gray-500">
                    <Loader2 className="w-4 h-4 sm:w-5 sm:h-5 animate-spin text-blue-500" />
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                      <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                      <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Error banner */}
            {error && (
              <div className="flex items-start gap-2 sm:gap-3 p-3 sm:p-4 bg-red-50 border border-red-200 rounded-lg">
                <AlertCircle className="w-4 h-4 sm:w-5 sm:h-5 text-red-600 flex-shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs sm:text-sm font-medium text-red-900">Error</p>
                  <p className="text-xs sm:text-sm text-red-700 mt-1 break-words">{error}</p>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* ── Input Area ────────────────────────────────────────────────────── */}
        <div className="border-t border-gray-200 bg-white p-3 sm:p-4 shadow-lg flex-shrink-0">
          <div className="max-w-6xl mx-auto">

            {/* Suggested questions */}
            {suggestions.length > 0 && messages.length > 0 && (
              <div className="mb-3 animate-fadeIn">
                <div className="flex items-center mb-2">
                  <button
                    onClick={() => setShowSuggestions(v => !v)}
                    className="flex items-center gap-1 text-xs font-medium text-gray-600 hover:text-gray-800 transition-colors"
                  >
                    <Sparkles className="w-3 h-3 text-blue-500" />
                    Suggested questions
                    {showSuggestions
                      ? <ChevronDown className="w-3 h-3" />
                      : <ChevronUp className="w-3 h-3" />
                    }
                  </button>
                </div>
                {showSuggestions && (
                  <div className="flex flex-wrap gap-2">
                    {suggestions.map((suggestion, idx) => (
                      <button
                        key={idx}
                        onClick={() => handleSuggestionClick(suggestion)}
                        className="group px-2 sm:px-3 py-1.5 sm:py-2 text-xs sm:text-sm text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 hover:border-blue-300 rounded-lg transition-all shadow-sm hover:shadow min-touch-target"
                      >
                        <span className="flex items-center gap-1">
                          <span className="line-clamp-1">{suggestion}</span>
                          <Send className="w-3 h-3 opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* Focus context hint */}
            {contextSummary?.primary_document && (
              <div className="mb-2 px-3 py-1.5 bg-blue-50 border border-blue-200 rounded-md">
                <p className="text-xs text-blue-700 truncate">
                  <Sparkles className="w-3 h-3 inline mr-1" />
                  I'll focus on{' '}
                  <span className="font-medium">{contextSummary.primary_document}</span>{' '}
                  for your next question
                </p>
              </div>
            )}

            {/* Text input */}
            <div className="flex items-end gap-2">
              <div className="flex-1 relative min-w-0">
                <textarea
                  ref={inputRef}
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me anything about your documents..."
                  rows={1}
                  className="w-full px-3 sm:px-4 py-2.5 sm:py-3 pr-10 sm:pr-12 border border-gray-300 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none shadow-sm text-sm sm:text-base"
                  style={{ minHeight: '44px', maxHeight: '120px' }}
                  disabled={loading}
                />
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!input.trim() || loading}
                className="px-3 sm:px-4 py-2.5 sm:py-3 bg-gradient-to-r from-blue-600 to-blue-500 text-white rounded-xl hover:from-blue-700 hover:to-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all shadow-md hover:shadow-lg disabled:shadow-none min-touch-target flex-shrink-0"
              >
                {loading
                  ? <Loader2 className="w-5 h-5 animate-spin" />
                  : <Send className="w-5 h-5" />
                }
              </button>
            </div>
            <p className="mt-2 text-xs text-gray-500 text-center hidden sm:block">
              Press Enter to send • Shift+Enter for new line
            </p>
          </div>
        </div>
      </div>

      {/* ── Sources Sidebar (desktop) — admin only ───────────────────────── */}
      {canSeeSources && sources.length > 0 && showSourcesPanel && (
        <div className="hidden lg:block w-80 border-l border-gray-200 bg-white overflow-y-auto scroll-smooth-touch flex-shrink-0">
          <div className="p-4 border-b border-gray-200 bg-gray-50 sticky top-0 z-10">
            <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
              <FileText className="w-4 h-4 text-blue-600" />
              Sources ({sources.length})
            </h3>
            {contextSummary?.primary_document && (
              <p className="text-xs text-gray-500 mt-1 truncate">
                Currently focusing on:{' '}
                <span className="font-medium">{contextSummary.primary_document}</span>
              </p>
            )}
          </div>
          <div className="p-4 space-y-2">
            {sources.map((source, index) => (
              <MemoizedSourceCard key={`${source.chunk_id}-${index}`} source={source} />
            ))}
          </div>
        </div>
      )}

      {/* ── Sources Drawer (mobile) — admin only ────────────────────────── */}
      {canSeeSources && sources.length > 0 && (
        <div
          id="sources-drawer"
          className={`fixed bottom-0 left-0 right-0 z-50 bg-white transform transition-transform duration-300 ease-in-out lg:hidden border-t-2 border-gray-200 rounded-t-2xl shadow-2xl ${
            showSourcesDrawer ? 'translate-y-0' : 'translate-y-full'
          }`}
          style={{ maxHeight: '70vh' }}
        >
          <div className="p-4 border-b border-gray-200 bg-gray-50 flex items-center justify-between sticky top-0 z-10">
            <h3 className="text-sm font-semibold text-gray-900 flex items-center gap-2">
              <FileText className="w-4 h-4 text-blue-600" />
              Sources ({sources.length})
            </h3>
            <button
              onClick={() => setShowSourcesDrawer(false)}
              className="p-2 hover:bg-gray-200 rounded-lg transition-colors min-touch-target"
            >
              <X className="w-5 h-5 text-gray-600" />
            </button>
          </div>
          <div className="p-4 space-y-2 overflow-y-auto scroll-smooth-touch" style={{ maxHeight: 'calc(70vh - 60px)' }}>
            {sources.map((source, index) => (
              <MemoizedSourceCard key={`${source.chunk_id}-${index}`} source={source} />
            ))}
          </div>
        </div>
      )}

      {/* ── Modals ───────────────────────────────────────────────────────── */}
      {currentConversationIdLocal && (
        <AnalyticsModal
          conversationId={currentConversationIdLocal}
          isOpen={showAnalytics}
          onClose={() => setShowAnalytics(false)}
        />
      )}

      {currentConversationIdLocal && (
        <SelectiveExportModal
          isOpen={showSelectiveExport}
          onClose={() => setShowSelectiveExport(false)}
          conversationId={currentConversationIdLocal}
          messages={messages}
        />
      )}
    </div>
  );
}