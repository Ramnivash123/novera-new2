import { useState, useEffect, useRef, Fragment } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Search, 
  MessageSquare, 
  FileText, 
  History, 
  Upload, 
  User,
  Settings,
  BarChart3,
  X,
  Command,
  ArrowRight,
  Clock,
  TrendingUp
} from 'lucide-react';
import api from '../../services/api';
import { useAuth } from '../../contexts/AuthContext';

interface CommandItem {
  id: string;
  label: string;
  description?: string;
  icon: any;
  action: () => void;
  category: 'actions' | 'documents' | 'conversations' | 'navigation' | 'recent';
  keywords?: string[];
}

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function CommandPalette({ isOpen, onClose }: CommandPaletteProps) {
  const navigate = useNavigate();
  const { isAdmin } = useAuth();
  const [search, setSearch] = useState('');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<any[]>([]);
  const [conversations, setConversations] = useState<any[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Load data when palette opens
  useEffect(() => {
    if (isOpen) {
      loadData();
      inputRef.current?.focus();
      setSearch('');
      setSelectedIndex(0);
    }
  }, [isOpen]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [docs, convs] = await Promise.all([
        api.getDocuments({ limit: 10 }),
        api.getConversations(10)
      ]);
      setDocuments(docs.documents || []);
      setConversations(convs || []);
    } catch (error) {
      console.error('Failed to load command palette data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Define actions
  const actions: CommandItem[] = [
    {
      id: 'new-chat',
      label: 'Start New Chat',
      description: 'Begin a new conversation',
      icon: MessageSquare,
      action: () => {
        navigate('/chat');
        onClose();
      },
      category: 'actions',
      keywords: ['new', 'chat', 'conversation', 'start']
    },
    {
      id: 'upload-doc',
      label: 'Upload Document',
      description: 'Upload a new document',
      icon: Upload,
      action: () => {
        navigate('/documents');
        onClose();
        // Trigger upload modal after navigation
        setTimeout(() => {
          const uploadBtn = document.querySelector('[data-upload-trigger]');
          if (uploadBtn instanceof HTMLElement) uploadBtn.click();
        }, 100);
      },
      category: 'actions',
      keywords: ['upload', 'document', 'file', 'add']
    },
    {
      id: 'view-profile',
      label: 'View Profile',
      description: 'Go to your profile settings',
      icon: User,
      action: () => {
        navigate('/profile');
        onClose();
      },
      category: 'navigation',
      keywords: ['profile', 'settings', 'account']
    },
  ];

  // Add admin actions
  if (isAdmin) {
    actions.push(
      {
        id: 'admin-dashboard',
        label: 'Admin Dashboard',
        description: 'View system statistics',
        icon: BarChart3,
        action: () => {
          navigate('/admin');
          onClose();
        },
        category: 'navigation',
        keywords: ['admin', 'dashboard', 'stats']
      },
      {
        id: 'manage-users',
        label: 'Manage Users',
        description: 'User management',
        icon: User,
        action: () => {
          navigate('/admin/users');
          onClose();
        },
        category: 'navigation',
        keywords: ['users', 'admin', 'manage']
      },
      {
        id: 'customization',
        label: 'Customization',
        description: 'Customize branding',
        icon: Settings,
        action: () => {
          navigate('/admin/customization');
          onClose();
        },
        category: 'navigation',
        keywords: ['customize', 'branding', 'theme']
      }
    );
  }

  // Convert documents to command items
  const documentItems: CommandItem[] = documents.map(doc => ({
    id: `doc-${doc.id}`,
    label: doc.filename,
    description: `${doc.doc_type} - ${doc.total_pages} pages`,
    icon: FileText,
    action: () => {
      navigate(`/documents/${doc.id}/edit`);
      onClose();
    },
    category: 'documents',
    keywords: [doc.filename, doc.doc_type, doc.department].filter(Boolean)
  }));

  // Convert conversations to command items
  const conversationItems: CommandItem[] = conversations.map(conv => {
    const preview = conv.messages?.[0]?.content?.substring(0, 50) || 'New conversation';
    return {
      id: `conv-${conv.id}`,
      label: preview,
      description: `${conv.messages?.length || 0} messages`,
      icon: History,
      action: () => {
        navigate(`/chat/${conv.id}`);
        onClose();
      },
      category: 'conversations',
      keywords: [preview]
    };
  });

  // Combine all items
  const allItems = [...actions, ...documentItems, ...conversationItems];

  // Filter items based on search
  const filteredItems = search.trim() === '' 
    ? allItems 
    : allItems.filter(item => {
        const searchLower = search.toLowerCase();
        return (
          item.label.toLowerCase().includes(searchLower) ||
          item.description?.toLowerCase().includes(searchLower) ||
          item.keywords?.some(k => k.toLowerCase().includes(searchLower))
        );
      });

  // Group items by category
  const groupedItems = filteredItems.reduce((acc, item) => {
    if (!acc[item.category]) acc[item.category] = [];
    acc[item.category].push(item);
    return acc;
  }, {} as Record<string, CommandItem[]>);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => Math.min(prev + 1, filteredItems.length - 1));
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => Math.max(prev - 1, 0));
      } else if (e.key === 'Enter') {
        e.preventDefault();
        filteredItems[selectedIndex]?.action();
      } else if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, selectedIndex, filteredItems, onClose]);

  // Scroll selected item into view
  useEffect(() => {
    if (listRef.current) {
      const selectedElement = listRef.current.querySelector(`[data-index="${selectedIndex}"]`);
      selectedElement?.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }, [selectedIndex]);

  if (!isOpen) return null;

  const categoryLabels = {
    actions: 'Actions',
    navigation: 'Navigation',
    documents: 'Documents',
    conversations: 'Recent Conversations',
    recent: 'Recent'
  };

  return (
    <div className="fixed inset-0 z-[100] flex items-start justify-center pt-[10vh] px-4">
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm animate-fadeIn"
        onClick={onClose}
      />

      {/* Palette */}
      <div className="relative w-full max-w-2xl bg-white rounded-xl shadow-2xl overflow-hidden animate-slideUp">
        {/* Search Input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-gray-200">
          <Search className="w-5 h-5 text-gray-400 flex-shrink-0" />
          <input
            ref={inputRef}
            type="text"
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setSelectedIndex(0);
            }}
            placeholder="Search for actions, documents, conversations..."
            className="flex-1 bg-transparent border-none outline-none text-gray-900 placeholder-gray-400 text-sm sm:text-base"
          />
          <div className="flex items-center gap-2">
            <kbd className="hidden sm:inline-block px-2 py-1 text-xs font-semibold text-gray-600 bg-gray-100 border border-gray-200 rounded">
              ESC
            </kbd>
            <button
              onClick={onClose}
              className="p-1 hover:bg-gray-100 rounded lg:hidden"
            >
              <X className="w-4 h-4 text-gray-500" />
            </button>
          </div>
        </div>

        {/* Results */}
        <div 
          ref={listRef}
          className="max-h-[60vh] overflow-y-auto scroll-smooth-touch"
        >
          {loading ? (
            <div className="p-8 text-center">
              <div className="inline-block w-6 h-6 border-2 border-gray-300 border-t-primary-500 rounded-full animate-spin" />
            </div>
          ) : filteredItems.length === 0 ? (
            <div className="p-8 text-center">
              <p className="text-sm text-gray-500">No results found for "{search}"</p>
            </div>
          ) : (
            <div className="py-2">
              {Object.entries(groupedItems).map(([category, items]) => (
                <div key={category} className="mb-4 last:mb-0">
                  <div className="px-4 py-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
                    {categoryLabels[category as keyof typeof categoryLabels]}
                  </div>
                  {items.map((item, idx) => {
                    const globalIndex = filteredItems.indexOf(item);
                    const isSelected = globalIndex === selectedIndex;
                    
                    return (
                      <button
                        key={item.id}
                        data-index={globalIndex}
                        onClick={item.action}
                        className={`w-full flex items-center gap-3 px-4 py-3 transition-colors ${
                          isSelected 
                            ? 'bg-primary-50 border-l-2 border-primary-500' 
                            : 'hover:bg-gray-50 border-l-2 border-transparent'
                        }`}
                      >
                        <div className={`p-2 rounded-lg ${
                          isSelected ? 'bg-primary-100' : 'bg-gray-100'
                        }`}>
                          <item.icon className={`w-4 h-4 ${
                            isSelected ? 'text-primary-600' : 'text-gray-600'
                          }`} />
                        </div>
                        <div className="flex-1 text-left min-w-0">
                          <p className={`text-sm font-medium truncate ${
                            isSelected ? 'text-primary-900' : 'text-gray-900'
                          }`}>
                            {item.label}
                          </p>
                          {item.description && (
                            <p className="text-xs text-gray-500 truncate">
                              {item.description}
                            </p>
                          )}
                        </div>
                        {isSelected && (
                          <ArrowRight className="w-4 h-4 text-primary-500 flex-shrink-0" />
                        )}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="hidden sm:flex items-center justify-between px-4 py-2 border-t border-gray-200 bg-gray-50 text-xs text-gray-500">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1">
              <kbd className="px-2 py-1 bg-white border border-gray-200 rounded">↑</kbd>
              <kbd className="px-2 py-1 bg-white border border-gray-200 rounded">↓</kbd>
              Navigate
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-2 py-1 bg-white border border-gray-200 rounded">↵</kbd>
              Select
            </span>
            <span className="flex items-center gap-1">
              <kbd className="px-2 py-1 bg-white border border-gray-200 rounded">ESC</kbd>
              Close
            </span>
          </div>
          <div className="flex items-center gap-1 text-gray-400">
            <Command className="w-3 h-3" />
            <span>Tip: Press Ctrl+K anytime</span>
          </div>
        </div>
      </div>
    </div>
  );
}