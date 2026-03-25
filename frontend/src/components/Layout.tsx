import { ReactNode, useEffect } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { 
  MessageSquare, 
  FileText, 
  History, 
  User, 
  LogOut, 
  Menu, 
  X,
  Shield,
  Users,
  LayoutDashboard,
  Palette,
  Moon,
  Sun
} from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { useConversation } from '../contexts/ConversationContext';
import { useCustomization } from '../contexts/CustomizationContext';
import TokenMeter from './chat/TokenMeter';

interface LayoutProps {
  children: ReactNode;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

function getFullImageUrl(path: string | null): string {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
  return `${API_BASE}${path}`;
}

export default function Layout({ children, sidebarOpen, setSidebarOpen }: LayoutProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout, isAdmin } = useAuth();
  const { currentConversationId } = useConversation();
  const { customization, darkMode, toggleDarkMode } = useCustomization();

  useEffect(() => {
    if (window.innerWidth < 1024) {
      setSidebarOpen(false);
    }
  }, [location.pathname, setSidebarOpen]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (window.innerWidth >= 1024) return;
      
      const sidebar = document.getElementById('mobile-sidebar');
      const menuButton = document.getElementById('mobile-menu-button');
      
      if (
        sidebarOpen &&
        sidebar &&
        !sidebar.contains(event.target as Node) &&
        menuButton &&
        !menuButton.contains(event.target as Node)
      ) {
        setSidebarOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [sidebarOpen, setSidebarOpen]);

  useEffect(() => {
    if (window.innerWidth < 1024) {
      if (sidebarOpen) {
        document.body.style.overflow = 'hidden';
      } else {
        document.body.style.overflow = 'unset';
      }
    }
    return () => {
      document.body.style.overflow = 'unset';
    };
  }, [sidebarOpen]);

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const isActive = (path: string) => {
    return location.pathname === path || location.pathname.startsWith(path + '/');
  };

  const navItems = [
    { path: '/chat', icon: MessageSquare, label: 'Chat' },
    { path: '/documents', icon: FileText, label: 'Documents' },
    { path: '/conversations', icon: History, label: 'Conversations' },
  ];

  const adminNavItems = [
    { path: '/admin', icon: LayoutDashboard, label: 'Dashboard' },
    { path: '/admin/users', icon: Users, label: 'User Management' },
    { path: '/admin/customization', icon: Palette, label: 'Customization' },
  ];

  const showTokenMeter = currentConversationId && (
    location.pathname.startsWith('/chat') || 
    location.pathname.startsWith('/conversations')
  );

  const appName = customization?.branding?.app_name || 'Novera';
  const logoUrl = customization?.branding?.logo_url ? getFullImageUrl(customization.branding.logo_url) : null;

  return (
    <div className="flex h-screen bg-theme overflow-hidden">
      {/* Mobile Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 lg:hidden transition-opacity duration-300"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}
      {/* Sidebar */}
      <aside
        id="mobile-sidebar"
        className={`
          fixed lg:static inset-y-0 left-0 z-50
          w-64 border-r border-theme
          transform transition-theme duration-300 ease-in-out
          flex flex-col
          bg-sidebar
          ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        {/* Logo */}
        <div className="h-14 sm:h-16 flex items-center justify-between px-4 sm:px-6 border-b border-theme flex-shrink-0">
          <div className="flex items-center space-x-2 min-w-0">
            {logoUrl ? (
              <img 
                src={logoUrl} 
                alt={appName}
                className="h-6 sm:h-8 w-auto object-contain max-w-[100px] sm:max-w-[120px]"
                onError={(e) => {
                  console.error('Failed to load logo:', logoUrl);
                  const parent = e.currentTarget.parentElement;
                  if (parent) {
                    e.currentTarget.style.display = 'none';
                    if (!parent.querySelector('.logo-fallback')) {
                      const fallback = document.createElement('div');
                      fallback.className = 'logo-fallback w-6 h-6 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center';
                      fallback.style.background = `linear-gradient(to bottom right, ${customization?.colors?.primary || '#0ea5e9'}, ${customization?.colors?.secondary || '#d946ef'})`;
                      fallback.innerHTML = `<span class="text-white font-bold text-sm sm:text-base">${appName.charAt(0)}</span>`;
                      parent.insertBefore(fallback, e.currentTarget);
                    }
                  }
                }}
              />
            ) : (
              <div 
                className="w-6 h-6 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                style={{
                  background: `linear-gradient(to bottom right, ${customization?.colors?.primary || '#0ea5e9'}, ${customization?.colors?.secondary || '#d946ef'})`
                }}
              >
                <span className="text-white font-bold text-sm sm:text-base">{appName.charAt(0)}</span>
              </div>
            )}
            <span className="font-semibold text-sm sm:text-base truncate text-theme-primary">
              {appName}
            </span>
          </div>

          <button
            onClick={() => setSidebarOpen(false)}
            className="lg:hidden p-2 -mr-2 rounded-lg hover:bg-theme-secondary transition-colors min-touch-target"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5 text-theme-secondary" />
          </button>
        </div>
        {/* Navigation */}
        <nav className="flex-1 px-3 sm:px-4 py-4 sm:py-6 space-y-1 overflow-y-auto scroll-smooth-touch">
          {/* Main Navigation */}
          <div className="mb-6">
            <p className="px-3 text-xs font-semibold uppercase tracking-wider mb-2 text-theme-secondary">
              Main
            </p>
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`
                  nav-link-theme flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-theme
                  min-touch-target
                  ${isActive(item.path) ? 'active' : ''}
                `}
              >
                <item.icon className="w-5 h-5 flex-shrink-0" />
                <span className="font-medium text-sm sm:text-base">{item.label}</span>
              </Link>
            ))}
          </div>
          {/* Admin Navigation */}
          {isAdmin && (
            <div className="mb-6">
              <p className="px-3 text-xs font-semibold uppercase tracking-wider mb-2 flex items-center gap-1 text-theme-secondary">
                <Shield className="w-3 h-3" />
                Admin
              </p>
              {adminNavItems.map((item) => (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`
                    nav-link-theme flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-theme
                    min-touch-target
                    ${isActive(item.path) ? 'active' : ''}
                  `}
                >
                  <item.icon className="w-5 h-5 flex-shrink-0" />
                  <span className="font-medium text-sm sm:text-base">{item.label}</span>
                </Link>
              ))}
            </div>
          )}
          {/* Account Navigation */}
          <div>
            <p className="px-3 text-xs font-semibold uppercase tracking-wider mb-2 text-theme-secondary">
              Account
            </p>
            <Link
              to="/profile"
              className={`
                nav-link-theme flex items-center space-x-3 px-3 py-2.5 rounded-lg transition-theme
                min-touch-target
                ${isActive('/profile') ? 'active' : ''}
              `}
            >
              <User className="w-5 h-5 flex-shrink-0" />
              <span className="font-medium text-sm sm:text-base">Profile</span>
            </Link>
          </div>
        </nav>
        {/* Token Meter */}
        {showTokenMeter && (
          <div className="px-3 sm:px-4 pb-3 border-t border-theme pt-3 flex-shrink-0">
            <TokenMeter 
              conversationId={currentConversationId} 
              isVisible={true}
            />
          </div>
        )}
        {/* User Info & Logout */}
        <div className="border-t border-theme p-3 sm:p-4 flex-shrink-0">
          <div className="flex items-center space-x-3 mb-3">
            <div 
              className="w-9 h-9 sm:w-10 sm:h-10 rounded-full flex items-center justify-center flex-shrink-0"
              style={{
                background: `linear-gradient(to bottom right, ${customization?.colors?.primary || '#0ea5e9'}, ${customization?.colors?.secondary || '#d946ef'})`
              }}
            >
              <span className="text-white font-bold text-xs sm:text-sm">
                {user?.username?.charAt(0).toUpperCase()}
              </span>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-xs sm:text-sm font-medium truncate text-theme-primary">
                {user?.full_name || user?.username}
              </p>
              <p className="text-xs truncate text-theme-secondary">
                {user?.email}
              </p>
              {isAdmin && (
                <span 
                  className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium mt-1"
                  style={{
                    backgroundColor: darkMode ? 'rgba(243, 232, 255, 0.2)' : '#f3e8ff',
                    color: darkMode ? '#e9d5ff' : '#7e22ce'
                  }}
                >
                  <Shield className="w-3 h-3 mr-1" />
                  Admin
                </span>
              )}
            </div>
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 text-sm text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 active:bg-red-100 dark:active:bg-red-900/30 rounded-lg transition-colors min-touch-target"
          >
            <LogOut className="w-4 h-4" />
            <span>Logout</span>
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden min-w-0">
        {/* Top Bar */}
        <header className="h-14 sm:h-16 bg-theme border-b border-theme flex items-center justify-between px-4 sm:px-6 flex-shrink-0">
          <button
            id="mobile-menu-button"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 -ml-2 rounded-lg hover:bg-theme-secondary active:bg-theme-tertiary transition-colors min-touch-target tap-highlight-transparent"
            aria-label={sidebarOpen ? 'Close menu' : 'Open menu'}
          >
            {sidebarOpen ? (
              <X className="w-5 h-5 text-theme-secondary" />
            ) : (
              <Menu className="w-5 h-5 text-theme-secondary" />
            )}
          </button>

          <div className="flex items-center space-x-2 sm:space-x-4 min-w-0">
            {/* Dark Mode Toggle */}
            {customization?.dark_mode?.enabled && (
              <div className="flex items-center gap-2">
                <button
                  onClick={toggleDarkMode}
                  className="p-2 rounded-lg hover:bg-theme-secondary transition-colors min-touch-target relative group"
                  title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                  aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  <div className="relative">
                    {darkMode ? (
                      <Sun className="w-5 h-5 text-yellow-500" />
                    ) : (
                      <Moon className="w-5 h-5 text-theme-secondary" />
                    )}
                    {/* Tooltip */}
                    <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 text-xs bg-gray-900 dark:bg-gray-100 text-white dark:text-gray-900 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none">
                      {darkMode ? 'Light mode' : 'Dark mode'}
                    </span>
                  </div>
                </button>
                {/* Optional: Show current mode text on desktop */}
                <span className="hidden md:inline text-xs text-theme-secondary">
                  {darkMode ? 'Dark' : 'Light'}
                </span>
              </div>
            )}
            <span className="text-xs sm:text-sm truncate text-theme-secondary">
              <span className="hidden sm:inline">Welcome, </span>
              <span className="font-medium text-theme-primary">
                {user?.username}
              </span>
            </span>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-hidden">
          {children}
        </main>
      </div>
    </div>
  );
}