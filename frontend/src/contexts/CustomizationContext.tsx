import { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import api from '../services/api';
import type { Customization } from '../types';
import { ThemeValidator } from '../utils/themeValidator';

interface CustomizationContextType {
  customization: Customization | null;
  loading: boolean;
  darkMode: boolean;
  toggleDarkMode: () => void;
  refreshCustomization: () => Promise<void>;
  applyTheme: (customization: Customization, forceDarkMode?: boolean) => void;
  validateTheme: (formData: any) => any;
  generateDarkMode: (lightColors: any) => any;
}

const CustomizationContext = createContext<CustomizationContextType | undefined>(undefined);

export function CustomizationProvider({ children }: { children: ReactNode }) {
  const [customization, setCustomization] = useState<Customization | null>(null);
  const [loading, setLoading] = useState(true);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    // Load saved dark mode preference FIRST
    const savedDarkMode = localStorage.getItem('darkMode') === 'true';
    setDarkMode(savedDarkMode);
    
    if (savedDarkMode) {
      document.documentElement.classList.add('dark');
    }

    // Then load customization
    loadCustomization();
    
    // Poll for updates every 30 seconds
    const interval = setInterval(() => {
      loadCustomization(true);
    }, 30000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  const loadCustomization = async (silent = false) => {
    if (!silent) setLoading(true);
    
    try {
      console.log('Loading customization from API...');
      const data = await api.getCurrentCustomization();
      console.log('Customization loaded:', data);
      
      setCustomization(data);
      
      // CRITICAL: Use current darkMode state, not localStorage
      // This prevents polling from overwriting user's toggle
      const currentDarkMode = localStorage.getItem('darkMode') === 'true';
      applyTheme(data, currentDarkMode);
      
      if (silent) {
        console.log('🔄 Background update - preserving dark mode state:', currentDarkMode);
      }
    } catch (error) {
      console.error('Failed to load customization:', error);
    } finally {
      if (!silent) setLoading(false);
    }
  };

  const refreshCustomization = async () => {
    await loadCustomization();
  };

  const toggleDarkMode = () => {
    setDarkMode(prev => {
      const newValue = !prev;
      localStorage.setItem('darkMode', String(newValue));
      
      if (newValue) {
        document.documentElement.classList.add('dark');
      } else {
        document.documentElement.classList.remove('dark');
      }
      
      if (customization) {
        applyTheme(customization, newValue);
      }
      
      console.log('🌙 Dark mode toggled:', newValue);
      
      return newValue;
    });
  };

  const applyTheme = (customization: Customization, forceDarkMode?: boolean) => {
    console.log('Applying comprehensive theme...');
    const root = document.documentElement;
    const isDark = forceDarkMode !== undefined ? forceDarkMode : darkMode;

    // Determine which colors to use
    let colors = customization.colors;
    
    if (isDark && customization.dark_mode.enabled && customization.dark_mode.colors) {
      const darkColors = customization.dark_mode.colors;
      
      // Merge dark colors with light colors as fallback
      colors = {
        primary: customization.colors.primary, // Keep brand colors
        secondary: customization.colors.secondary,
        accent: customization.colors.accent,
        success: customization.colors.success,
        warning: customization.colors.warning,
        error: customization.colors.error,
        info: customization.colors.info,
        background: darkColors.background || '#1f2937',
        background_secondary: darkColors.background_secondary || '#111827',
        background_tertiary: darkColors.background_tertiary || '#0f172a',
        sidebar: darkColors.background || '#1f2937',
        text_primary: darkColors.text_primary || '#f9fafb',
        text_secondary: darkColors.text_secondary || '#d1d5db',
        border: darkColors.border || '#374151',
        shadow: darkColors.shadow || 'rgba(0, 0, 0, 0.4)',
      };
      
      console.log('🌙 Using dark mode colors:', darkColors);
    }

    // Apply colors
    root.style.setProperty('--color-primary', colors.primary);
    root.style.setProperty('--color-secondary', colors.secondary);
    root.style.setProperty('--color-accent', colors.accent);
    
    root.style.setProperty('--color-success', colors.success);
    root.style.setProperty('--color-warning', colors.warning);
    root.style.setProperty('--color-error', colors.error);
    root.style.setProperty('--color-info', colors.info);
    
    root.style.setProperty('--color-background', colors.background);
    root.style.setProperty('--color-background-secondary', colors.background_secondary);
    root.style.setProperty('--color-background-tertiary', colors.background_tertiary);
    root.style.setProperty('--color-sidebar', colors.sidebar);

    root.style.setProperty('--nav-text-color', isDark ? colors.text_secondary : customization.navigation.text_color);
    root.style.setProperty('--nav-active-color', customization.navigation.active_color);
    root.style.setProperty('--nav-hover-color', isDark 
      ? `rgba(${hexToRgb(customization.colors.primary)?.r || 14}, ${hexToRgb(customization.colors.primary)?.g || 165}, ${hexToRgb(customization.colors.primary)?.b || 233}, 0.15)`
      : customization.navigation.hover_color
    );
    
    root.style.setProperty('--color-text-primary', colors.text_primary);
    root.style.setProperty('--color-text-secondary', colors.text_secondary);
    
    root.style.setProperty('--color-border', colors.border);
    root.style.setProperty('--color-shadow', colors.shadow);

    // Apply component styling (these should adapt in dark mode)
    root.style.setProperty('--button-primary-bg', customization.buttons.primary_color);
    root.style.setProperty('--button-primary-text', customization.buttons.primary_text);
    root.style.setProperty('--button-secondary-bg', isDark ? colors.background_tertiary : customization.buttons.secondary_color);
    root.style.setProperty('--button-secondary-text', isDark ? colors.text_primary : customization.buttons.secondary_text);
    root.style.setProperty('--button-border-radius', customization.buttons.border_radius);

    root.style.setProperty('--input-border-color', isDark ? colors.border : customization.inputs.border_color);
    root.style.setProperty('--input-focus-color', customization.inputs.focus_color);
    root.style.setProperty('--input-border-radius', customization.inputs.border_radius);

    root.style.setProperty('--card-background', isDark ? colors.background_secondary : customization.cards.background);
    root.style.setProperty('--card-border-color', isDark ? colors.border : customization.cards.border_color);
    root.style.setProperty('--card-border-radius', customization.cards.border_radius);
    root.style.setProperty('--card-shadow', customization.cards.shadow);

    root.style.setProperty('--nav-background', customization.navigation.background);
    root.style.setProperty('--nav-text-color', customization.navigation.text_color);
    root.style.setProperty('--nav-active-color', customization.navigation.active_color);
    root.style.setProperty('--nav-hover-color', customization.navigation.hover_color);

    // Apply typography
    if (customization.typography.font_family) {
      root.style.setProperty('--font-family', customization.typography.font_family);
      document.body.style.fontFamily = customization.typography.font_family;
    }
    root.style.setProperty('--font-size-base', customization.typography.font_size_base);
    root.style.setProperty('--font-size-heading', customization.typography.font_size_heading);
    root.style.setProperty('--font-weight-normal', customization.typography.font_weight_normal);
    root.style.setProperty('--font-weight-medium', customization.typography.font_weight_medium);
    root.style.setProperty('--font-weight-bold', customization.typography.font_weight_bold);
    root.style.setProperty('--line-height-base', customization.typography.line_height_base);
    root.style.setProperty('--line-height-heading', customization.typography.line_height_heading);
    root.style.setProperty('--letter-spacing', customization.typography.letter_spacing);

    // Apply layout
    root.style.setProperty('--border-radius', customization.layout.border_radius);
    root.style.setProperty('--spacing-unit', customization.layout.spacing_unit);
    root.style.setProperty('--spacing-xs', customization.layout.spacing_xs);
    root.style.setProperty('--spacing-sm', customization.layout.spacing_sm);
    root.style.setProperty('--spacing-md', customization.layout.spacing_md);
    root.style.setProperty('--spacing-lg', customization.layout.spacing_lg);
    root.style.setProperty('--spacing-xl', customization.layout.spacing_xl);

    // Apply animations
    root.style.setProperty('--animation-speed', customization.animations.speed);
    if (!customization.animations.enabled) {
      root.style.setProperty('--animation-speed', '0ms');
    }

    // Generate color shades
    const rgb = hexToRgb(customization.colors.primary);
    if (rgb) {
      root.style.setProperty('--color-primary-rgb', `${rgb.r}, ${rgb.g}, ${rgb.b}`);
      generateColorShades(customization.colors.primary, 'primary');
      generateColorShades(customization.colors.secondary, 'secondary');
      generateColorShades(customization.colors.accent, 'accent');
    }

    // Apply custom CSS
    if (customization.advanced.custom_css) {
      applyCustomCSS(customization.advanced.custom_css);
    }

    // Update favicon
    if (customization.branding.favicon_url) {
      const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const fullFaviconUrl = customization.branding.favicon_url.startsWith('http')
        ? customization.branding.favicon_url
        : `${API_BASE}${customization.branding.favicon_url}`;
      updateFavicon(fullFaviconUrl);
    }

    // Update document title
    if (customization.branding.app_name) {
      document.title = customization.branding.app_name;
    }

    console.log('Theme application complete! Dark mode:', isDark);
  };

  const validateTheme = (formData: any) => {
    return ThemeValidator.validateTheme(formData);
  };

  const generateDarkMode = (lightColors: any) => {
    return ThemeValidator.generateDarkModeColors(lightColors);
  };

  const value = {
    customization,
    loading,
    darkMode,
    toggleDarkMode,
    refreshCustomization,
    applyTheme,
    validateTheme,
    generateDarkMode,
  };

  return (
    <CustomizationContext.Provider value={value}>
      {children}
    </CustomizationContext.Provider>
  );
}

export function useCustomization() {
  const context = useContext(CustomizationContext);
  if (context === undefined) {
    throw new Error('useCustomization must be used within a CustomizationProvider');
  }
  return context;
}

// Helper functions
function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16),
      }
    : null;
}

function generateColorShades(baseColor: string, name: string) {
  const root = document.documentElement;
  const rgb = hexToRgb(baseColor);
  
  if (!rgb) return;

  const shades = [
    { name: '50', lightness: 0.95 },
    { name: '100', lightness: 0.9 },
    { name: '200', lightness: 0.8 },
    { name: '300', lightness: 0.6 },
    { name: '400', lightness: 0.4 },
    { name: '500', lightness: 0 },
    { name: '600', lightness: -0.1 },
    { name: '700', lightness: -0.2 },
    { name: '800', lightness: -0.3 },
    { name: '900', lightness: -0.4 },
  ];

  shades.forEach(shade => {
    const adjusted = adjustLightness(rgb, shade.lightness);
    root.style.setProperty(
      `--color-${name}-${shade.name}`,
      `rgb(${adjusted.r}, ${adjusted.g}, ${adjusted.b})`
    );
  });
}

function adjustLightness(
  rgb: { r: number; g: number; b: number },
  amount: number
): { r: number; g: number; b: number } {
  const adjust = (value: number) => {
    if (amount > 0) {
      return Math.round(value + (255 - value) * amount);
    } else {
      return Math.round(value * (1 + amount));
    }
  };

  return {
    r: Math.max(0, Math.min(255, adjust(rgb.r))),
    g: Math.max(0, Math.min(255, adjust(rgb.g))),
    b: Math.max(0, Math.min(255, adjust(rgb.b))),
  };
}

function applyCustomCSS(css: string) {
  const styleId = 'custom-theme-css';
  let styleElement = document.getElementById(styleId);
  
  if (!styleElement) {
    styleElement = document.createElement('style');
    styleElement.id = styleId;
    document.head.appendChild(styleElement);
  }
  
  styleElement.textContent = css;
}

function updateFavicon(faviconUrl: string) {
  const link = document.querySelector("link[rel~='icon']") as HTMLLinkElement;
  if (link) {
    link.href = faviconUrl;
  } else {
    const newLink = document.createElement('link');
    newLink.rel = 'icon';
    newLink.href = faviconUrl;
    document.head.appendChild(newLink);
  }
}