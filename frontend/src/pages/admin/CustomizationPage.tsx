import { useState, useEffect } from 'react';
import { 
  Palette, Upload, Save, RotateCcw, Eye, X, ChevronDown, ChevronUp,
  Sparkles, Type, Layout as LayoutIcon, Package, Settings, Moon,
  BookOpen, AlertCircle, CheckCircle, Download, Upload as UploadIcon,
  Lightbulb, Zap
} from 'lucide-react';
import Editor from '@monaco-editor/react';
import api from '../../services/api';
import { useCustomization } from '../../contexts/CustomizationContext';
import { toast } from '../../utils/toast';
import type { CustomizationUpdateRequest, ThemePreset } from '../../types';
import ImagePreviewModal from '../../components/admin/ImagePreviewModal';
import { CSS_TEMPLATES, CSSTemplate } from '../../utils/cssTemplates';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

function getFullImageUrl(path: string | null): string {
  if (!path) return '';
  if (path.startsWith('http')) return path;
  return `${API_BASE}${path}`;
}

type SectionKey = 'branding' | 'colors' | 'typography' | 'layout' | 'components' | 'darkmode' | 'advanced';

interface DarkModeColors {
  background: string;
  background_secondary: string;
  background_tertiary: string;
  text_primary: string;
  text_secondary: string;
  border: string;
  shadow: string;
}

export default function CustomizationPage() {
  const { refreshCustomization, validateTheme, generateDarkMode} = useCustomization();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [showPreview, setShowPreview] = useState(true);
  const [activePreset, setActivePreset] = useState<string | null>(null);
  const [presets, setPresets] = useState<ThemePreset[]>([]);
  const [validationResults, setValidationResults] = useState<any>(null);
  const [showTemplates, setShowTemplates] = useState(false);

  const [expandedSections, setExpandedSections] = useState<Record<SectionKey, boolean>>({
    branding: true,
    colors: false,
    typography: false,
    layout: false,
    components: false,
    darkmode: false,
    advanced: false,
  });

  const organizationName = 'default';
  
  const [formData, setFormData] = useState({
    app_name: '',
    app_tagline: '',
    primary_color: '#0ea5e9',
    secondary_color: '#d946ef',
    accent_color: '#8b5cf6',
    success_color: '#10b981',
    warning_color: '#f59e0b',
    error_color: '#ef4444',
    info_color: '#3b82f6',
    background_color: '#ffffff',
    background_secondary: '#f9fafb',
    background_tertiary: '#f3f4f6',
    sidebar_color: '#ffffff',
    text_primary_color: '#111827',
    text_secondary_color: '#6b7280',
    border_color: '#e5e7eb',
    shadow_color: '#00000010',
    button_primary_color: '#0ea5e9',
    button_text_color: '#ffffff',
    button_secondary_color: '#f3f4f6',
    button_secondary_text: '#374151',
    button_border_radius: '8px',
    input_border_color: '#d1d5db',
    input_focus_color: '#0ea5e9',
    input_border_radius: '8px',
    card_background: '#ffffff',
    card_border_color: '#e5e7eb',
    card_border_radius: '12px',
    card_shadow: '0 1px 3px rgba(0,0,0,0.1)',
    nav_background: '#ffffff',
    nav_text_color: '#6b7280',
    nav_active_color: '#0ea5e9',
    nav_hover_color: '#0ea5e910',
    font_family: '',
    font_size_base: '14px',
    font_size_heading: '24px',
    font_weight_normal: '400',
    font_weight_medium: '500',
    font_weight_bold: '700',
    line_height_base: '1.5',
    line_height_heading: '1.2',
    letter_spacing: '0',
    border_radius: '8px',
    spacing_unit: '16px',
    spacing_xs: '4px',
    spacing_sm: '8px',
    spacing_md: '16px',
    spacing_lg: '24px',
    spacing_xl: '32px',
    animation_speed: '300ms',
    enable_animations: true,
    dark_mode_enabled: false,
    custom_css: '',
    theme_name: '',
    theme_description: '',
  });

  const [darkModeColors, setDarkModeColors] = useState<DarkModeColors>({
    background: '#1f2937',
    background_secondary: '#111827',
    background_tertiary: '#0f172a',
    text_primary: '#f9fafb',
    text_secondary: '#d1d5db',
    border: '#374151',
    shadow: '#00000040',
  });

  const [logos, setLogos] = useState({
    light: null as string | null,
    dark: null as string | null,
    favicon: null as string | null,
  });

  const [uploadingLogo, setUploadingLogo] = useState<string | null>(null);

  const [previewModal, setPreviewModal] = useState({
    isOpen: false,
    file: null as File | null,
    previewUrl: null as string | null,
    logoType: 'light' as 'light' | 'dark' | 'favicon',
  });

  useEffect(() => {
    loadCustomization();
    loadPresets();
  }, []);

  useEffect(() => {
    const results = validateTheme(formData);
    setValidationResults(results);
  }, [formData]);

  const loadCustomization = async () => {
    setLoading(true);
    try {
      const data = await api.getAdminCustomization(organizationName);
      
      setFormData({
        app_name: data.branding.app_name || '',
        app_tagline: data.branding.app_tagline || '',
        primary_color: data.colors.primary,
        secondary_color: data.colors.secondary,
        accent_color: data.colors.accent,
        success_color: data.colors.success,
        warning_color: data.colors.warning,
        error_color: data.colors.error,
        info_color: data.colors.info,
        background_color: data.colors.background,
        background_secondary: data.colors.background_secondary,
        background_tertiary: data.colors.background_tertiary,
        sidebar_color: data.colors.sidebar,
        text_primary_color: data.colors.text_primary,
        text_secondary_color: data.colors.text_secondary,
        border_color: data.colors.border,
        shadow_color: data.colors.shadow,
        button_primary_color: data.buttons.primary_color,
        button_text_color: data.buttons.primary_text,
        button_secondary_color: data.buttons.secondary_color,
        button_secondary_text: data.buttons.secondary_text,
        button_border_radius: data.buttons.border_radius,
        input_border_color: data.inputs.border_color,
        input_focus_color: data.inputs.focus_color,
        input_border_radius: data.inputs.border_radius,
        card_background: data.cards.background,
        card_border_color: data.cards.border_color,
        card_border_radius: data.cards.border_radius,
        card_shadow: data.cards.shadow,
        nav_background: data.navigation.background,
        nav_text_color: data.navigation.text_color,
        nav_active_color: data.navigation.active_color,
        nav_hover_color: data.navigation.hover_color,
        font_family: data.typography.font_family || '',
        font_size_base: data.typography.font_size_base,
        font_size_heading: data.typography.font_size_heading,
        font_weight_normal: data.typography.font_weight_normal,
        font_weight_medium: data.typography.font_weight_medium,
        font_weight_bold: data.typography.font_weight_bold,
        line_height_base: data.typography.line_height_base,
        line_height_heading: data.typography.line_height_heading,
        letter_spacing: data.typography.letter_spacing,
        border_radius: data.layout.border_radius,
        spacing_unit: data.layout.spacing_unit,
        spacing_xs: data.layout.spacing_xs,
        spacing_sm: data.layout.spacing_sm,
        spacing_md: data.layout.spacing_md,
        spacing_lg: data.layout.spacing_lg,
        spacing_xl: data.layout.spacing_xl,
        animation_speed: data.animations.speed,
        enable_animations: data.animations.enabled,
        dark_mode_enabled: data.dark_mode.enabled,
        custom_css: data.advanced.custom_css || '',
        theme_name: data.metadata.theme_name || '',
        theme_description: data.metadata.theme_description || '',
      });

      if (data.dark_mode.colors && Object.keys(data.dark_mode.colors).length > 0) {
        const loadedColors: DarkModeColors = {
          background: data.dark_mode.colors.background || '#1f2937',
          background_secondary: data.dark_mode.colors.background_secondary || '#111827',
          background_tertiary: data.dark_mode.colors.background_tertiary || '#0f172a',
          text_primary: data.dark_mode.colors.text_primary || '#f9fafb',
          text_secondary: data.dark_mode.colors.text_secondary || '#d1d5db',
          border: data.dark_mode.colors.border || '#374151',
          shadow: data.dark_mode.colors.shadow || '#00000040',
        };
        setDarkModeColors(loadedColors);
      }
      
      setLogos({
        light: data.branding.logo_url ? getFullImageUrl(data.branding.logo_url) : null,
        dark: data.branding.logo_dark_url ? getFullImageUrl(data.branding.logo_dark_url) : null,
        favicon: data.branding.favicon_url ? getFullImageUrl(data.branding.favicon_url) : null,
      });
    } catch (error: any) {
      toast.error('Failed to load customization');
      console.error('Load customization error:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadPresets = async () => {
    try {
      const data = await api.getThemePresets();
      setPresets(data);
    } catch (error) {
      console.error('Failed to load presets:', error);
    }
  };

  const toggleSection = (section: SectionKey) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleFieldChange = (field: string, value: any) => {
    setFormData(prev => ({ ...prev, [field]: value }));
  };

  const handleGenerateDarkMode = () => {
    const generated = generateDarkMode(formData);
    
    const validatedColors: DarkModeColors = {
      background: generated.background || '#1f2937',
      background_secondary: generated.background_secondary || '#111827',
      background_tertiary: generated.background_tertiary || '#0f172a',
      text_primary: generated.text_primary || '#f9fafb',
      text_secondary: generated.text_secondary || '#d1d5db',
      border: generated.border || '#374151',
      shadow: generated.shadow || '#00000040',
    };
    
    setDarkModeColors(validatedColors);
    toast.success('Dark mode colors generated! Review and adjust as needed.');
  };

  const handleLogoUpload = async (
    event: React.ChangeEvent<HTMLInputElement>,
    logoType: 'light' | 'dark' | 'favicon'
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.size > 5 * 1024 * 1024) {
      toast.error('File size must be less than 5MB');
      event.target.value = '';
      return;
    }

    const validTypes = ['image/png', 'image/jpeg', 'image/svg+xml', 'image/x-icon'];
    if (!validTypes.includes(file.type)) {
      toast.error('Invalid file type. Use PNG, JPG, SVG, or ICO');
      event.target.value = '';
      return;
    }

    const previewUrl = URL.createObjectURL(file);

    setPreviewModal({
      isOpen: true,
      file,
      previewUrl,
      logoType,
    });

    event.target.value = '';
  };

  const confirmLogoUpload = async () => {
    const { file, logoType } = previewModal;
    
    if (!file) return;

    closePreviewModal();

    setUploadingLogo(logoType);
    try {
      const response = await api.uploadLogo(file, logoType, organizationName);
      const fullUrl = getFullImageUrl(response.url);
      setLogos(prev => ({ ...prev, [logoType]: fullUrl }));
      toast.success(`${logoType} logo uploaded successfully`);
    } catch (error: any) {
      toast.error(`Failed to upload ${logoType} logo`);
      console.error(error);
    } finally {
      setUploadingLogo(null);
    }
  };

  const closePreviewModal = () => {
    if (previewModal.previewUrl) {
      URL.revokeObjectURL(previewModal.previewUrl);
    }
    
    setPreviewModal({
      isOpen: false,
      file: null,
      previewUrl: null,
      logoType: 'light',
    });
  };

  const handleLogoDelete = async (logoType: 'light' | 'dark' | 'favicon') => {
    if (!confirm(`Are you sure you want to delete the ${logoType} logo?`)) {
      return;
    }

    try {
      await api.deleteLogo(logoType, organizationName);
      setLogos(prev => ({ ...prev, [logoType]: null }));
      toast.success(`${logoType} logo deleted`);
    } catch (error: any) {
      toast.error(`Failed to delete ${logoType} logo`);
      console.error(error);
    }
  };

  const handleApplyPreset = async (presetName: string) => {
    console.log('Applying preset:', presetName);
    setLoading(true);
    
    try {
      const response = await api.applyThemePreset(presetName, organizationName);
      console.log('Preset applied on backend:', response);
      
      setActivePreset(presetName);
      
      await loadCustomization();
      console.log('Customization reloaded');
      
      await refreshCustomization();
      console.log('Global context refreshed');
      
      toast.success(`Theme preset "${presetName}" applied successfully!`);
      
      const preview = document.querySelector('.lg\\:col-span-1');
      if (preview) {
        preview.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
      
    } catch (error: any) {
      console.error('Failed to apply preset:', error);
      toast.error('Failed to apply preset: ' + (error.response?.data?.detail || error.message));
    } finally {
      setLoading(false);
    }
  };

  const handleApplyTemplate = (template: CSSTemplate) => {
    setFormData(prev => ({
      ...prev,
      custom_css: prev.custom_css
        ? `${prev.custom_css}\n\n${template.cssCode}`
        : template.cssCode
    }));
    setShowTemplates(false);
    toast.success(`Template "${template.name}" added to Custom CSS!`);
  };

  const handleExportTheme = () => {
    const themeData = {
      ...formData,
      dark_mode_colors: darkModeColors,
      exported_at: new Date().toISOString(),
      version: '1.0.0'
    };

    const blob = new Blob([JSON.stringify(themeData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `theme-${formData.theme_name || 'custom'}-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success('Theme exported successfully!');
  };

  const handleImportTheme = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const imported = JSON.parse(e.target?.result as string);
        
        setFormData(prev => ({
          ...prev,
          ...imported,
        }));

        if (imported.dark_mode_colors) {
          setDarkModeColors(imported.dark_mode_colors);
        }

        toast.success('Theme imported successfully!');
      } catch (error) {
        toast.error('Invalid theme file');
        console.error(error);
      }
    };
    reader.readAsText(file);
    event.target.value = '';
  };

  const handleSave = async () => {
    if (validationResults && !validationResults.valid) {
      toast.error('Please fix validation errors before saving');
      return;
    }

    setSaving(true);
    try {
      const updateData: CustomizationUpdateRequest = { 
        ...formData,
        dark_mode_colors: formData.dark_mode_enabled ? darkModeColors : undefined
      };

      console.log('💾 Saving customization:', {
        dark_mode_enabled: updateData.dark_mode_enabled,
        dark_mode_colors: updateData.dark_mode_colors,
      });

      const response = await api.updateCustomization(updateData, organizationName);
      
      console.log('✅ Server response:', response);
      
      // Refresh the customization context
      await refreshCustomization();
      
      toast.success('Customization saved successfully! Theme applied.');
      
    } catch (error: any) {
      console.error('❌ Save error:', error.response?.data || error);
      toast.error('Failed to save customization: ' + (error.response?.data?.detail || error.message));
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    if (!confirm('Are you sure you want to reset all customization to defaults?')) {
      return;
    }

    try {
      await api.resetCustomization(organizationName);
      await loadCustomization();
      await refreshCustomization();
      toast.success('Customization reset to defaults');
      
      setTimeout(() => {
        window.location.reload();
      }, 1000);
    } catch (error: any) {
      toast.error('Failed to reset customization');
      console.error(error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto bg-gray-50 scroll-smooth-touch">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 py-4 sm:py-8">
        {/* Header */}
        <div className="mb-6 sm:mb-8">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 sm:gap-0">
            <div>
              <div className="flex items-center gap-2 mb-2">
                <Palette className="w-6 h-6 sm:w-8 sm:h-8 text-primary-600" />
                <h1 className="text-2xl sm:text-3xl font-bold text-gray-900">Customization</h1>
              </div>
              <p className="text-sm sm:text-base text-gray-600">
                Visual theme editor
              </p>
            </div>

            <div className="flex flex-col sm:flex-row gap-2 sm:gap-3">
              <button
                onClick={() => setShowPreview(!showPreview)}
                className="flex items-center justify-center gap-2 px-4 py-2.5 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-sm sm:text-base font-medium min-touch-target"
              >
                <Eye className="w-4 h-4" />
                {showPreview ? 'Hide Preview' : 'Show Preview'}
              </button>
              <button
                onClick={handleReset}
                className="flex items-center justify-center gap-2 px-4 py-2.5 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-orange-600 text-sm sm:text-base font-medium min-touch-target"
              >
                <RotateCcw className="w-4 h-4" />
                <span className="hidden xs:inline">Reset to Defaults</span>
                <span className="xs:hidden">Reset</span>
              </button>
              <button
                onClick={handleSave}
                disabled={saving || (validationResults && !validationResults.valid)}
                className="flex items-center justify-center gap-2 px-4 sm:px-6 py-2.5 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed text-sm sm:text-base font-medium min-touch-target"
              >
                <Save className="w-4 h-4" />
                {saving ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </div>

          {/* Validation Status */}
          {validationResults && (
            <div className="mt-4">
              {validationResults.valid ? (
                <div className="flex items-center gap-2 text-sm text-green-600 bg-green-50 px-4 py-2 rounded-lg">
                  <CheckCircle className="w-4 h-4" />
                  <span>All validation checks passed ✓</span>
                </div>
              ) : (
                <div className="space-y-2">
                  {validationResults.errors.map((error: string, i: number) => (
                    <div key={i} className="flex items-start gap-2 text-sm text-red-600 bg-red-50 px-4 py-2 rounded-lg">
                      <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>{error}</span>
                    </div>
                  ))}
                  {validationResults.warnings.map((warning: string, i: number) => (
                    <div key={i} className="flex items-start gap-2 text-sm text-yellow-600 bg-yellow-50 px-4 py-2 rounded-lg">
                      <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                      <span>{warning}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Theme Presets */}
        {presets.length > 0 && (
          <div className="mb-6 bg-white rounded-lg border border-gray-200 p-4 sm:p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-primary-600" />
                <h2 className="text-lg font-semibold text-gray-900">Quick Start Presets</h2>
              </div>
              <div className="flex gap-2">
                <label className="cursor-pointer">
                  <input
                    type="file"
                    accept=".json"
                    onChange={handleImportTheme}
                    className="hidden"
                  />
                  <div className="flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors">
                    <UploadIcon className="w-4 h-4" />
                    Import
                  </div>
                </label>
                <button
                  onClick={handleExportTheme}
                  className="flex items-center gap-2 px-3 py-2 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <Download className="w-4 h-4" />
                  Export
                </button>
              </div>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
              {presets.map((preset) => (
                <button
                  key={preset.name}
                  onClick={() => handleApplyPreset(preset.name)}
                  className={`
                    relative p-4 rounded-lg border-2 transition-all hover:shadow-md
                    ${activePreset === preset.name 
                      ? 'border-primary-500 bg-primary-50' 
                      : 'border-gray-200 hover:border-primary-300'
                    }
                  `}
                >
                  <div className="flex flex-col items-center gap-2">
                    <div className="flex gap-1">
                      <div 
                        className="w-6 h-6 rounded-full border border-gray-300"
                        style={{ backgroundColor: preset.colors.primary }}
                      />
                      <div 
                        className="w-6 h-6 rounded-full border border-gray-300"
                        style={{ backgroundColor: preset.colors.secondary }}
                      />
                    </div>
                    <span className="text-xs font-medium text-gray-900 text-center">
                      {preset.name}
                    </span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
          {/* Main Settings */}
          <div className="lg:col-span-2 space-y-3 sm:space-y-4">
            
            {/* Section 1: Branding & Identity */}
            <AccordionSection
              title="Branding & Identity"
              icon={<Sparkles className="w-5 h-5" />}
              isExpanded={expandedSections.branding}
              onToggle={() => toggleSection('branding')}
            >
              <div className="space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Application Name
                    </label>
                    <input
                      type="text"
                      value={formData.app_name}
                      onChange={(e) => handleFieldChange('app_name', e.target.value)}
                      placeholder="My Company"
                      className="w-full px-3 sm:px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm sm:text-base"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Tagline
                    </label>
                    <input
                      type="text"
                      value={formData.app_tagline}
                      onChange={(e) => handleFieldChange('app_tagline', e.target.value)}
                      placeholder="AI-Powered Knowledge Assistant"
                      className="w-full px-3 sm:px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm sm:text-base"
                    />
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-3">Brand Assets</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4">
                    <LogoUploadCard
                      title="Light Mode Logo"
                      logoUrl={logos.light}
                      uploading={uploadingLogo === 'light'}
                      onUpload={(e) => handleLogoUpload(e, 'light')}
                      onDelete={() => handleLogoDelete('light')}
                    />
                    
                    <LogoUploadCard
                      title="Dark Mode Logo"
                      logoUrl={logos.dark}
                      uploading={uploadingLogo === 'dark'}
                      onUpload={(e) => handleLogoUpload(e, 'dark')}
                      onDelete={() => handleLogoDelete('dark')}
                    />
                    
                    <LogoUploadCard
                      title="Favicon"
                      logoUrl={logos.favicon}
                      uploading={uploadingLogo === 'favicon'}
                      onUpload={(e) => handleLogoUpload(e, 'favicon')}
                      onDelete={() => handleLogoDelete('favicon')}
                    />
                  </div>
                </div>
              </div>
            </AccordionSection>

            {/* Section 2: Color System */}
            <AccordionSection
              title="Color System"
              icon={<Palette className="w-5 h-5" />}
              isExpanded={expandedSections.colors}
              onToggle={() => toggleSection('colors')}
            >
              <div className="space-y-6">
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Brand Colors</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <ColorPicker
                      label="Primary"
                      value={formData.primary_color}
                      onChange={(value) => handleFieldChange('primary_color', value)}
                      description="Main brand color"
                    />
                    <ColorPicker
                      label="Secondary"
                      value={formData.secondary_color}
                      onChange={(value) => handleFieldChange('secondary_color', value)}
                      description="Secondary accent"
                    />
                    <ColorPicker
                      label="Accent"
                      value={formData.accent_color}
                      onChange={(value) => handleFieldChange('accent_color', value)}
                      description="Highlight color"
                    />
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Semantic Colors</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <ColorPicker
                      label="Success"
                      value={formData.success_color}
                      onChange={(value) => handleFieldChange('success_color', value)}
                    />
                    <ColorPicker
                      label="Warning"
                      value={formData.warning_color}
                      onChange={(value) => handleFieldChange('warning_color', value)}
                    />
                    <ColorPicker
                      label="Error"
                      value={formData.error_color}
                      onChange={(value) => handleFieldChange('error_color', value)}
                    />
                    <ColorPicker
                      label="Info"
                      value={formData.info_color}
                      onChange={(value) => handleFieldChange('info_color', value)}
                    />
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Background Colors</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <ColorPicker
                      label="Primary BG"
                      value={formData.background_color}
                      onChange={(value) => handleFieldChange('background_color', value)}
                    />
                    <ColorPicker
                      label="Secondary BG"
                      value={formData.background_secondary}
                      onChange={(value) => handleFieldChange('background_secondary', value)}
                    />
                    <ColorPicker
                      label="Tertiary BG"
                      value={formData.background_tertiary}
                      onChange={(value) => handleFieldChange('background_tertiary', value)}
                    />
                    <ColorPicker
                      label="Sidebar"
                      value={formData.sidebar_color}
                      onChange={(value) => handleFieldChange('sidebar_color', value)}
                    />
                  </div>
                </div>

                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Text & Borders</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <ColorPicker
                      label="Primary Text"
                      value={formData.text_primary_color}
                      onChange={(value) => handleFieldChange('text_primary_color', value)}
                    />
                    <ColorPicker
                      label="Secondary Text"
                      value={formData.text_secondary_color}
                      onChange={(value) => handleFieldChange('text_secondary_color', value)}
                    />
                    <ColorPicker
                      label="Border"
                      value={formData.border_color}
                      onChange={(value) => handleFieldChange('border_color', value)}
                    />
                    <ColorPicker
                      label="Shadow"
                      value={formData.shadow_color}
                      onChange={(value) => handleFieldChange('shadow_color', value)}
                    />
                  </div>
                </div>
              </div>
            </AccordionSection>

            {/* Section 3: Typography */}
            <AccordionSection
              title="Typography"
              icon={<Type className="w-5 h-5" />}
              isExpanded={expandedSections.typography}
              onToggle={() => toggleSection('typography')}
            >
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Font Family
                  </label>
                  <select
                    value={formData.font_family}
                    onChange={(e) => handleFieldChange('font_family', e.target.value)}
                    className="w-full px-3 sm:px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm sm:text-base"
                  >
                    <option value="">System Default</option>
                    <option value="Inter">Inter</option>
                    <option value="Roboto">Roboto</option>
                    <option value="Poppins">Poppins</option>
                    <option value="Open Sans">Open Sans</option>
                    <option value="Lato">Lato</option>
                    <option value="Montserrat">Montserrat</option>
                  </select>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Base Size
                    </label>
                    <input
                      type="text"
                      value={formData.font_size_base}
                      onChange={(e) => handleFieldChange('font_size_base', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Heading Size
                    </label>
                    <input
                      type="text"
                      value={formData.font_size_heading}
                      onChange={(e) => handleFieldChange('font_size_heading', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Letter Spacing
                    </label>
                    <input
                      type="text"
                      value={formData.letter_spacing}
                      onChange={(e) => handleFieldChange('letter_spacing', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Normal Weight
                    </label>
                    <select
                      value={formData.font_weight_normal}
                      onChange={(e) => handleFieldChange('font_weight_normal', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    >
                      <option value="300">300 (Light)</option>
                      <option value="400">400 (Normal)</option>
                      <option value="500">500 (Medium)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Medium Weight
                    </label>
                    <select
                      value={formData.font_weight_medium}
                      onChange={(e) => handleFieldChange('font_weight_medium', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    >
                      <option value="400">400 (Normal)</option>
                      <option value="500">500 (Medium)</option>
                      <option value="600">600 (Semi-Bold)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Bold Weight
                    </label>
                    <select
                      value={formData.font_weight_bold}
                      onChange={(e) => handleFieldChange('font_weight_bold', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    >
                      <option value="600">600 (Semi-Bold)</option>
                      <option value="700">700 (Bold)</option>
                      <option value="800">800 (Extra Bold)</option>
                      <option value="900">900 (Black)</option>
                    </select>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Base Line Height
                    </label>
                    <select
                      value={formData.line_height_base}
                      onChange={(e) => handleFieldChange('line_height_base', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    >
                      <option value="1.2">1.2 (Tight)</option>
                      <option value="1.3">1.3 (Snug)</option>
                      <option value="1.5">1.5 (Normal)</option>
                      <option value="1.6">1.6 (Relaxed)</option>
                      <option value="1.8">1.8 (Loose)</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Heading Line Height
                    </label>
                    <select
                      value={formData.line_height_heading}
                      onChange={(e) => handleFieldChange('line_height_heading', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    >
                      <option value="1">1 (None)</option>
                      <option value="1.1">1.1 (Tight)</option>
                      <option value="1.2">1.2 (Snug)</option>
                      <option value="1.25">1.25 (Normal)</option>
                      <option value="1.4">1.4 (Relaxed)</option>
                    </select>
                  </div>
                </div>
              </div>
            </AccordionSection>

            {/* Section 4: Layout & Spacing */}
            <AccordionSection
              title="Layout & Spacing"
              icon={<LayoutIcon className="w-5 h-5" />}
              isExpanded={expandedSections.layout}
              onToggle={() => toggleSection('layout')}
            >
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Global Border Radius
                  </label>
                  <select
                    value={formData.border_radius}
                    onChange={(e) => handleFieldChange('border_radius', e.target.value)}
                    className="w-full px-3 sm:px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 text-sm sm:text-base"
                  >
                    <option value="0px">None (0px)</option>
                    <option value="4px">Small (4px)</option>
                    <option value="6px">Medium (6px)</option>
                    <option value="8px">Large (8px)</option>
                    <option value="12px">Extra Large (12px)</option>
                    <option value="16px">Rounded (16px)</option>
                  </select>
                </div>

                <div>
                  <h3 className="text-sm font-medium text-gray-700 mb-3">Spacing Scale</h3>
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Extra Small (XS)</label>
                      <input
                        type="text"
                        value={formData.spacing_xs}
                        onChange={(e) => handleFieldChange('spacing_xs', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Small (SM)</label>
                      <input
                        type="text"
                        value={formData.spacing_sm}
                        onChange={(e) => handleFieldChange('spacing_sm', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Medium (MD)</label>
                      <input
                        type="text"
                        value={formData.spacing_md}
                        onChange={(e) => handleFieldChange('spacing_md', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Large (LG)</label>
                      <input
                        type="text"
                        value={formData.spacing_lg}
                        onChange={(e) => handleFieldChange('spacing_lg', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Extra Large (XL)</label>
                      <input
                        type="text"
                        value={formData.spacing_xl}
                        onChange={(e) => handleFieldChange('spacing_xl', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-xs text-gray-600 mb-1">Base Unit</label>
                      <input
                        type="text"
                        value={formData.spacing_unit}
                        onChange={(e) => handleFieldChange('spacing_unit', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                      />
                    </div>
                  </div>
                </div>
              </div>
            </AccordionSection>

            {/* Section 5: Component Styling */}
            <AccordionSection
              title="Component Styling"
              icon={<Package className="w-5 h-5" />}
              isExpanded={expandedSections.components}
              onToggle={() => toggleSection('components')}
            >
              <div className="space-y-6">
                {/* Buttons */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Buttons</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <ColorPicker
                      label="Primary Button"
                      value={formData.button_primary_color}
                      onChange={(value) => handleFieldChange('button_primary_color', value)}
                    />
                    <ColorPicker
                      label="Primary Button Text"
                      value={formData.button_text_color}
                      onChange={(value) => handleFieldChange('button_text_color', value)}
                    />
                    <ColorPicker
                      label="Secondary Button"
                      value={formData.button_secondary_color}
                      onChange={(value) => handleFieldChange('button_secondary_color', value)}
                    />
                    <ColorPicker
                      label="Secondary Button Text"
                      value={formData.button_secondary_text}
                      onChange={(value) => handleFieldChange('button_secondary_text', value)}
                    />
                  </div>
                  <div className="mt-3">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Button Border Radius
                    </label>
                    <select
                      value={formData.button_border_radius}
                      onChange={(e) => handleFieldChange('button_border_radius', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    >
                      <option value="4px">4px</option>
                      <option value="6px">6px</option>
                      <option value="8px">8px</option>
                      <option value="10px">10px</option>
                      <option value="12px">12px</option>
                      <option value="16px">16px (Pill)</option>
                    </select>
                  </div>
                </div>

                {/* Input Fields */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Input Fields</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <ColorPicker
                      label="Border Color"
                      value={formData.input_border_color}
                      onChange={(value) => handleFieldChange('input_border_color', value)}
                    />
                    <ColorPicker
                      label="Focus Color"
                      value={formData.input_focus_color}
                      onChange={(value) => handleFieldChange('input_focus_color', value)}
                    />
                  </div>
                  <div className="mt-3">
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Input Border Radius
                    </label>
                    <select
                      value={formData.input_border_radius}
                      onChange={(e) => handleFieldChange('input_border_radius', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    >
                      <option value="4px">4px</option>
                      <option value="6px">6px</option>
                      <option value="8px">8px</option>
                      <option value="10px">10px</option>
                      <option value="12px">12px</option>
                    </select>
                  </div>
                </div>

                {/* Cards */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Cards</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <ColorPicker
                      label="Background"
                      value={formData.card_background}
                      onChange={(value) => handleFieldChange('card_background', value)}
                    />
                    <ColorPicker
                      label="Border Color"
                      value={formData.card_border_color}
                      onChange={(value) => handleFieldChange('card_border_color', value)}
                    />
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Border Radius
                      </label>
                      <select
                        value={formData.card_border_radius}
                        onChange={(e) => handleFieldChange('card_border_radius', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                      >
                        <option value="6px">6px</option>
                        <option value="8px">8px</option>
                        <option value="10px">10px</option>
                        <option value="12px">12px</option>
                        <option value="16px">16px</option>
                        <option value="20px">20px</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Shadow
                      </label>
                      <select
                        value={formData.card_shadow}
                        onChange={(e) => handleFieldChange('card_shadow', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                      >
                        <option value="none">None</option>
                        <option value="0 1px 2px rgba(0,0,0,0.05)">Subtle</option>
                        <option value="0 1px 3px rgba(0,0,0,0.1)">Small</option>
                        <option value="0 4px 6px rgba(0,0,0,0.1)">Medium</option>
                        <option value="0 10px 15px rgba(0,0,0,0.1)">Large</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Navigation */}
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Navigation</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <ColorPicker
                      label="Background"
                      value={formData.nav_background}
                      onChange={(value) => handleFieldChange('nav_background', value)}
                    />
                    <ColorPicker
                      label="Text Color"
                      value={formData.nav_text_color}
                      onChange={(value) => handleFieldChange('nav_text_color', value)}
                    />
                    <ColorPicker
                      label="Active Color"
                      value={formData.nav_active_color}
                      onChange={(value) => handleFieldChange('nav_active_color', value)}
                    />
                    <ColorPicker
                      label="Hover Color"
                      value={formData.nav_hover_color}
                      onChange={(value) => handleFieldChange('nav_hover_color', value)}
                    />
                  </div>
                </div>
              </div>
            </AccordionSection>

            {/* Section 6: Dark Mode */}
            <AccordionSection
              title="Dark Mode (Hybrid)"
              icon={<Moon className="w-5 h-5" />}
              isExpanded={expandedSections.darkmode}
              onToggle={() => toggleSection('darkmode')}
            >
              <div className="space-y-4">
                <div className="flex items-center justify-between p-4 bg-purple-50 border border-purple-200 rounded-lg">
                  <div className="flex items-center gap-3">
                    <Zap className="w-5 h-5 text-purple-600" />
                    <div>
                      <h4 className="text-sm font-medium text-gray-900">Auto-Generate Dark Colors</h4>
                      <p className="text-xs text-gray-600 mt-1">
                        Automatically create dark mode variants from your light theme
                      </p>
                    </div>
                  </div>
                  <button
                    onClick={handleGenerateDarkMode}
                    className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium whitespace-nowrap"
                  >
                    Generate
                  </button>
                </div>

                <div>
                  <label className="flex items-center gap-2 cursor-pointer mb-4">
                    <input
                      type="checkbox"
                      checked={formData.dark_mode_enabled}
                      onChange={(e) => handleFieldChange('dark_mode_enabled', e.target.checked)}
                      className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                    />
                    <span className="text-sm font-medium text-gray-700">Enable Dark Mode Support</span>
                  </label>
                </div>

                {formData.dark_mode_enabled && (
                  <div className="space-y-4">
                    <h3 className="text-sm font-semibold text-gray-700">Dark Mode Colors (Override)</h3>
                    <p className="text-xs text-gray-600">
                      These colors will be used when dark mode is active. Leave as-is to use auto-generated values.
                    </p>
                    
                    <div className="grid grid-cols-2 gap-4">
                      <ColorPicker
                        label="Dark Background"
                        value={darkModeColors.background}
                        onChange={(value) => setDarkModeColors(prev => ({ ...prev, background: value }))}
                      />
                      <ColorPicker
                        label="Dark BG Secondary"
                        value={darkModeColors.background_secondary}
                        onChange={(value) => setDarkModeColors(prev => ({ ...prev, background_secondary: value }))}
                      />
                      <ColorPicker
                        label="Dark BG Tertiary"
                        value={darkModeColors.background_tertiary}
                        onChange={(value) => setDarkModeColors(prev => ({ ...prev, background_tertiary: value }))}
                      />
                      <ColorPicker
                        label="Dark Text Primary"
                        value={darkModeColors.text_primary}
                        onChange={(value) => setDarkModeColors(prev => ({ ...prev, text_primary: value }))}
                      />
                      <ColorPicker
                        label="Dark Text Secondary"
                        value={darkModeColors.text_secondary}
                        onChange={(value) => setDarkModeColors(prev => ({ ...prev, text_secondary: value }))}
                      />
                      <ColorPicker
                        label="Dark Border"
                        value={darkModeColors.border}
                        onChange={(value) => setDarkModeColors(prev => ({ ...prev, border: value }))}
                      />
                    </div>
                  </div>
                )}
              </div>
            </AccordionSection>

            {/* Section 7: Advanced Settings */}
            <AccordionSection
              title="Advanced Settings"
              icon={<Settings className="w-5 h-5" />}
              isExpanded={expandedSections.advanced}
              onToggle={() => toggleSection('advanced')}
            >
              <div className="space-y-4">
                <div>
                  <h3 className="text-sm font-semibold text-gray-700 mb-3">Animations</h3>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Animation Speed
                      </label>
                      <select
                        value={formData.animation_speed}
                        onChange={(e) => handleFieldChange('animation_speed', e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                      >
                        <option value="150ms">Fast (150ms)</option>
                        <option value="200ms">Quick (200ms)</option>
                        <option value="300ms">Normal (300ms)</option>
                        <option value="500ms">Slow (500ms)</option>
                        <option value="700ms">Very Slow (700ms)</option>
                      </select>
                    </div>
                    <div>
                      <label className="flex items-center gap-2 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={formData.enable_animations}
                          onChange={(e) => handleFieldChange('enable_animations', e.target.checked)}
                          className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                        />
                        <span className="text-sm font-medium text-gray-700">Enable Animations</span>
                      </label>
                      <p className="text-xs text-gray-500 mt-1 ml-6">
                        Disable for better performance or accessibility
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div>
                      <label className="block text-sm font-medium text-gray-700">
                        Custom CSS (Advanced)
                      </label>
                      <p className="text-xs text-gray-500 mt-1">
                        Add custom styles to override or extend the theme
                      </p>
                    </div>
                    <button
                      onClick={() => setShowTemplates(!showTemplates)}
                      className="flex items-center gap-2 px-3 py-1.5 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors"
                    >
                      <BookOpen className="w-4 h-4" />
                      Templates
                    </button>
                  </div>

                  {showTemplates && (
                    <div className="mb-4 p-4 bg-gray-50 border border-gray-200 rounded-lg">
                      <h4 className="text-sm font-medium text-gray-700 mb-3">CSS Templates</h4>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-60 overflow-y-auto">
                        {CSS_TEMPLATES.map((template) => (
                          <button
                            key={template.id}
                            onClick={() => handleApplyTemplate(template)}
                            className="text-left p-3 border border-gray-200 rounded-lg hover:border-primary-500 hover:bg-white transition-all group"
                          >
                            <div className="flex items-start justify-between mb-1">
                              <span className="text-xs font-medium text-gray-900 group-hover:text-primary-600">
                                {template.name}
                              </span>
                              <span className={`text-xs px-1.5 py-0.5 rounded ${
                                template.difficulty === 'beginner' ? 'bg-green-100 text-green-700' :
                                template.difficulty === 'intermediate' ? 'bg-yellow-100 text-yellow-700' :
                                'bg-red-100 text-red-700'
                              }`}>
                                {template.difficulty}
                              </span>
                            </div>
                            <p className="text-xs text-gray-500 line-clamp-2">
                              {template.description}
                            </p>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="border border-gray-300 rounded-lg overflow-hidden">
                    <Editor
                      height="400px"
                      defaultLanguage="css"
                      value={formData.custom_css}
                      onChange={(value) => handleFieldChange('custom_css', value || '')}
                      theme="vs-light"
                      options={{
                        minimap: { enabled: false },
                        fontSize: 13,
                        lineNumbers: 'on',
                        scrollBeyondLastLine: false,
                        wordWrap: 'on',
                        automaticLayout: true,
                        tabSize: 2,
                        suggest: {
                          showKeywords: true,
                          showSnippets: true,
                        },
                      }}
                    />
                  </div>
                  
                  <div className="mt-2 flex items-start gap-2 text-xs text-gray-500">
                    <Lightbulb className="w-4 h-4 mt-0.5 flex-shrink-0 text-yellow-500" />
                    <div>
                      <p className="font-medium text-gray-700">Tips:</p>
                      <ul className="list-disc list-inside space-y-1 mt-1">
                        <li>Use CSS variables like <code className="bg-gray-100 px-1 rounded">var(--color-primary)</code></li>
                        <li>Click "Templates" above for ready-made effects</li>
                        <li>Changes apply globally across the app</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Theme Name
                    </label>
                    <input
                      type="text"
                      value={formData.theme_name}
                      onChange={(e) => handleFieldChange('theme_name', e.target.value)}
                      placeholder="My Custom Theme"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Theme Description
                    </label>
                    <input
                      type="text"
                      value={formData.theme_description}
                      onChange={(e) => handleFieldChange('theme_description', e.target.value)}
                      placeholder="A brief description of your theme"
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                    />
                  </div>
                </div>
              </div>
            </AccordionSection>
          </div>

          {/* Preview Panel - Desktop Sticky, Mobile Bottom */}
          {showPreview && (
            <div className="lg:col-span-1">
              <div className="bg-white rounded-lg border border-gray-200 p-4 sm:p-6 lg:sticky lg:top-6">
                <div className="flex items-center gap-2 mb-4">
                  <Eye className="w-5 h-5 text-primary-600" />
                  <h2 className="text-base sm:text-lg font-semibold text-gray-900">Live Preview</h2>
                </div>
                
                <LivePreview
                  formData={formData}
                  logoUrl={logos.light}
                  validationResults={validationResults}
                />
              </div>
            </div>
          )}
        </div>

        {/* Image Preview Modal */}
        <ImagePreviewModal
          isOpen={previewModal.isOpen}
          onClose={closePreviewModal}
          onConfirm={confirmLogoUpload}
          file={previewModal.file}
          previewUrl={previewModal.previewUrl}
          logoType={previewModal.logoType}
        />
      </div>
    </div>
  );
}

// Supporting Components

interface AccordionSectionProps {
  title: string;
  icon: React.ReactNode;
  isExpanded: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

function AccordionSection({ title, icon, isExpanded, onToggle, children }: AccordionSectionProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 sm:px-6 py-3 sm:py-4 hover:bg-gray-50 transition-colors min-touch-target"
      >
        <div className="flex items-center gap-3">
          <div className="text-primary-600">{icon}</div>
          <h2 className="text-base sm:text-lg font-semibold text-gray-900">{title}</h2>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-gray-500 flex-shrink-0" />
        ) : (
          <ChevronDown className="w-5 h-5 text-gray-500 flex-shrink-0" />
        )}
      </button>
      {isExpanded && (
        <div className="px-4 sm:px-6 py-4 border-t border-gray-200 animate-fadeIn">
          {children}
        </div>
      )}
    </div>
  );
}

interface ColorPickerProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  description?: string;
}

function ColorPicker({ label, value, onChange, description }: ColorPickerProps) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
      </label>
      <div className="flex items-center gap-3">
        <input
          type="color"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-12 h-12 sm:w-14 sm:h-14 rounded border border-gray-300 cursor-pointer flex-shrink-0"
        />
        <div className="flex-1 min-w-0">
          <input
            type="text"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            placeholder="#000000"
            pattern="^#[0-9A-Fa-f]{6}$"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent font-mono text-sm"
          />
          {description && (
            <p className="text-xs text-gray-500 mt-1">{description}</p>
          )}
        </div>
      </div>
    </div>
  );
}

interface LogoUploadCardProps {
  title: string;
  logoUrl: string | null;
  uploading: boolean;
  onUpload: (event: React.ChangeEvent<HTMLInputElement>) => void;
  onDelete: () => void;
}

function LogoUploadCard({ title, logoUrl, uploading, onUpload, onDelete }: LogoUploadCardProps) {
  return (
    <div className="border border-gray-200 rounded-lg p-3 sm:p-4">
      <h3 className="text-sm font-medium text-gray-700 mb-3">{title}</h3>
      
      {logoUrl ? (
        <div className="space-y-3">
          <div className="w-full h-24 sm:h-32 bg-gray-50 rounded-lg flex items-center justify-center border border-gray-200">
            <img
              src={logoUrl}
              alt={title}
              className="max-w-full max-h-full object-contain p-2"
              onError={(e) => {
                console.error('Failed to load logo:', logoUrl);
                e.currentTarget.src = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="100" height="100"%3E%3Ctext x="50%25" y="50%25" text-anchor="middle" dy=".3em" fill="%23999"%3EError%3C/text%3E%3C/svg%3E';
              }}
            />
          </div>
          <button
            onClick={onDelete}
            className="w-full flex items-center justify-center gap-2 px-3 py-2 text-sm text-red-600 border border-red-200 rounded-lg hover:bg-red-50 transition-colors min-touch-target"
          >
            <X className="w-4 h-4" />
            Remove
          </button>
        </div>
      ) : (
        <label className="block">
          <input
            type="file"
            accept="image/png,image/jpeg,image/svg+xml,image/x-icon"
            onChange={onUpload}
            className="hidden"
            disabled={uploading}
          />
          <div className="w-full h-24 sm:h-32 bg-gray-50 rounded-lg flex flex-col items-center justify-center border-2 border-dashed border-gray-300 cursor-pointer hover:border-primary-500 hover:bg-primary-50 transition-colors">
            {uploading ? (
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
            ) : (
              <>
                <Upload className="w-6 h-6 sm:w-8 sm:h-8 text-gray-400 mb-2" />
                <span className="text-xs sm:text-sm text-gray-500 text-center px-2">Click to upload</span>
                <span className="text-xs text-gray-400 mt-1">PNG, JPG, SVG, ICO</span>
              </>
            )}
          </div>
        </label>
      )}
    </div>
  );
}

interface LivePreviewProps {
  formData: any;
  logoUrl: string | null;
  validationResults: any;
}

function LivePreview({ formData, logoUrl, validationResults }: LivePreviewProps) {
  return (
    <div className="space-y-4">
      {/* Logo Preview */}
      <div className="border border-gray-200 rounded-lg p-3 sm:p-4 bg-gray-50">
        <p className="text-xs font-medium text-gray-600 mb-2">Logo & Branding</p>
        <div className="h-12 sm:h-16 flex items-center justify-center bg-white rounded border border-gray-200 mb-2">
          {logoUrl ? (
            <img src={logoUrl} alt="Logo" className="max-h-10 sm:max-h-12 max-w-full object-contain" />
          ) : (
            <div 
              className="w-8 h-8 sm:w-10 sm:h-10 rounded-lg flex items-center justify-center"
              style={{ backgroundColor: formData.primary_color }}
            >
              <span className="text-white font-bold text-sm sm:text-base">
                {formData.app_name?.charAt(0) || 'N'}
              </span>
            </div>
          )}
        </div>
        {formData.app_name && (
          <p 
            className="font-semibold text-sm text-center"
            style={{ color: formData.text_primary_color }}
          >
            {formData.app_name}
          </p>
        )}
        {formData.app_tagline && (
          <p 
            className="text-xs text-center mt-1"
            style={{ color: formData.text_secondary_color }}
          >
            {formData.app_tagline}
          </p>
        )}
      </div>

      {/* Contrast Checker */}
      {validationResults && (
        <div className="border border-gray-200 rounded-lg p-3 sm:p-4 bg-gray-50">
          <p className="text-xs font-medium text-gray-600 mb-3">Accessibility Check</p>
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Text Contrast:</span>
              <span className={`font-medium ${
                validationResults.contrast.textOnBackground.passesAAA ? 'text-green-600' :
                validationResults.contrast.textOnBackground.passesAA ? 'text-yellow-600' :
                'text-red-600'
              }`}>
                {validationResults.contrast.textOnBackground.ratio}:1 
                ({validationResults.contrast.textOnBackground.rating.toUpperCase()})
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-gray-600">Button Contrast:</span>
              <span className={`font-medium ${
                validationResults.contrast.buttonContrast.passesAAA ? 'text-green-600' :
                validationResults.contrast.buttonContrast.passesAA ? 'text-yellow-600' :
                'text-red-600'
              }`}>
                {validationResults.contrast.buttonContrast.ratio}:1 
                ({validationResults.contrast.buttonContrast.rating.toUpperCase()})
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Color Palette */}
      <div className="border border-gray-200 rounded-lg p-3 sm:p-4 bg-gray-50">
        <p className="text-xs font-medium text-gray-600 mb-3">Color Palette</p>
        <div className="grid grid-cols-3 gap-2">
          <ColorSwatch label="Primary" color={formData.primary_color} />
          <ColorSwatch label="Secondary" color={formData.secondary_color} />
          <ColorSwatch label="Accent" color={formData.accent_color} />
          <ColorSwatch label="Success" color={formData.success_color} />
          <ColorSwatch label="Warning" color={formData.warning_color} />
          <ColorSwatch label="Error" color={formData.error_color} />
        </div>
      </div>

      {/* Button Preview */}
      <div className="border border-gray-200 rounded-lg p-3 sm:p-4 bg-gray-50">
        <p className="text-xs font-medium text-gray-600 mb-3">Buttons</p>
        <div className="space-y-2">
          <button
            className="w-full px-4 py-2 font-medium transition-opacity hover:opacity-90 text-sm"
            style={{
              backgroundColor: formData.button_primary_color,
              color: formData.button_text_color,
              borderRadius: formData.button_border_radius
            }}
          >
            Primary Button
          </button>
          <button
            className="w-full px-4 py-2 font-medium border text-sm"
            style={{
              backgroundColor: formData.button_secondary_color,
              color: formData.button_secondary_text,
              borderColor: formData.border_color,
              borderRadius: formData.button_border_radius
            }}
          >
            Secondary Button
          </button>
        </div>
      </div>

      {/* Input Preview */}
      <div className="border border-gray-200 rounded-lg p-3 sm:p-4 bg-gray-50">
        <p className="text-xs font-medium text-gray-600 mb-3">Input Fields</p>
        <input
          type="text"
          placeholder="Sample input field"
          className="w-full px-3 py-2 text-sm"
          style={{
            borderWidth: '1px',
            borderColor: formData.input_border_color,
            borderRadius: formData.input_border_radius,
            color: formData.text_primary_color
          }}
        />
      </div>

      {/* Card Preview */}
      <div 
        className="p-4"
        style={{
          backgroundColor: formData.card_background,
          borderWidth: '1px',
          borderColor: formData.card_border_color,
          borderRadius: formData.card_border_radius,
          boxShadow: formData.card_shadow
        }}
      >
        <p 
          className="text-sm font-medium mb-2"
          style={{ color: formData.text_primary_color }}
        >
          Card Component
        </p>
        <p 
          className="text-xs"
          style={{ color: formData.text_secondary_color }}
        >
          This is how your cards will look with the current styling.
        </p>
      </div>

      {/* Typography Preview */}
      <div 
        className="border border-gray-200 rounded-lg p-3 sm:p-4"
        style={{ backgroundColor: formData.background_color }}
      >
        <p 
          className="text-sm font-medium mb-2"
          style={{ 
            color: formData.text_primary_color,
            fontFamily: formData.font_family || 'inherit',
            fontSize: formData.font_size_base
          }}
        >
          Typography Preview
        </p>
        <p 
          className="text-xs"
          style={{ 
            color: formData.text_secondary_color,
            fontFamily: formData.font_family || 'inherit',
            fontSize: formData.font_size_base
          }}
        >
          Secondary text for descriptions and less important information.
        </p>
      </div>

      {/* Spacing Preview */}
      <div className="border border-gray-200 rounded-lg p-3 sm:p-4 bg-gray-50">
        <p className="text-xs font-medium text-gray-600 mb-3">Spacing Scale</p>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div 
              className="bg-primary-500"
              style={{ 
                width: formData.spacing_xs,
                height: '16px',
                backgroundColor: formData.primary_color
              }}
            />
            <span className="text-xs text-gray-600">XS: {formData.spacing_xs}</span>
          </div>
          <div className="flex items-center gap-2">
            <div 
              className="bg-primary-500"
              style={{ 
                width: formData.spacing_sm,
                height: '16px',
                backgroundColor: formData.primary_color
              }}
            />
            <span className="text-xs text-gray-600">SM: {formData.spacing_sm}</span>
          </div>
          <div className="flex items-center gap-2">
            <div 
              className="bg-primary-500"
              style={{ 
                width: formData.spacing_md,
                height: '16px',
                backgroundColor: formData.primary_color
              }}
            />
            <span className="text-xs text-gray-600">MD: {formData.spacing_md}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

interface ColorSwatchProps {
  label: string;
  color: string;
}

function ColorSwatch({ label, color }: ColorSwatchProps) {
  return (
    <div className="text-center">
      <div
        className="w-full h-10 sm:h-12 rounded-lg border border-gray-200 mb-1"
        style={{ backgroundColor: color }}
      ></div>
      <p className="text-xs text-gray-600 truncate">{label}</p>
    </div>
  );
}

export { AccordionSection, ColorPicker, LogoUploadCard, LivePreview, ColorSwatch };
