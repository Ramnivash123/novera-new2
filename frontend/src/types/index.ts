// Enhanced Customization Types

export interface CustomizationColors {
  primary: string;
  secondary: string;
  accent: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  background: string;
  background_secondary: string;
  background_tertiary: string;
  sidebar: string;
  text_primary: string;
  text_secondary: string;
  border: string;
  shadow: string;
}

export interface CustomizationButtons {
  primary_color: string;
  primary_text: string;
  secondary_color: string;
  secondary_text: string;
  border_radius: string;
}

export interface CustomizationInputs {
  border_color: string;
  focus_color: string;
  border_radius: string;
}

export interface CustomizationCards {
  background: string;
  border_color: string;
  border_radius: string;
  shadow: string;
}

export interface CustomizationNavigation {
  background: string;
  text_color: string;
  active_color: string;
  hover_color: string;
}

export interface CustomizationTypography {
  font_family: string | null;
  font_size_base: string;
  font_size_heading: string;
  font_weight_normal: string;
  font_weight_medium: string;
  font_weight_bold: string;
  line_height_base: string;
  line_height_heading: string;
  letter_spacing: string;
}

export interface CustomizationLayout {
  border_radius: string;
  spacing_unit: string;
  spacing_xs: string;
  spacing_sm: string;
  spacing_md: string;
  spacing_lg: string;
  spacing_xl: string;
}

export interface CustomizationAnimations {
  speed: string;
  enabled: boolean;
}

export interface CustomizationDarkMode {
  enabled: boolean;
  colors: Record<string, any>;
}

export interface CustomizationAdvanced {
  custom_css: string | null;
  custom_settings: Record<string, any>;
}

export interface CustomizationBranding {
  app_name: string | null;
  app_tagline: string | null;
  logo_url: string | null;
  logo_dark_url: string | null;
  favicon_url: string | null;
}

export interface CustomizationMetadata {
  theme_name: string | null;
  theme_description: string | null;
  is_preset: boolean;
  is_active: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface Customization {
  id: string;
  organization_name: string;
  branding: CustomizationBranding;
  colors: CustomizationColors;
  buttons: CustomizationButtons;
  inputs: CustomizationInputs;
  cards: CustomizationCards;
  navigation: CustomizationNavigation;
  typography: CustomizationTypography;
  layout: CustomizationLayout;
  animations: CustomizationAnimations;
  dark_mode: CustomizationDarkMode;
  advanced: CustomizationAdvanced;
  metadata: CustomizationMetadata;
}

export interface CustomizationUpdateRequest {
  organization_name?: string;
  
  // Branding
  app_name?: string;
  app_tagline?: string;
  
  // Colors
  primary_color?: string;
  secondary_color?: string;
  accent_color?: string;
  success_color?: string;
  warning_color?: string;
  error_color?: string;
  info_color?: string;
  background_color?: string;
  background_secondary?: string;
  background_tertiary?: string;
  sidebar_color?: string;
  text_primary_color?: string;
  text_secondary_color?: string;
  border_color?: string;
  shadow_color?: string;
  
  // Buttons
  button_primary_color?: string;
  button_text_color?: string;
  button_secondary_color?: string;
  button_secondary_text?: string;
  button_border_radius?: string;
  
  // Inputs
  input_border_color?: string;
  input_focus_color?: string;
  input_border_radius?: string;
  
  // Cards
  card_background?: string;
  card_border_color?: string;
  card_border_radius?: string;
  card_shadow?: string;
  
  // Navigation
  nav_background?: string;
  nav_text_color?: string;
  nav_active_color?: string;
  nav_hover_color?: string;
  
  // Typography
  font_family?: string;
  font_size_base?: string;
  font_size_heading?: string;
  font_weight_normal?: string;
  font_weight_medium?: string;
  font_weight_bold?: string;
  line_height_base?: string;
  line_height_heading?: string;
  letter_spacing?: string;
  
  // Layout
  border_radius?: string;
  spacing_unit?: string;
  spacing_xs?: string;
  spacing_sm?: string;
  spacing_md?: string;
  spacing_lg?: string;
  spacing_xl?: string;
  
  // Animations
  animation_speed?: string;
  enable_animations?: boolean;
  
  // Dark Mode
  dark_mode_enabled?: boolean;
  dark_mode_colors?: {
    background?: string;
    background_secondary?: string;
    background_tertiary?: string;
    text_primary?: string;
    text_secondary?: string;
    border?: string;
    shadow?: string;
  };
  
  // Advanced
  custom_css?: string;
  custom_settings?: Record<string, any>;
  
  // Metadata
  theme_name?: string;
  theme_description?: string;
}

export interface ThemePreset {
  name: string;
  description: string;
  colors: Record<string, string>;
  components: Record<string, string>;
  preview_image?: string;
}