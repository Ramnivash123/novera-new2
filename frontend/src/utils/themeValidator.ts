export interface ContrastResult {
  ratio: number;
  passesAA: boolean;
  passesAAA: boolean;
  rating: 'fail' | 'aa' | 'aaa';
}

export interface ValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
  contrast: {
    primaryOnBackground: ContrastResult;
    textOnBackground: ContrastResult;
    buttonContrast: ContrastResult;
  };
}

export class ThemeValidator {
  private static hexToRgb(hex: string): { r: number; g: number; b: number } | null {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result
      ? {
          r: parseInt(result[1], 16),
          g: parseInt(result[2], 16),
          b: parseInt(result[3], 16),
        }
      : null;
  }

  private static getLuminance(hex: string): number {
    const rgb = this.hexToRgb(hex);
    if (!rgb) return 0;

    const [r, g, b] = [rgb.r, rgb.g, rgb.b].map((val) => {
      const normalized = val / 255;
      return normalized <= 0.03928
        ? normalized / 12.92
        : Math.pow((normalized + 0.055) / 1.055, 2.4);
    });

    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  static calculateContrast(color1: string, color2: string): ContrastResult {
    const l1 = this.getLuminance(color1);
    const l2 = this.getLuminance(color2);
    const ratio =
      (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);

    const roundedRatio = Math.round(ratio * 100) / 100;

    return {
      ratio: roundedRatio,
      passesAA: ratio >= 4.5,
      passesAAA: ratio >= 7,
      rating: ratio >= 7 ? 'aaa' : ratio >= 4.5 ? 'aa' : 'fail',
    };
  }

  static validateColorHex(color: string): boolean {
    return /^#[0-9A-F]{6}$/i.test(color);
  }

  static validateTheme(formData: any): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    const colorFields = [
      'primary_color',
      'secondary_color',
      'accent_color',
      'background_color',
      'text_primary_color',
      'button_primary_color',
      'button_text_color',
    ];

    colorFields.forEach((field) => {
      const value = formData[field];
      if (value && !this.validateColorHex(value)) {
        errors.push(`${field.replace('_', ' ')} must be a valid hex color (e.g., #000000)`);
      }
    });

    const primaryOnBackground = this.calculateContrast(
      formData.primary_color || '#0ea5e9',
      formData.background_color || '#ffffff'
    );

    const textOnBackground = this.calculateContrast(
      formData.text_primary_color || '#111827',
      formData.background_color || '#ffffff'
    );

    const buttonContrast = this.calculateContrast(
      formData.button_text_color || '#ffffff',
      formData.button_primary_color || '#0ea5e9'
    );

    if (!textOnBackground.passesAA) {
      errors.push(
        `Text contrast (${textOnBackground.ratio}:1) fails WCAG AA standards. Minimum required: 4.5:1`
      );
    }

    if (!buttonContrast.passesAA) {
      warnings.push(
        `Button text contrast (${buttonContrast.ratio}:1) may be hard to read. Consider adjusting colors.`
      );
    }

    if (!primaryOnBackground.passesAA) {
      warnings.push(
        `Primary color on background has low contrast (${primaryOnBackground.ratio}:1). May affect accessibility.`
      );
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings,
      contrast: {
        primaryOnBackground,
        textOnBackground,
        buttonContrast,
      },
    };
  }

  static generateDarkModeColors(lightColors: any): any {
    const darkColors: any = {};

    const adjustForDarkMode = (hex: string, type: 'background' | 'text' | 'accent'): string => {
      const rgb = this.hexToRgb(hex);
      if (!rgb) return hex;

      let { r, g, b } = rgb;

      if (type === 'background') {
        r = Math.max(0, Math.min(255, Math.floor(r * 0.15)));
        g = Math.max(0, Math.min(255, Math.floor(g * 0.15)));
        b = Math.max(0, Math.min(255, Math.floor(b * 0.15)));
      } else if (type === 'text') {
        r = Math.max(0, Math.min(255, Math.floor(r + (255 - r) * 0.8)));
        g = Math.max(0, Math.min(255, Math.floor(g + (255 - g) * 0.8)));
        b = Math.max(0, Math.min(255, Math.floor(b + (255 - b) * 0.8)));
      } else {
        r = Math.max(0, Math.min(255, Math.floor(r * 1.2)));
        g = Math.max(0, Math.min(255, Math.floor(g * 1.2)));
        b = Math.max(0, Math.min(255, Math.floor(b * 1.2)));
      }

      return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
    };

    darkColors.background = adjustForDarkMode(lightColors.background_color, 'background');
    darkColors.background_secondary = adjustForDarkMode(
      lightColors.background_secondary,
      'background'
    );
    darkColors.background_tertiary = adjustForDarkMode(
      lightColors.background_tertiary,
      'background'
    );

    darkColors.text_primary = adjustForDarkMode(lightColors.text_primary_color, 'text');
    darkColors.text_secondary = adjustForDarkMode(lightColors.text_secondary_color, 'text');

    darkColors.primary = lightColors.primary_color;
    darkColors.secondary = lightColors.secondary_color;
    darkColors.accent = lightColors.accent_color;

    darkColors.border = '#374151';
    darkColors.shadow = '#00000040';

    return darkColors;
  }

  static suggestAccessibleTextColor(backgroundColor: string): string {
    const luminance = this.getLuminance(backgroundColor);
    return luminance > 0.5 ? '#111827' : '#f9fafb';
  }

  static suggestComplementaryColor(baseColor: string): string {
    const rgb = this.hexToRgb(baseColor);
    if (!rgb) return baseColor;

    const complementary = {
      r: 255 - rgb.r,
      g: 255 - rgb.g,
      b: 255 - rgb.b,
    };

    return `#${((1 << 24) + (complementary.r << 16) + (complementary.g << 8) + complementary.b)
      .toString(16)
      .slice(1)}`;
  }
}