/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      screens: {
        'xs': '475px',
        'sm': '640px',
        'md': '768px',
        'lg': '1024px',
        'xl': '1280px',
        '2xl': '1536px',
      },
      colors: {
        // Map Tailwind color names to CSS variables
        primary: {
          DEFAULT: 'var(--color-primary)',
          50: 'var(--color-primary-50)',
          100: 'var(--color-primary-100)',
          200: 'var(--color-primary-200)',
          300: 'var(--color-primary-300)',
          400: 'var(--color-primary-400)',
          500: 'var(--color-primary-500)',
          600: 'var(--color-primary-600)',
          700: 'var(--color-primary-700)',
          800: 'var(--color-primary-800)',
          900: 'var(--color-primary-900)',
        },
        secondary: {
          DEFAULT: 'var(--color-secondary)',
          50: 'var(--color-secondary-50)',
          100: 'var(--color-secondary-100)',
          200: 'var(--color-secondary-200)',
          300: 'var(--color-secondary-300)',
          400: 'var(--color-secondary-400)',
          500: 'var(--color-secondary-500)',
          600: 'var(--color-secondary-600)',
          700: 'var(--color-secondary-700)',
          800: 'var(--color-secondary-800)',
          900: 'var(--color-secondary-900)',
        },
        accent: {
          DEFAULT: 'var(--color-accent)',
          50: 'var(--color-accent-50)',
          100: 'var(--color-accent-100)',
          200: 'var(--color-accent-200)',
          300: 'var(--color-accent-300)',
          400: 'var(--color-accent-400)',
          500: 'var(--color-accent-500)',
          600: 'var(--color-accent-600)',
          700: 'var(--color-accent-700)',
          800: 'var(--color-accent-800)',
          900: 'var(--color-accent-900)',
        },
        success: 'var(--color-success)',
        warning: 'var(--color-warning)',
        error: 'var(--color-error)',
        info: 'var(--color-info)',
        
        // Background colors
        background: {
          DEFAULT: 'var(--color-background)',
          secondary: 'var(--color-background-secondary)',
          tertiary: 'var(--color-background-tertiary)',
        },
        
        // Text colors
        text: {
          primary: 'var(--color-text-primary)',
          secondary: 'var(--color-text-secondary)',
        },
        
        // Border and shadow
        border: 'var(--color-border)',
        
        // Component-specific
        sidebar: 'var(--color-sidebar)',
        card: {
          DEFAULT: 'var(--card-background)',
          border: 'var(--card-border-color)',
        },
        nav: {
          DEFAULT: 'var(--nav-background)',
          text: 'var(--nav-text-color)',
          active: 'var(--nav-active-color)',
          hover: 'var(--nav-hover-color)',
        },
        input: {
          border: 'var(--input-border-color)',
          focus: 'var(--input-focus-color)',
        },
        button: {
          primary: {
            DEFAULT: 'var(--button-primary-bg)',
            text: 'var(--button-primary-text)',
          },
          secondary: {
            DEFAULT: 'var(--button-secondary-bg)',
            text: 'var(--button-secondary-text)',
          },
        },
      },
      borderRadius: {
        DEFAULT: 'var(--border-radius)',
        'button': 'var(--button-border-radius)',
        'input': 'var(--input-border-radius)',
        'card': 'var(--card-border-radius)',
      },
      boxShadow: {
        'card': 'var(--card-shadow)',
        'theme': 'var(--color-shadow)',
      },
      spacing: {
        'xs': 'var(--spacing-xs)',
        'sm': 'var(--spacing-sm)',
        'md': 'var(--spacing-md)',
        'lg': 'var(--spacing-lg)',
        'xl': 'var(--spacing-xl)',
      },
      fontFamily: {
        'theme': 'var(--font-family)',
      },
      fontSize: {
        'theme-base': 'var(--font-size-base)',
        'theme-heading': 'var(--font-size-heading)',
      },
      fontWeight: {
        'theme-normal': 'var(--font-weight-normal)',
        'theme-medium': 'var(--font-weight-medium)',
        'theme-bold': 'var(--font-weight-bold)',
      },
      lineHeight: {
        'theme-base': 'var(--line-height-base)',
        'theme-heading': 'var(--line-height-heading)',
      },
      letterSpacing: {
        'theme': 'var(--letter-spacing)',
      },
      transitionDuration: {
        'theme': 'var(--animation-speed)',
      },
      keyframes: {
        'pulse-slow': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '.5' },
        },
        slideIn: {
          '0%': { 
            transform: 'translateX(100%)',
            opacity: '0'
          },
          '100%': { 
            transform: 'translateX(0)',
            opacity: '1'
          },
        },
        slideOut: {
          '0%': { 
            transform: 'translateX(0)',
            opacity: '1'
          },
          '100%': { 
            transform: 'translateX(100%)',
            opacity: '0'
          },
        },
        slideUp: {
          '0%': { 
            transform: 'translateY(100%)',
            opacity: '0'
          },
          '100%': { 
            transform: 'translateY(0)',
            opacity: '1'
          },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideInLeft: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(0)' },
        },
        slideOutLeft: {
          '0%': { transform: 'translateX(0)' },
          '100%': { transform: 'translateX(-100%)' },
        },
        shimmer: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' },
        },
      },
      animation: {
        'pulse-slow': 'pulse-slow 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slideIn': 'slideIn 0.3s ease-out',
        'slideOut': 'slideOut 0.3s ease-out',
        'slideUp': 'slideUp 0.3s ease-out',
        'fadeIn': 'fadeIn 0.3s ease-out',
        'slideInLeft': 'slideInLeft 0.3s ease-out',
        'slideOutLeft': 'slideOutLeft 0.3s ease-out',
        'shimmer': 'shimmer 2s infinite',
      },
    },
  },
  plugins: [],
}