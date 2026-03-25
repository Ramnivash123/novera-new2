export interface CSSTemplate {
  id: string;
  name: string;
  description: string;
  category: 'effects' | 'layouts' | 'animations' | 'utilities';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  cssCode: string;
  preview?: string;
}

export const CSS_TEMPLATES: CSSTemplate[] = [
  {
    id: 'glassmorphism',
    name: 'Glassmorphism Cards',
    description: 'Modern frosted glass effect for cards and panels',
    category: 'effects',
    difficulty: 'beginner',
    cssCode: `/* Glassmorphism Effect */
.glass-card {
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
}

/* Apply to specific components */
.bg-white {
  background: rgba(255, 255, 255, 0.95) !important;
  backdrop-filter: blur(10px);
}`,
    preview: 'Frosted glass cards with blur effect'
  },
  {
    id: 'neumorphism',
    name: 'Neumorphism (Soft UI)',
    description: 'Soft, extruded design style with subtle shadows',
    category: 'effects',
    difficulty: 'intermediate',
    cssCode: `/* Neumorphism Effect */
.neomorph-card {
  background: #e0e5ec;
  border-radius: 12px;
  box-shadow: 
    9px 9px 16px rgba(163, 177, 198, 0.6),
    -9px -9px 16px rgba(255, 255, 255, 0.5);
}

.neomorph-button {
  background: linear-gradient(145deg, #e6e6e6, #ffffff);
  box-shadow: 
    5px 5px 10px #d1d1d1,
    -5px -5px 10px #ffffff;
}

.neomorph-button:active {
  box-shadow: 
    inset 5px 5px 10px #d1d1d1,
    inset -5px -5px 10px #ffffff;
}`,
    preview: 'Soft, embossed UI elements'
  },
  {
    id: 'gradient-borders',
    name: 'Gradient Borders',
    description: 'Animated gradient borders for modern look',
    category: 'effects',
    difficulty: 'intermediate',
    cssCode: `/* Gradient Border Effect */
.gradient-border {
  position: relative;
  background: #fff;
  border-radius: 12px;
  padding: 2px;
  background: linear-gradient(
    90deg,
    var(--color-primary),
    var(--color-secondary),
    var(--color-accent)
  );
  background-size: 200% 200%;
  animation: gradientShift 3s ease infinite;
}

.gradient-border > * {
  background: #fff;
  border-radius: 10px;
  padding: 1rem;
}

@keyframes gradientShift {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}`,
    preview: 'Animated gradient borders'
  },
  {
    id: 'hover-lift',
    name: 'Hover Lift Animation',
    description: 'Smooth lift effect on hover for cards',
    category: 'animations',
    difficulty: 'beginner',
    cssCode: `/* Hover Lift Effect */
.hover-lift {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.hover-lift:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.15);
}

/* Apply to document cards */
.bg-white:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
}`,
    preview: 'Cards lift up on hover'
  },
  {
    id: 'smooth-transitions',
    name: 'Smooth Page Transitions',
    description: 'Add smooth fade-in animations to page loads',
    category: 'animations',
    difficulty: 'beginner',
    cssCode: `/* Smooth Page Transitions */
@keyframes pageEnter {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

main {
  animation: pageEnter 0.4s ease-out;
}

/* Stagger child elements */
.animate-children > * {
  animation: pageEnter 0.4s ease-out;
  animation-fill-mode: backwards;
}

.animate-children > *:nth-child(1) { animation-delay: 0.1s; }
.animate-children > *:nth-child(2) { animation-delay: 0.2s; }
.animate-children > *:nth-child(3) { animation-delay: 0.3s; }`,
    preview: 'Smooth fade-in on page load'
  },
  {
    id: 'rainbow-text',
    name: 'Rainbow Text Effect',
    description: 'Animated rainbow gradient text',
    category: 'effects',
    difficulty: 'intermediate',
    cssCode: `/* Rainbow Text Effect */
.rainbow-text {
  background: linear-gradient(
    90deg,
    #ff0000,
    #ff7f00,
    #ffff00,
    #00ff00,
    #0000ff,
    #4b0082,
    #9400d3
  );
  background-size: 200% auto;
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: rainbowShift 3s linear infinite;
}

@keyframes rainbowShift {
  to { background-position: 200% center; }
}

/* Apply to headings */
h1, h2 {
  background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}`,
    preview: 'Animated rainbow text'
  },
  {
    id: 'scroll-reveal',
    name: 'Scroll Reveal Animation',
    description: 'Elements fade in as you scroll',
    category: 'animations',
    difficulty: 'advanced',
    cssCode: `/* Scroll Reveal Animation */
.reveal {
  opacity: 0;
  transform: translateY(30px);
  transition: all 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.reveal.active {
  opacity: 1;
  transform: translateY(0);
}

/* Auto-apply to cards */
.bg-white {
  opacity: 0;
  animation: revealCard 0.6s ease-out forwards;
}

@keyframes revealCard {
  from {
    opacity: 0;
    transform: scale(0.95);
  }
  to {
    opacity: 1;
    transform: scale(1);
  }
}`,
    preview: 'Elements animate on scroll'
  },
  {
    id: 'custom-scrollbar',
    name: 'Styled Scrollbars',
    description: 'Beautiful custom scrollbar design',
    category: 'utilities',
    difficulty: 'beginner',
    cssCode: `/* Custom Scrollbar Styling */
::-webkit-scrollbar {
  width: 10px;
  height: 10px;
}

::-webkit-scrollbar-track {
  background: var(--color-background-secondary);
  border-radius: 10px;
}

::-webkit-scrollbar-thumb {
  background: linear-gradient(
    180deg,
    var(--color-primary),
    var(--color-secondary)
  );
  border-radius: 10px;
  border: 2px solid var(--color-background-secondary);
}

::-webkit-scrollbar-thumb:hover {
  background: var(--color-primary);
}

/* Firefox */
* {
  scrollbar-width: thin;
  scrollbar-color: var(--color-primary) var(--color-background-secondary);
}`,
    preview: 'Gradient scrollbars matching theme'
  },
  {
    id: 'focus-glow',
    name: 'Focus Glow Effect',
    description: 'Beautiful glow on focused inputs',
    category: 'effects',
    difficulty: 'beginner',
    cssCode: `/* Focus Glow Effect */
input:focus,
textarea:focus,
select:focus {
  outline: none;
  border-color: var(--color-primary);
  box-shadow: 
    0 0 0 3px rgba(var(--color-primary-rgb), 0.1),
    0 0 20px rgba(var(--color-primary-rgb), 0.2);
  transform: scale(1.01);
  transition: all 0.3s ease;
}

button:focus {
  outline: none;
  box-shadow: 
    0 0 0 3px rgba(var(--color-primary-rgb), 0.3),
    0 0 20px rgba(var(--color-primary-rgb), 0.4);
}`,
    preview: 'Glowing inputs on focus'
  },
  {
    id: 'loading-skeleton',
    name: 'Skeleton Loading',
    description: 'Smooth skeleton screens for loading states',
    category: 'utilities',
    difficulty: 'intermediate',
    cssCode: `/* Skeleton Loading Animation */
@keyframes shimmer {
  0% {
    background-position: -1000px 0;
  }
  100% {
    background-position: 1000px 0;
  }
}

.skeleton {
  background: linear-gradient(
    90deg,
    var(--color-background-secondary) 0%,
    var(--color-background-tertiary) 50%,
    var(--color-background-secondary) 100%
  );
  background-size: 1000px 100%;
  animation: shimmer 2s infinite linear;
  border-radius: 8px;
}

.skeleton-text {
  height: 1rem;
  margin-bottom: 0.5rem;
}

.skeleton-avatar {
  width: 40px;
  height: 40px;
  border-radius: 50%;
}`,
    preview: 'Animated loading skeletons'
  }
];

export function getTemplatesByCategory(category: CSSTemplate['category']): CSSTemplate[] {
  return CSS_TEMPLATES.filter(t => t.category === category);
}

export function getTemplateById(id: string): CSSTemplate | undefined {
  return CSS_TEMPLATES.find(t => t.id === id);
}

export function getBeginnerTemplates(): CSSTemplate[] {
  return CSS_TEMPLATES.filter(t => t.difficulty === 'beginner');
}