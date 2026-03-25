# Novera Frontend - React Application

## Overview

Modern React frontend for the Novera RAG system. Built with React 18, TypeScript, and Tailwind CSS.

## Features

### 🎨 **User Interface**
- Clean, modern design with gradient accents
- Responsive layout (mobile, tablet, desktop)
- Dark mode ready
- Smooth animations and transitions

### 💬 **Chat Interface**
- Real-time chat with AI assistant
- Message history and conversation threads
- Source citations displayed inline
- Confidence indicators
- Loading states and error handling
- Example questions for quick start

### 📄 **Document Management**
- Upload documents (PDF, DOCX, TXT, XLSX)
- Drag-and-drop support
- Processing status tracking
- Document statistics dashboard
- Search and filter documents
- Delete documents

### 📚 **Conversation History**
- View all past conversations
- Continue previous chats
- Delete conversations
- Conversation metadata (date, message count)

## Tech Stack

- **Framework**: React 18
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Routing**: React Router v6
- **HTTP Client**: Axios
- **Icons**: Lucide React
- **Markdown**: React Markdown
- **Build Tool**: Vite
- **Date Formatting**: date-fns

## Project Structure

```
frontend/
├── public/
├── src/
│   ├── components/
│   │   ├── chat/
│   │   │   ├── MessageBubble.tsx
│   │   │   └── SourceCard.tsx
│   │   ├── documents/
│   │   │   ├── DocumentCard.tsx
│   │   │   └── UploadModal.tsx
│   │   └── Layout.tsx
│   ├── pages/
│   │   ├── ChatPage.tsx
│   │   ├── DocumentsPage.tsx
│   │   └── ConversationsPage.tsx
│   ├── services/
│   │   └── api.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── index.html
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── tsconfig.json
```

## Installation

### Prerequisites
- Node.js 18+
- npm or yarn
- Backend running on http://localhost:8000

### Quick Start

```bash
# 1. Navigate to frontend directory
cd frontend

# 2. Install dependencies
npm install

# 3. Create .env file
cp .env.example .env

# 4. Start development server
npm run dev
```

### Using Setup Script

```bash
chmod +x setup_frontend.sh
./setup_frontend.sh
```

## Environment Variables

Create a `.env` file in the frontend directory:

```bash
# API Configuration
VITE_API_URL=http://localhost:8000

# App Configuration
VITE_APP_NAME=Mentanova AI
VITE_APP_VERSION=1.0.0
```

## Available Scripts

```bash
# Development server (http://localhost:3000)
npm run dev

# Type checking
npm run build

# Preview production build
npm run preview

# Linting
npm run lint
```

## Pages

### 1. Chat Page (`/chat`)

**Features:**
- Send messages to AI assistant
- View conversation history
- See source citations
- Confidence indicators
- Example questions

**Components:**
- `ChatPage.tsx` - Main chat interface
- `MessageBubble.tsx` - Individual message display
- `SourceCard.tsx` - Source citation cards

### 2. Documents Page (`/documents`)

**Features:**
- Upload new documents
- View all documents
- Filter by status/type
- Search documents
- Delete documents
- Processing status tracking

**Components:**
- `DocumentsPage.tsx` - Main documents interface
- `DocumentCard.tsx` - Individual document card
- `UploadModal.tsx` - Upload dialog

### 3. Conversations Page (`/conversations`)

**Features:**
- List all conversations
- Continue previous chats
- Delete conversations
- View conversation metadata

**Components:**
- `ConversationsPage.tsx` - Conversation list

## API Integration

The frontend communicates with the backend via the API service (`src/services/api.ts`).

### Endpoints Used

```typescript
// Chat
POST /api/v1/chat
GET /api/v1/chat/conversations
GET /api/v1/chat/conversations/{id}
DELETE /api/v1/chat/conversations/{id}

// Documents
POST /api/v1/documents/upload
GET /api/v1/documents
GET /api/v1/documents/{id}/status
DELETE /api/v1/documents/{id}

// Health
GET /api/v1/health
```

### Example Usage

```typescript
import api from './services/api';

// Send chat message
const response = await api.sendChatMessage({
  query: "What is the Q4 revenue?",
  conversation_id: null
});

// Upload document
const result = await api.uploadDocument(
  file,
  'finance',
  'Finance Department'
);

// Get conversations
const conversations = await api.getConversations(10);
```

## Styling

### Tailwind CSS

Custom theme in `tailwind.config.js`:

```javascript
colors: {
  primary: {
    500: '#0ea5e9',  // Main blue
    600: '#0284c7',
  },
  secondary: {
    500: '#d946ef',  // Accent purple
    600: '#c026d3',
  },
}
```

### Custom Styles

Global styles in `src/index.css`:
- Custom scrollbar
- Markdown prose styles
- Animation classes

## Components

### Layout Component

Responsive layout with:
- Collapsible sidebar
- Mobile-friendly navigation
- User profile section

### MessageBubble Component

Displays chat messages with:
- User/assistant differentiation
- Markdown rendering
- Metadata (confidence, sources)

### SourceCard Component

Shows source citations:
- Document name
- Page number
- Section title
- Quick view button

### DocumentCard Component

Document information display:
- Status indicator
- Statistics (pages, chunks)
- Action buttons (delete, refresh)

### UploadModal Component

Document upload interface:
- Drag-and-drop
- File validation
- Progress bar
- Form fields (type, department)

## Responsive Design

### Breakpoints

- Mobile: < 768px
- Tablet: 768px - 1024px
- Desktop: > 1024px

### Mobile Features
- Collapsible sidebar
- Touch-friendly buttons
- Optimized layouts
- Hamburger menu

## Production Build

### Build for Production

```bash
npm run build
```

Output: `dist/` directory

### Using Docker

```bash
# Build image
docker build -t mentanova-frontend .

# Run container
docker run -p 80:80 mentanova-frontend
```

### Deploy to Static Hosting

Build artifacts in `dist/` can be deployed to:
- Netlify
- Vercel
- AWS S3 + CloudFront
- GitHub Pages
- Render Static Site

## Performance Optimization

### Implemented
- Code splitting
- Lazy loading
- Image optimization
- Bundle size optimization
- Gzip compression (nginx)

### Recommendations
- Enable CDN for assets
- Implement service workers
- Add caching strategies
- Use React.memo for expensive components

## Accessibility

### Features
- Semantic HTML
- ARIA labels
- Keyboard navigation
- Focus indicators
- Alt text for images

### Testing
```bash
# Install accessibility testing tools
npm install -D @axe-core/react

# Run accessibility audits
npm run build
npm run preview
# Use browser DevTools Lighthouse
```

## Browser Support

- Chrome (last 2 versions)
- Firefox (last 2 versions)
- Safari (last 2 versions)
- Edge (last 2 versions)

## Troubleshooting

### Common Issues

**Problem**: API requests fail with CORS errors
- **Solution**: Ensure backend CORS is configured to allow `http://localhost:3000`

**Problem**: Build fails with TypeScript errors
- **Solution**: Run `npm run build` to see specific errors, fix type issues

**Problem**: Styles not loading
- **Solution**: Clear build cache: `rm -rf node_modules/.vite`

**Problem**: Slow development server
- **Solution**: Reduce file watching, exclude node_modules

## Development Tips

### Hot Reload
Vite provides instant hot module replacement (HMR). Changes reflect immediately.

### TypeScript
Use proper types for better development experience:

```typescript
import { ChatMessage, Source } from '../services/api';

const [messages, setMessages] = useState<ChatMessage[]>([]);
```

### Debugging
```typescript
// Enable debug mode
localStorage.setItem('debug', 'true');

// View API calls in console
```

## Future Enhancements

- [ ] Real-time streaming responses
- [ ] Voice input
- [ ] Document preview
- [ ] Advanced search filters
- [ ] User preferences
- [ ] Themes (dark mode)
- [ ] Internationalization (i18n)
- [ ] Export conversations
- [ ] Share conversations

## Contributing

1. Follow TypeScript best practices
2. Use Tailwind CSS classes
3. Add proper types
4. Test on multiple screen sizes
5. Follow component structure

## Support

- **Documentation**: This README
- **API Docs**: http://localhost:8000/api/docs
- **Issues**: GitHub Issues

---

**Built by Harish with ❤️ using React + TypeScript**