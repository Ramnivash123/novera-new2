import { User, Bot, Copy, Check } from 'lucide-react';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ChatMessage } from '../../services/api';

interface MessageBubbleProps {
  message: ChatMessage;
}

// Renders [1], [2,3], [1,2,3] as small superscript badges inline
function renderWithCitationBadges(text: string): React.ReactNode {
  const CITATION_RE = /(\[\d+(?:,\s*\d+)*\])/g;
  const parts = text.split(CITATION_RE);
  if (parts.length === 1) return text;

  return (
    <>
      {parts.map((part, i) => {
        if (CITATION_RE.test(part)) {
          // Reset lastIndex after .test()
          CITATION_RE.lastIndex = 0;
          const nums = part.replace(/[\[\]]/g, '').split(',').map(n => n.trim());
          return (
            <sup key={i} className="inline-flex items-center gap-0.5 mx-0.5" style={{ verticalAlign: 'super', lineHeight: 0 }}>
              {nums.map((n, j) => (
                <span
                  key={j}
                  className="inline-flex items-center justify-center w-4 h-4 text-[9px] font-bold text-blue-600 bg-blue-50 border border-blue-200 rounded-full leading-none select-none"
                >
                  {n}
                </span>
              ))}
            </sup>
          );
        }
        // Reset after test
        CITATION_RE.lastIndex = 0;
        return <span key={i}>{part}</span>;
      })}
    </>
  );
}

export default function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === 'user';
  const [copied, setCopied] = useState(false);

  // Clean content: remove old [Document: X] format and strip [N] for plain text copy
  const cleanedContent = isUser
    ? message.content
    : message.content.replace(/\[Document:\s*[^\]]+\]/gi, '').trim();

  // For clipboard copy — strip citation badges too so pasted text is clean
  const copyableContent = cleanedContent.replace(/\[\d+(?:,\s*\d+)*\]/g, '').replace(/\s{2,}/g, ' ').trim();

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(copyableContent);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      fallbackCopy(copyableContent);
    }
  };

  const fallbackCopy = (text: string) => {
    const el = document.createElement('textarea');
    el.value = text;
    el.style.position = 'fixed';
    el.style.left = '-999999px';
    document.body.appendChild(el);
    el.select();
    try {
      document.execCommand('copy');
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {}
    document.body.removeChild(el);
  };

  return (
    <div className={`flex items-start gap-2 sm:gap-4 w-full ${isUser ? 'flex-row-reverse' : ''}`}>
      {/* Avatar */}
      {!isUser ? (
        <div className="w-7 h-7 sm:w-9 sm:h-9 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0 shadow-lg">
          <Bot className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
        </div>
      ) : (
        <div className="w-7 h-7 sm:w-9 sm:h-9 rounded-full bg-gradient-to-br from-gray-400 to-gray-500 flex items-center justify-center flex-shrink-0 shadow-md">
          <User className="w-4 h-4 sm:w-5 sm:h-5 text-white" />
        </div>
      )}

      {/* Message Content */}
      <div className={`flex-1 ${isUser ? 'flex justify-end' : ''} max-w-none min-w-0`}>
        <div className="relative group">
          <div
            className={`rounded-2xl px-3 py-2.5 sm:px-5 sm:py-4 ${
              isUser
                ? 'bg-gradient-to-r from-blue-600 to-blue-500 text-white shadow-lg max-w-[85%] sm:max-w-2xl'
                : 'bg-white border border-gray-200 shadow-md w-full'
            }`}
          >
            {/* Copy Button — assistant only */}
            {!isUser && (
              <button
                onClick={handleCopy}
                className={`absolute top-2 right-2 sm:top-3 sm:right-3 p-1.5 sm:p-2 rounded-lg transition-all duration-200 ${
                  copied
                    ? 'bg-green-100 text-green-600'
                    : 'bg-gray-100 hover:bg-gray-200 text-gray-600 hover:text-gray-800'
                } opacity-0 group-hover:opacity-100 focus:opacity-100 shadow-sm hover:shadow-md min-touch-target`}
                title={copied ? 'Copied!' : 'Copy response'}
                aria-label={copied ? 'Copied' : 'Copy to clipboard'}
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                ) : (
                  <Copy className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                )}
              </button>
            )}

            {isUser ? (
              <p className="text-sm sm:text-[15px] leading-relaxed whitespace-pre-wrap break-words">
                {cleanedContent}
              </p>
            ) : (
              <div className="prose prose-sm sm:prose-base max-w-none
                prose-headings:font-semibold prose-headings:text-gray-900
                prose-p:text-gray-800 prose-p:leading-relaxed prose-p:text-sm sm:prose-p:text-[15px]
                prose-a:text-blue-600 prose-a:no-underline hover:prose-a:underline
                prose-strong:text-gray-900 prose-strong:font-semibold
                prose-code:text-pink-600 prose-code:bg-pink-50 prose-code:px-1.5 prose-code:py-0.5
                prose-code:rounded prose-code:text-xs sm:prose-code:text-sm
                prose-code:before:content-[''] prose-code:after:content-['']
                prose-pre:bg-gray-900 prose-pre:text-gray-100 prose-pre:text-xs sm:prose-pre:text-sm prose-pre:overflow-x-auto
                prose-ul:list-disc prose-ul:pl-5 sm:prose-ul:pl-6 prose-ul:my-2 sm:prose-ul:my-3
                prose-ol:list-decimal prose-ol:pl-5 sm:prose-ol:pl-6 prose-ol:my-2 sm:prose-ol:my-3
                prose-li:text-gray-800 prose-li:my-1 sm:prose-li:my-2 prose-li:text-sm sm:prose-li:text-[15px]
                prose-blockquote:border-l-4 prose-blockquote:border-blue-500
                prose-blockquote:pl-3 sm:prose-blockquote:pl-4 prose-blockquote:italic
                prose-blockquote:text-gray-600 prose-blockquote:my-3 sm:prose-blockquote:my-4">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    // Intercept text nodes to render [1], [2,3] as superscript badges
                    text: ({ children }: any) => {
                      const str = String(children ?? '');
                      const CITATION_RE = /(\[\d+(?:,\s*\d+)*\])/g;
                      if (!CITATION_RE.test(str)) return <>{str}</>;
                      CITATION_RE.lastIndex = 0;
                      const parts = str.split(CITATION_RE);
                      return (
                        <>
                          {parts.map((part, i) => {
                            const isCite = /^\[\d+(?:,\s*\d+)*\]$/.test(part);
                            if (isCite) {
                              const nums = part.replace(/[\[\]]/g, '').split(',').map(n => n.trim());
                              return (
                                <sup key={i} className="inline-flex items-center gap-0.5 mx-0.5" style={{ verticalAlign: 'super', lineHeight: 0 }}>
                                  {nums.map((n, j) => (
                                    <span
                                      key={j}
                                      className="inline-flex items-center justify-center w-4 h-4 text-[9px] font-bold text-blue-600 bg-blue-50 border border-blue-200 rounded-full leading-none select-none"
                                    >
                                      {n}
                                    </span>
                                  ))}
                                </sup>
                              );
                            }
                            return <span key={i}>{part}</span>;
                          })}
                        </>
                      );
                    },
                    h1: ({ node, ...props }) => (
                      <h1 className="text-xl sm:text-2xl font-bold text-gray-900 mt-4 sm:mt-6 mb-2 sm:mb-3" {...props} />
                    ),
                    h2: ({ node, ...props }) => (
                      <h2 className="text-lg sm:text-xl font-semibold text-gray-900 mt-4 sm:mt-5 mb-2 sm:mb-3" {...props} />
                    ),
                    h3: ({ node, ...props }) => (
                      <h3 className="text-base sm:text-lg font-semibold text-gray-900 mt-3 sm:mt-4 mb-2" {...props} />
                    ),
                    p: ({ node, ...props }) => (
                      <p className="text-gray-800 leading-relaxed my-2 sm:my-3 text-sm sm:text-[15px] break-words" {...props} />
                    ),
                    ul: ({ node, ...props }) => (
                      <ul className="list-disc pl-5 sm:pl-6 my-2 sm:my-3 space-y-1 sm:space-y-2" {...props} />
                    ),
                    ol: ({ node, ...props }) => (
                      <ol className="list-decimal pl-5 sm:pl-6 my-2 sm:my-3 space-y-1 sm:space-y-2" {...props} />
                    ),
                    li: ({ node, ...props }) => (
                      <li className="text-gray-800 leading-relaxed text-sm sm:text-[15px] break-words" {...props} />
                    ),
                    blockquote: ({ node, ...props }) => (
                      <blockquote className="border-l-4 border-blue-500 pl-3 sm:pl-4 py-2 italic text-gray-600 my-3 sm:my-4 bg-blue-50 rounded-r break-words" {...props} />
                    ),
                    code: ({ node, inline, ...props }: any) =>
                      inline ? (
                        <code className="bg-pink-50 text-pink-600 px-1.5 py-0.5 rounded text-xs sm:text-sm font-mono break-words" {...props} />
                      ) : (
                        <code className="block bg-gray-900 text-gray-100 p-3 sm:p-4 rounded-lg overflow-x-auto text-xs sm:text-sm font-mono my-2 sm:my-3 leading-relaxed" {...props} />
                      ),
                    a: ({ node, ...props }) => (
                      <a className="text-blue-600 hover:text-blue-700 hover:underline font-medium break-all" target="_blank" rel="noopener noreferrer" {...props} />
                    ),
                    table: ({ node, ...props }) => (
                      <div className="overflow-x-auto my-3 sm:my-5 rounded-lg border border-gray-300 shadow-md -mx-3 sm:mx-0">
                        <table className="min-w-full divide-y divide-gray-300 text-xs sm:text-sm" {...props} />
                      </div>
                    ),
                    thead: ({ node, ...props }) => (
                      <thead className="bg-gradient-to-r from-blue-50 to-indigo-50" {...props} />
                    ),
                    tbody: ({ node, ...props }) => (
                      <tbody className="bg-white divide-y divide-gray-200" {...props} />
                    ),
                    tr: ({ node, ...props }) => (
                      <tr className="hover:bg-blue-50 transition-colors duration-150" {...props} />
                    ),
                    th: ({ node, ...props }) => (
                      <th className="px-3 sm:px-5 py-2.5 sm:py-4 text-left text-xs font-bold text-gray-800 uppercase tracking-wider border-b-2 border-gray-400" {...props} />
                    ),
                    td: ({ node, ...props }) => (
                      <td className="px-3 sm:px-5 py-2.5 sm:py-4 text-xs sm:text-[15px] text-gray-800 whitespace-normal break-words" {...props} />
                    ),
                    hr: ({ node, ...props }) => (
                      <hr className="my-4 sm:my-6 border-t-2 border-gray-300" {...props} />
                    ),
                  }}
                >
                  {cleanedContent}
                </ReactMarkdown>
              </div>
            )}

            {/* Confidence badge — assistant only */}
            {!isUser && message.metadata?.confidence && (
              <div className="mt-3 sm:mt-4 pt-3 sm:pt-4 border-t border-gray-200">
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-gray-500 font-medium">Confidence:</span>
                  <span
                    className={`px-2 sm:px-2.5 py-0.5 sm:py-1 rounded-full font-semibold text-xs ${
                      message.metadata.confidence === 'high'
                        ? 'bg-green-100 text-green-700'
                        : message.metadata.confidence === 'medium'
                        ? 'bg-yellow-100 text-yellow-700'
                        : 'bg-gray-100 text-gray-600'
                    }`}
                  >
                    {message.metadata.confidence}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Timestamp */}
          {message.timestamp && (
            <p className={`text-xs text-gray-400 mt-1 sm:mt-2 ${isUser ? 'text-right' : 'text-left'}`}>
              {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}