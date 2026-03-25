import { FileText, MessageSquare, TrendingUp, Clock } from 'lucide-react';

interface AnalyticsProps {
  analytics: {
    total_messages: number;
    documents_referenced: string[];
    confidence_distribution: {
      high: number;
      medium: number;
      low: number;
    };
    duration_minutes: number;
  };
}

export default function ConversationAnalytics({ analytics }: AnalyticsProps) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Conversation Insights</h3>
      
      <div className="grid grid-cols-2 gap-4">
        {/* Total Messages */}
        <div className="flex items-center gap-3 p-3 bg-blue-50 rounded-lg">
          <MessageSquare className="w-8 h-8 text-blue-600" />
          <div>
            <p className="text-sm text-gray-600">Messages</p>
            <p className="text-2xl font-bold text-gray-900">{analytics.total_messages}</p>
          </div>
        </div>
        
        {/* Documents */}
        <div className="flex items-center gap-3 p-3 bg-green-50 rounded-lg">
          <FileText className="w-8 h-8 text-green-600" />
          <div>
            <p className="text-sm text-gray-600">Documents</p>
            <p className="text-2xl font-bold text-gray-900">{analytics.documents_referenced.length}</p>
          </div>
        </div>
        
        {/* Duration */}
        <div className="flex items-center gap-3 p-3 bg-purple-50 rounded-lg">
          <Clock className="w-8 h-8 text-purple-600" />
          <div>
            <p className="text-sm text-gray-600">Duration</p>
            <p className="text-2xl font-bold text-gray-900">{analytics.duration_minutes}m</p>
          </div>
        </div>
        
        {/* Confidence */}
        <div className="flex items-center gap-3 p-3 bg-yellow-50 rounded-lg">
          <TrendingUp className="w-8 h-8 text-yellow-600" />
          <div>
            <p className="text-sm text-gray-600">High Confidence</p>
            <p className="text-2xl font-bold text-gray-900">{analytics.confidence_distribution.high}</p>
          </div>
        </div>
      </div>
      
      {/* Documents List */}
      {analytics.documents_referenced.length > 0 && (
        <div className="mt-4">
          <p className="text-sm font-medium text-gray-700 mb-2">Documents Referenced:</p>
          <div className="space-y-1">
            {analytics.documents_referenced.map((doc, idx) => (
              <div key={idx} className="text-sm text-gray-600 flex items-center gap-2">
                <FileText className="w-3 h-3" />
                {doc}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}