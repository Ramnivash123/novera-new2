import { FileText, Calendar, Trash2, CheckCircle, Clock, AlertCircle, Eye, Edit } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { Document } from '../../services/api';

interface DocumentCardProps {
  document: Document;
  onDelete: (id: string) => void;
  onRefresh: () => void;
  canDelete?: boolean;
}

export default function DocumentCard({ document, onDelete, canDelete = false }: DocumentCardProps) {
  const navigate = useNavigate();

  const getStatusIcon = () => {
    switch (document.status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 sm:w-5 sm:h-5 text-green-500" />;
      case 'processing':
        return <Clock className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-500 animate-spin" />;
      case 'failed':
        return <AlertCircle className="w-4 h-4 sm:w-5 sm:h-5 text-red-500" />;
      default:
        return <FileText className="w-4 h-4 sm:w-5 sm:h-5 text-gray-400" />;
    }
  };

  const getStatusColor = () => {
    switch (document.status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'processing':
        return 'bg-yellow-100 text-yellow-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  const handleEdit = () => {
    if (document.status === 'completed') {
      navigate(`/documents/${document.id}/edit`);
    }
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 hover:shadow-md transition-shadow overflow-hidden">
      <div className="p-3 sm:p-4">
        {/* Header */}
        <div className="flex items-start justify-between mb-2 sm:mb-3 gap-2">
          <div className="flex items-center gap-2 flex-1 min-w-0">
            {getStatusIcon()}
            <h3 className="font-medium text-sm sm:text-base text-gray-900 truncate" title={document.filename}>
              {document.filename}
            </h3>
          </div>
        </div>

        {/* Status Badge */}
        <div className="mb-2 sm:mb-3">
          <span className={`inline-flex items-center px-2 sm:px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor()}`}>
            {document.status}
          </span>
        </div>

        {/* Metadata - Responsive Grid */}
        <div className="space-y-1.5 sm:space-y-2 text-xs sm:text-sm text-gray-600">
          <div className="flex items-center justify-between">
            <span>Type:</span>
            <span className="font-medium text-gray-900 truncate ml-2">{document.doc_type}</span>
          </div>
          {document.department && (
            <div className="flex items-center justify-between">
              <span>Department:</span>
              <span className="font-medium text-gray-900 truncate ml-2">{document.department}</span>
            </div>
          )}
          <div className="grid grid-cols-2 gap-2">
            <div className="flex items-center justify-between">
              <span>Pages:</span>
              <span className="font-medium text-gray-900">{document.total_pages}</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Chunks:</span>
              <span className="font-medium text-gray-900">{document.total_chunks}</span>
            </div>
          </div>
          <div className="flex items-center gap-1 text-xs text-gray-500 pt-1">
            <Calendar className="w-3 h-3 flex-shrink-0" />
            <span className="truncate">{new Date(document.upload_date).toLocaleDateString()}</span>
          </div>
        </div>

        {/* Actions - Responsive */}
        <div className="mt-3 sm:mt-4 pt-3 sm:pt-4 border-t border-gray-200 flex items-center justify-between gap-2">
          <button
            onClick={handleEdit}
            disabled={document.status !== 'completed'}
            className="flex items-center gap-1 text-xs sm:text-sm text-primary-600 hover:text-primary-700 disabled:text-gray-400 disabled:cursor-not-allowed transition-colors min-touch-target flex-1 justify-center sm:justify-start"
            title={document.status === 'completed' ? 'View and edit chunks' : 'Document must be processed first'}
          >
            {document.status === 'completed' ? (
              <>
                <Edit className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                <span>Edit</span>
              </>
            ) : (
              <>
                <Eye className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                <span>View</span>
              </>
            )}
          </button>
          
          {canDelete ? (
            <button
              onClick={() => onDelete(document.id)}
              className="flex items-center gap-1 text-xs sm:text-sm text-red-600 hover:text-red-700 transition-colors min-touch-target flex-1 justify-center sm:justify-end"
              title="Delete document"
            >
              <Trash2 className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              <span>Delete</span>
            </button>
          ) : (
            <span className="text-xs text-gray-400 text-right flex-1">Read-only</span>
          )}
        </div>
      </div>
    </div>
  );
}