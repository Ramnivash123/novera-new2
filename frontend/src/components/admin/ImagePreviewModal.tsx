import { useState } from 'react';
import { X, Upload, AlertCircle, Check } from 'lucide-react';

interface ImagePreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  file: File | null;
  previewUrl: string | null;
  logoType: 'light' | 'dark' | 'favicon';
}

export default function ImagePreviewModal({
  isOpen,
  onClose,
  onConfirm,
  file,
  previewUrl,
  logoType,
}: ImagePreviewModalProps) {
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);

  if (!isOpen || !file || !previewUrl) return null;

  const getRecommendations = () => {
    switch (logoType) {
      case 'light':
      case 'dark':
        return {
          title: 'Logo Recommendations',
          dimensions: '200x50px to 400x100px',
          aspectRatio: '4:1 or 2:1',
          format: 'PNG or SVG (transparent background)',
          maxSize: '500KB',
        };
      case 'favicon':
        return {
          title: 'Favicon Recommendations',
          dimensions: '32x32px or 64x64px',
          aspectRatio: '1:1 (square)',
          format: 'PNG or ICO',
          maxSize: '100KB',
        };
      default:
        return null;
    }
  };

  const recommendations = getRecommendations();
  const fileSizeKB = (file.size / 1024).toFixed(2);
  const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
  const displaySize = file.size > 1024 * 1024 ? `${fileSizeMB} MB` : `${fileSizeKB} KB`;

  const handleImageLoad = (e: React.SyntheticEvent<HTMLImageElement>) => {
    const img = e.currentTarget;
    setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
  };

  const checkDimensions = () => {
    if (!imageDimensions) return null;

    if (logoType === 'favicon') {
      const isSquare = imageDimensions.width === imageDimensions.height;
      const isRecommendedSize = 
        (imageDimensions.width === 32 && imageDimensions.height === 32) ||
        (imageDimensions.width === 64 && imageDimensions.height === 64);
      
      return {
        isGood: isSquare && isRecommendedSize,
        message: isSquare && isRecommendedSize
          ? 'Perfect dimensions!'
          : !isSquare
          ? 'Favicon should be square (1:1 ratio)'
          : 'Recommended: 32x32 or 64x64 pixels',
      };
    } else {
      const aspectRatio = imageDimensions.width / imageDimensions.height;
      const isGoodAspectRatio = aspectRatio >= 2 && aspectRatio <= 4;
      const isGoodSize = 
        imageDimensions.width >= 200 && 
        imageDimensions.width <= 600 &&
        imageDimensions.height >= 40 &&
        imageDimensions.height <= 150;

      return {
        isGood: isGoodAspectRatio && isGoodSize,
        message: isGoodAspectRatio && isGoodSize
          ? 'Good dimensions!'
          : !isGoodAspectRatio
          ? 'Logo should have 2:1 to 4:1 aspect ratio'
          : 'Recommended: 200-600px width, 40-150px height',
      };
    }
  };

  const dimensionCheck = checkDimensions();

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-2 sm:p-4">
      <div className="bg-white rounded-lg sm:rounded-xl max-w-2xl w-full max-h-[95vh] overflow-y-auto scroll-smooth-touch">
        {/* Header */}
        <div className="flex items-center justify-between p-4 sm:p-6 border-b border-gray-200">
          <h2 className="text-lg sm:text-xl font-semibold text-gray-900">
            Preview {logoType === 'favicon' ? 'Favicon' : 'Logo'}
          </h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors min-touch-target"
            aria-label="Close"
          >
            <X className="w-5 h-5 text-gray-500" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 sm:p-6 space-y-4 sm:space-y-6">
          {/* Image Preview */}
          <div className="border-2 border-dashed border-gray-300 rounded-lg p-4 sm:p-8 bg-gray-50">
            <div className="flex items-center justify-center min-h-[150px] sm:min-h-[200px]">
              <img
                src={previewUrl}
                alt="Preview"
                onLoad={handleImageLoad}
                className="max-w-full max-h-[200px] sm:max-h-[300px] object-contain"
              />
            </div>
          </div>

          {/* File Info */}
          <div className="grid grid-cols-2 gap-3 sm:gap-4">
            <div className="bg-gray-50 rounded-lg p-3 sm:p-4">
              <p className="text-xs sm:text-sm text-gray-600 mb-1">File Name</p>
              <p className="font-medium text-gray-900 truncate text-xs sm:text-base" title={file.name}>{file.name}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 sm:p-4">
              <p className="text-xs sm:text-sm text-gray-600 mb-1">File Size</p>
              <p className="font-medium text-gray-900 text-xs sm:text-base">{displaySize}</p>
            </div>
            <div className="bg-gray-50 rounded-lg p-3 sm:p-4">
              <p className="text-xs sm:text-sm text-gray-600 mb-1">File Type</p>
              <p className="font-medium text-gray-900 text-xs sm:text-base uppercase">{file.type.split('/')[1]}</p>
            </div>
            {imageDimensions && (
              <div className="bg-gray-50 rounded-lg p-3 sm:p-4">
                <p className="text-xs sm:text-sm text-gray-600 mb-1">Dimensions</p>
                <p className="font-medium text-gray-900 text-xs sm:text-base">
                  {imageDimensions.width} x {imageDimensions.height}px
                </p>
              </div>
            )}
          </div>

          {/* Dimension Check */}
          {dimensionCheck && (
            <div
              className={`flex items-start gap-2 sm:gap-3 p-3 sm:p-4 rounded-lg ${
                dimensionCheck.isGood
                  ? 'bg-green-50 border border-green-200'
                  : 'bg-yellow-50 border border-yellow-200'
              }`}
            >
              {dimensionCheck.isGood ? (
                <Check className="w-4 h-4 sm:w-5 sm:h-5 text-green-600 flex-shrink-0 mt-0.5" />
              ) : (
                <AlertCircle className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
              )}
              <div>
                <p
                  className={`font-medium text-xs sm:text-sm ${
                    dimensionCheck.isGood ? 'text-green-900' : 'text-yellow-900'
                  }`}
                >
                  {dimensionCheck.message}
                </p>
              </div>
            </div>
          )}

          {/* Recommendations */}
          {recommendations && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 sm:p-4">
              <h3 className="font-semibold text-blue-900 mb-2 sm:mb-3 text-sm sm:text-base">
                {recommendations.title}
              </h3>
              <div className="space-y-1.5 sm:space-y-2 text-xs sm:text-sm text-blue-800">
                <div className="flex items-start gap-2">
                  <span className="font-medium min-w-[100px] sm:min-w-[120px]">Dimensions:</span>
                  <span>{recommendations.dimensions}</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium min-w-[100px] sm:min-w-[120px]">Aspect Ratio:</span>
                  <span>{recommendations.aspectRatio}</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium min-w-[100px] sm:min-w-[120px]">Format:</span>
                  <span>{recommendations.format}</span>
                </div>
                <div className="flex items-start gap-2">
                  <span className="font-medium min-w-[100px] sm:min-w-[120px]">Max Size:</span>
                  <span>{recommendations.maxSize}</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex flex-col-reverse sm:flex-row items-center justify-end gap-2 sm:gap-3 p-4 sm:p-6 border-t border-gray-200">
          <button
            onClick={onClose}
            className="w-full sm:w-auto px-4 py-2.5 border border-gray-300 rounded-lg hover:bg-gray-50 transition-colors text-sm sm:text-base font-medium min-touch-target"
          >
            Cancel
          </button>
          <button
            onClick={onConfirm}
            className="w-full sm:w-auto flex items-center justify-center gap-2 px-6 py-2.5 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm sm:text-base font-medium min-touch-target"
          >
            <Upload className="w-4 h-4" />
            Upload Image
          </button>
        </div>
      </div>
    </div>
  );
}