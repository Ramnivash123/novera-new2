/**
 * API Service for communicating with the backend
 */
import axios, { AxiosInstance } from 'axios';
import type { Customization, CustomizationUpdateRequest, ThemePreset } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';const API_VERSION = '/api/v1';

console.log('🔧 API Configuration:', {
  baseURL: `${API_BASE_URL}${API_VERSION}`,
  env: import.meta.env.MODE
});

// Types
export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: string;
  metadata?: {
    sources?: Source[];
    confidence?: string;
    citations?: any[];
    error?: boolean;
    reformulated_query?: string;
  };
}

export interface ChatRequest {
  query: string;
  conversation_id?: string | null;
  doc_type?: string;
  department?: string;
  stream?: boolean;
}

export interface Source {
  document: string;
  page: number | null;
  section: string | null;
  chunk_id: string;
}

export interface ChatResponse {
  answer: string;
  conversation_id: string;
  sources: Source[];
  citations: any[];
  confidence: string;
  status: string;
  suggestions?: string[];
  metadata: {
    chunks_used?: number;
    tokens?: any;
    searched_documents?: boolean;
    retrieval_metadata?: any;
    context_summary?: {
      primary_document?: string;
      active_documents?: string[];
      recent_time_period?: string;
      message_count?: number;
      last_intent?: string;
    };
    query_reformulated?: boolean;
  };
}

export interface Document {
  id: string;
  filename: string;
  doc_type: string;
  department: string | null;
  total_pages: number;
  total_chunks: number;
  status: string;
  upload_date: string;
  processed_date: string | null;
}

export interface Conversation {
  id: string;
  user_id: string;
  created_at: string;
  updated_at: string;
  messages: ChatMessage[];
  metadata: any;
}

export interface AdminUser {
  id: string;
  email: string;
  username: string;
  full_name: string | null;
  role: string;
  is_active: boolean;
  is_verified: boolean;
  created_at: string;
  last_login: string | null;
  document_count: number;
}

export interface UserStats {
  total_users: number;
  active_users: number;
  admin_users: number;
  regular_users: number;
  verified_users: number;
}

export interface SystemStats {
  total_users: number;
  total_documents: number;
  total_chunks: number;
  active_sessions: number;
  storage_used_mb: number;
}

export interface TokenStats {
  conversation_id: string;
  summary: {
    total_messages: number;
    total_tokens: number;
    cached_tokens: number;
    prompt_tokens: number;
    completion_tokens: number;
    cache_efficiency_percent: number;
    total_cost_usd: number;
    avg_tokens_per_message: number;
    avg_cost_per_message: number;
  };
  messages: MessageTokenStat[];
  pricing_info: {
    model: string;
    cached_token_price: number;
    regular_token_price: number;
    currency: string;
  };
}

export interface MessageTokenStat {
  message_id: string;
  timestamp: string;
  tokens: {
    total: number;
    cached: number;
    prompt: number;
    completion: number;
  };
  cost_usd: number;
}

export interface ChunkData {
  id: string;
  document_id: string;
  chunk_index: number;
  content: string;
  chunk_type: string;
  page_numbers: number[];
  section_title: string | null;
  token_count: number;
  title: string | null;
  is_edited: boolean;
  edited_at: string | null;
  edited_by: string | null;
  edit_count: number;
  metadata: any;
}

export interface DocumentInfo {
  id: string;
  filename: string;
  original_filename: string;
  file_size_mb: number;
  doc_type: string;
  department: string | null;
  total_pages: number;
  total_chunks: number;
  has_tables: boolean;
  has_images: boolean;
  status: string;
  upload_date: string;
  processed_date: string | null;
  mime_type: string;
  preview_type: string;
  is_previewable: boolean;
}

export interface EditHistoryItem {
  id: string;
  edited_at: string;
  edited_by: string;
  old_content: string;
  new_content: string;
  change_summary: string | null;
  metadata: any;
}

export interface DocumentEditStats {
  total_chunks: number;
  edited_chunks: number;
  unedited_chunks: number;
  total_edits: number;
  edit_percentage: number;
}

class ApiService {
  private api: AxiosInstance;

  constructor() {
    this.api = axios.create({
      baseURL: `${API_BASE_URL}${API_VERSION}`,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 120000,
      withCredentials: true,
    });

    // Request interceptor
    this.api.interceptors.request.use(
      (config) => {
        const token = localStorage.getItem('access_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        
        console.log('📤 API Request:', {
          method: config.method?.toUpperCase(),
          url: config.url,
          baseURL: config.baseURL,
          fullURL: `${config.baseURL}${config.url}`,
          headers: {
            ...config.headers,
            Authorization: token ? 'Bearer ***' : 'None'
          }
        });
        
        return config;
      },
      (error) => {
        console.error('❌ Request Error:', error);
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
    (response) => {
      console.log('✅ API Response:', {
        status: response.status,
        url: response.config.url,
      });
      return response;
    },
    async (error) => {
      const originalRequest = error.config;

      console.error('❌ API Error:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        url: originalRequest?.url,
        method: originalRequest?.method,
        hasToken: !!localStorage.getItem('access_token'),
      });

      // Handle authentication errors
      if (error.response?.status === 401 && !originalRequest._retry) {
        originalRequest._retry = true;

        const refreshToken = localStorage.getItem('refresh_token');
        
        if (!refreshToken) {
          console.error('No refresh token available, redirecting to login');
          localStorage.clear();
          window.location.href = '/login';
          return Promise.reject(error);
        }

        try {
          console.log('🔄 Attempting to refresh token...');
          
          const response = await axios.post(
            `${API_BASE_URL}${API_VERSION}/auth/refresh`,
            { refresh_token: refreshToken },
            {
              headers: {
                'Content-Type': 'application/json',
              }
            }
          );

          const { access_token } = response.data;
          
          console.log('✅ Token refreshed successfully');
          localStorage.setItem('access_token', access_token);

          // Retry original request with new token
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
          return this.api(originalRequest);
          
        } catch (refreshError: any) {
          console.error('❌ Token refresh failed:', refreshError);
          
          // Clear tokens and redirect to login
          localStorage.removeItem('access_token');
          localStorage.removeItem('refresh_token');
          
          // Don't redirect if already on login page
          if (!window.location.pathname.includes('/login')) {
            window.location.href = '/login';
          }
          
          return Promise.reject(refreshError);
        }
      }

      // Handle validation errors
      if (error.response?.status === 422) {
        console.error('Validation Error (422):', {
          url: originalRequest.url,
          method: originalRequest.method,
          data: originalRequest.data,
          errors: error.response.data.detail,
        });
      }

      // Handle network errors
      if (error.message === 'Network Error') {
        console.error('🚫 Network Error - Possible CORS issue or server not running');
      }

      return Promise.reject(error);
    }
  );
  }

  async healthCheck() {
    try {
      const response = await this.api.get('/health');
      console.log('💚 Health check passed:', response.data);
      return response.data;
    } catch (error) {
      console.error('💔 Health check failed:', error);
      throw error;
    }
  }

  async sendChatMessage(request: ChatRequest): Promise<ChatResponse> {
    const response = await this.api.post('/chat', request);
    return response.data;
  }

  async getConversations(limit: number = 10): Promise<Conversation[]> {
    const response = await this.api.get('/chat/conversations', {
      params: { limit },
    });
    return response.data;
  }

  async getConversation(conversationId: string): Promise<Conversation> {
    const response = await this.api.get(`/chat/conversations/${conversationId}`);
    return response.data;
  }

  async deleteConversation(conversationId: string): Promise<void> {
    await this.api.delete(`/chat/conversations/${conversationId}`);
  }

  async getConversationAnalytics(conversationId: string): Promise<any> {
    const response = await this.api.get(`/chat/conversations/${conversationId}/analytics`);
    return response.data;
  }

  async getConversationTokenStats(conversationId: string): Promise<TokenStats> {
    const response = await this.api.get(`/chat/conversations/${conversationId}/token-stats`);
    return response.data;
  }

  async uploadDocument(
    file: File,
    docType: string,
    department?: string
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const params: any = { doc_type: docType };
    if (department) params.department = department;

    const response = await this.api.post('/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      params,
    });
    return response.data;
  }

  async uploadDocuments(
    files: File[],
    docType: string,
    department?: string
  ): Promise<any> {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    const params: any = { doc_type: docType };
    if (department) params.department = department;

    const response = await this.api.post('/documents/upload-multiple', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      params,
    });
    return response.data;
  }

  async getDocuments(params?: {
    skip?: number;
    limit?: number;
    doc_type?: string;
    status?: string;
    department?: string;
  }): Promise<{ total: number; documents: Document[] }> {
    const response = await this.api.get('/documents', { params });
    return response.data;
  }

  async getDocumentStatus(documentId: string): Promise<any> {
    const response = await this.api.get(`/documents/${documentId}/status`);
    return response.data;
  }

  async deleteDocument(documentId: string): Promise<void> {
    await this.api.delete(`/documents/${documentId}`);
  }

  async search(query: string, topK: number = 5, docType?: string) {
    const response = await this.api.post('/search', null, {
      params: { query, top_k: topK, doc_type: docType },
    });
    return response.data;
  }

  async register(
    email: string,
    username: string,
    password: string,
    fullName?: string
  ): Promise<any> {
    try {
      console.log('Registration attempt:', { email, username });
      
      const response = await this.api.post('/auth/register', {
        email,
        username,
        password,
        full_name: fullName,
      });
      
      console.log('Registration successful');
      return response.data;
      
    } catch (error: any) {
      if (error.response?.status === 422) {
        console.error('Registration validation failed');
        this.handleValidationError(error);
      }
      throw error;
    }
  }

  async login(email: string, password: string): Promise<any> {
    try {
      console.log('Login attempt:', { 
        email, 
        passwordLength: password.length,
        requestPayload: { email, password: '***' }
      });
      
      const response = await this.api.post('/auth/login', {
        email: email,
        password: password,
      });
      
      console.log('Login successful:', {
        hasAccessToken: !!response.data.access_token,
        hasRefreshToken: !!response.data.refresh_token,
        user: response.data.user?.email,
      });
      
      return response.data;
      
    } catch (error: any) {
      console.error('Login failed:', {
        status: error.response?.status,
        statusText: error.response?.statusText,
        data: error.response?.data,
      });
      
      if (error.response?.status === 422) {
        this.handleValidationError(error);
      }
      
      throw error;
    }
  }

  async logout(refreshToken: string): Promise<void> {
    await this.api.post('/auth/logout', {
      refresh_token: refreshToken,
    });
  }

  async getCurrentUser(): Promise<any> {
    const response = await this.api.get('/auth/me');
    return response.data;
  }

  async updateProfile(data: {
    full_name?: string;
    avatar_url?: string;
    preferences?: Record<string, any>;
    metadata?: Record<string, any>;
  }): Promise<any> {
    const response = await this.api.put('/auth/profile', data);
    return response.data;
  }

  async changePassword(
    currentPassword: string,
    newPassword: string
  ): Promise<void> {
    await this.api.post('/auth/change-password', {
      current_password: currentPassword,
      new_password: newPassword,
    });
  }

  async forgotPassword(email: string): Promise<any> {
  try {
    console.log('Sending forgot password request for:', email);
    const response = await this.api.post('/auth/forgot-password', {
      email,
    });
    console.log('Forgot password response:', response.data);
    return response.data;
  } catch (error: any) {
    console.error('Forgot password error:', error.response?.data);
    throw error;
  }
}

  async verifyResetToken(token: string): Promise<any> {
    try {
      const response = await this.api.post('/auth/verify-reset-token', {
        token,
      });
      return response.data;
    } catch (error: any) {
      console.error('Verify reset token error:', error.response?.data);
      throw error;
    }
  }

  async resetPassword(token: string, newPassword: string): Promise<any> {
    try {
      const response = await this.api.post('/auth/reset-password', {
        token,
        new_password: newPassword,
      });
      return response.data;
    } catch (error: any) {
      console.error('Reset password error:', error.response?.data);
      throw error;
    }
  }

  async verifyEmail(token: string): Promise<any> {
    try {
      const response = await this.api.post('/auth/verify-email', {
        token,
      });
      return response.data;
    } catch (error: any) {
      console.error('Email verification error:', error.response?.data);
      throw error;
    }
  }

  async resendVerificationEmail(): Promise<any> {
    try {
      const response = await this.api.post('/auth/resend-verification');
      return response.data;
    } catch (error: any) {
      console.error('Resend verification error:', error.response?.data);
      throw error;
    }
  }

  async getUsers(params?: {
    skip?: number;
    limit?: number;
    role?: string;
    is_active?: boolean;
    search?: string;
  }): Promise<{ total: number; users: AdminUser[] }> {
    const response = await this.api.get('/admin/users', { params });
    return response.data;
  }

  async getUserStats(): Promise<UserStats> {
    const response = await this.api.get('/admin/users/stats');
    return response.data;
  }

  async getSystemStats(): Promise<SystemStats> {
    const response = await this.api.get('/admin/stats');
    return response.data;
  }

  async createUser(data: {
    email: string;
    username: string;
    password: string;
    full_name?: string;
    role: string;
    is_active: boolean;
  }): Promise<any> {
    const response = await this.api.post('/admin/users', data);
    return response.data;
  }

  async getUserDetails(userId: string): Promise<any> {
    const response = await this.api.get(`/admin/users/${userId}`);
    return response.data;
  }

  async updateUser(
    userId: string,
    data: {
      full_name?: string;
      role?: string;
      is_active?: boolean;
      is_verified?: boolean;
    }
  ): Promise<any> {
    const response = await this.api.put(`/admin/users/${userId}`, data);
    return response.data;
  }

  async deleteUser(userId: string): Promise<void> {
    await this.api.delete(`/admin/users/${userId}`);
  }

  async resetUserPassword(userId: string, newPassword: string): Promise<void> {
    await this.api.post(`/admin/users/${userId}/reset-password`, null, {
      params: { new_password: newPassword },
    });
  }

  async getAllDocuments(params?: {
    skip?: number;
    limit?: number;
    doc_type?: string;
    status?: string;
    user_id?: string;
  }): Promise<{ total: number; documents: any[] }> {
    const response = await this.api.get('/admin/documents', { params });
    return response.data;
  }

  async getDocumentInfo(documentId: string): Promise<DocumentInfo> {
    const response = await this.api.get(`/documents/${documentId}/info`);
    return response.data;
  }

  async getDocumentChunks(documentId: string): Promise<ChunkData[]> {
    const response = await this.api.get(`/documents/${documentId}/chunks`);
    return response.data;
  }

  async getTextPreview(documentId: string, maxChars: number = 5000): Promise<any> {
    const response = await this.api.get(`/documents/${documentId}/preview/text`, {
      params: { max_chars: maxChars }
    });
    return response.data;
  }

  async downloadDocument(documentId: string): Promise<Blob> {
    const response = await this.api.get(`/documents/${documentId}/download`, {
      responseType: 'blob'
    });
    return response.data;
  }

  async editChunk(chunkId: string, newContent: string, metadata?: any): Promise<ChunkData> {
    const response = await this.api.put(`/chunks/${chunkId}/edit`, {
      chunk_id: chunkId,
      new_content: newContent,
      metadata: metadata
    });
    return response.data;
  }

  async batchEditChunks(edits: Array<{ chunk_id: string; new_content: string }>): Promise<any> {
    const response = await this.api.post('/chunks/batch-edit', { edits });
    return response.data;
  }

  async revertChunk(chunkId: string): Promise<void> {
    await this.api.post(`/chunks/${chunkId}/revert`);
  }

  async deleteChunk(chunkId: string): Promise<void> {
    await this.api.delete(`/chunks/${chunkId}`);
  }

  async getChunkHistory(chunkId: string, limit: number = 10): Promise<EditHistoryItem[]> {
    const response = await this.api.get(`/chunks/${chunkId}/history`, {
      params: { limit }
    });
    return response.data;
  }

  async getDocumentEditStats(documentId: string): Promise<DocumentEditStats> {
    const response = await this.api.get(`/documents/${documentId}/edit-stats`);
    return response.data;
  }

  async exportConversation(
    conversationId: string,
    format: 'markdown' | 'json' = 'markdown'
  ): Promise<Blob | any> {
    const response = await this.api.get(
      `/chat/conversations/${conversationId}/export`,
      {
        params: { format },
        responseType: format === 'markdown' ? 'blob' : 'json'
      }
    );
    return response.data;
  }

  private handleValidationError(error: any): void {
    const validationErrors = error.response?.data?.detail;
    
    if (Array.isArray(validationErrors)) {
      console.error('Validation errors:');
      validationErrors.forEach((err: any, index: number) => {
        console.error(`  ${index + 1}. Field: ${err.loc?.join('.')}`);
        console.error(`     Error: ${err.msg}`);
        console.error(`     Input: ${err.input}`);
      });
      
      const errorMessage = validationErrors
        .map((err: any) => `${err.loc?.join('.')}: ${err.msg}`)
        .join('; ');
      
      throw new Error(errorMessage);
    }
  }

  // ============================================================================
  // CUSTOMIZATION API
  // ============================================================================

  async getCurrentCustomization(): Promise<Customization> {
    const response = await this.api.get('/customization/current');
    return response.data;
  }

  async getAdminCustomization(organizationName: string = 'default'): Promise<Customization> {
    const response = await this.api.get('/admin/customization', {
      params: { organization_name: organizationName }
    });
    return response.data;
  }

  async updateCustomization(
    data: CustomizationUpdateRequest,
    organizationName: string = 'default'
  ): Promise<Customization> {
    const response = await this.api.put('/admin/customization', data, {
      params: { organization_name: organizationName }
    });
    return response.data;
  }

  async uploadLogo(
    file: File,
    logoType: 'light' | 'dark' | 'favicon' = 'light',
    organizationName: string = 'default'
  ): Promise<any> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await this.api.post('/admin/customization/logo', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      params: {
        organization_name: organizationName,
        logo_type: logoType
      }
    });
    return response.data;
  }

  async deleteLogo(
    logoType: 'light' | 'dark' | 'favicon' = 'light',
    organizationName: string = 'default'
  ): Promise<void> {
    await this.api.delete('/admin/customization/logo', {
      params: {
        logo_type: logoType,
        organization_name: organizationName
      }
    });
  }

  async resetCustomization(organizationName: string = 'default'): Promise<any> {
    const response = await this.api.post('/admin/customization/reset', null, {
      params: { organization_name: organizationName }
    });
    return response.data;
  }

  async generateChunkTitle(chunkId: string): Promise<any> {
    const response = await this.api.post(`/chunks/${chunkId}/generate-title`);
    return response.data;
  }

  async generateAllTitles(documentId: string): Promise<any> {
    const response = await this.api.post(`/documents/${documentId}/generate-all-titles`);
    return response.data;
  }

  async updateChunkTitle(chunkId: string, title: string): Promise<any> {
    const response = await this.api.put(`/chunks/${chunkId}/title`, null, {
      params: { title }
    });
    return response.data;
  }

  async getThemePresets(): Promise<ThemePreset[]> {
    const response = await this.api.get('/admin/customization/presets');
    return response.data;
  }

  async applyThemePreset(
    presetName: string,
    organizationName: string = 'default'
  ): Promise<any> {
    const response = await this.api.post('/admin/customization/apply-preset', null, {
      params: {
        preset_name: presetName,
        organization_name: organizationName
      }
    });
    return response.data;
  }

  async exportCustomization(organizationName: string = 'default'): Promise<Customization> {
    const response = await this.api.get('/admin/customization/export', {
      params: { organization_name: organizationName }
    });
    return response.data;
  }

  async validateColorContrast(
    backgroundColor: string,
    textColor: string
  ): Promise<{ ratio: number; passes_aa: boolean; passes_aaa: boolean }> {
    // Client-side WCAG contrast calculation
    const getLuminance = (color: string): number => {
      const hex = color.replace('#', '');
      const r = parseInt(hex.substr(0, 2), 16) / 255;
      const g = parseInt(hex.substr(2, 2), 16) / 255;
      const b = parseInt(hex.substr(4, 2), 16) / 255;
      
      const [rs, gs, bs] = [r, g, b].map(c => 
        c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4)
      );
      
      return 0.2126 * rs + 0.7152 * gs + 0.0722 * bs;
    };

    const l1 = getLuminance(backgroundColor);
    const l2 = getLuminance(textColor);
    const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    
    return {
      ratio: Math.round(ratio * 100) / 100,
      passes_aa: ratio >= 4.5,
      passes_aaa: ratio >= 7
    };
  }
}

export const api = new ApiService();
export default api;
