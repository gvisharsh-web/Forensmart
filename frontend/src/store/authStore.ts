import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import axios from 'axios';

/**
 * Authentication state and actions
 */
interface AuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: {
    id: string;
    email: string;
    firstName: string;
    lastName: string;
    role: string;
  } | null;
  token: string | null;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  register: (email: string, password: string, firstName: string, lastName: string) => Promise<void>;
  checkAuth: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      isAuthenticated: false,
      isLoading: true,
      user: null,
      token: null,

      login: async (email: string, password: string) => {
        try {
          const response = await axios.post('/api/auth/login', { email, password });
          const { token, user } = response.data;

          localStorage.setItem('authToken', token);
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

          set({
            isAuthenticated: true,
            token,
            user,
            isLoading: false,
          });
        } catch (error) {
          console.error('Login failed:', error);
          throw error;
        }
      },

      logout: () => {
        localStorage.removeItem('authToken');
        delete axios.defaults.headers.common['Authorization'];
        set({
          isAuthenticated: false,
          token: null,
          user: null,
          isLoading: false,
        });
      },

      register: async (email: string, password: string, firstName: string, lastName: string) => {
        try {
          const response = await axios.post('/api/auth/register', {
            email,
            password,
            firstName,
            lastName,
          });
          const { token, user } = response.data;

          localStorage.setItem('authToken', token);
          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;

          set({
            isAuthenticated: true,
            token,
            user,
            isLoading: false,
          });
        } catch (error) {
          console.error('Registration failed:', error);
          throw error;
        }
      },

      checkAuth: async () => {
        try {
          const token = localStorage.getItem('authToken');
          if (!token) {
            set({ isLoading: false });
            return;
          }

          axios.defaults.headers.common['Authorization'] = `Bearer ${token}`;
          const response = await axios.get('/api/auth/me');
          const { user } = response.data;

          set({
            isAuthenticated: true,
            token,
            user,
            isLoading: false,
          });
        } catch (error) {
          console.error('Auth check failed:', error);
          localStorage.removeItem('authToken');
          set({ isLoading: false });
        }
      },
    }),
    {
      name: 'auth-storage',
    }
  )
);
