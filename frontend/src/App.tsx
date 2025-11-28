import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from './store/authStore';
import Layout from './components/Layout';
import LoginPage from './pages/LoginPage';
import Dashboard from './pages/Dashboard';
import CaseManagement from './pages/CaseManagement';
import ReportGeneration from './pages/ReportGeneration';
import ReportViewer from './pages/ReportViewer';
import ComplianceDashboard from './pages/ComplianceDashboard';
import ExportManagement from './pages/ExportManagement';
import Settings from './pages/Settings';
import Help from './pages/Help';

/**
 * Main application component
 * Handles routing and authentication
 */
const App: React.FC = () => {
  const { isAuthenticated, isLoading } = useAuthStore();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-900">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <Router>
      <Routes>
        {/* Public Routes */}
        <Route path="/login" element={<LoginPage />} />

        {/* Protected Routes */}
        {isAuthenticated ? (
          <Route element={<Layout />}>
            <Route path="/" element={<Dashboard />} />
            <Route path="/cases" element={<CaseManagement />} />
            <Route path="/reports/generate" element={<ReportGeneration />} />
            <Route path="/reports/:reportId" element={<ReportViewer />} />
            <Route path="/compliance" element={<ComplianceDashboard />} />
            <Route path="/exports" element={<ExportManagement />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/help" element={<Help />} />
          </Route>
        ) : (
          <Route path="*" element={<Navigate to="/login" replace />} />
        )}
      </Routes>
    </Router>
  );
};

export default App;
