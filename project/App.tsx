import React, { Suspense } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

import Layout from './src/components/Layout';
import Auth from './src/pages/Auth';
import Dashboard from './src/pages/Dashboard';
import Predict from './src/pages/Predict';
import Transactions from './src/pages/Transactions';
import NotFound from './src/pages/NotFound';

const App: React.FC = () => {
  return (
    <Router>
      <Suspense fallback={<div className="min-h-screen flex items-center justify-center">Loading...</div>}>
        <Routes>
          {/* Public Route */}
          <Route path="/auth" element={<Auth />} />
          
          {/* Protected Routes Wrapper */}
          <Route element={<Layout />}>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/transactions" element={<Transactions />} />
            <Route path="*" element={<NotFound />} />
          </Route>
        </Routes>
      </Suspense>
      <ToastContainer
        position="top-right"
        autoClose={4000}
        theme="light"
        toastClassName="rounded-xl border border-border shadow-lg"
      />
    </Router>
  );
}

export default App;