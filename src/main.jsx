import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import AdminPage from './AdminPage.jsx';

// Simple path-based routing -- visit /admin to enroll faces.
// (No react-router needed for just two screens.)
const isAdminRoute = window.location.pathname.startsWith('/admin');

createRoot(document.getElementById('root')).render(
  <StrictMode>
    {isAdminRoute ? <AdminPage /> : <App />}
  </StrictMode>,
);