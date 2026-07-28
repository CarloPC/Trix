import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import AdminPage from './AdminPage.jsx';
import Registration from './Registration.jsx';

// Simple path-based routing -- no react-router needed for three screens:
//   /                -> App (the robot face)
//   /admin           -> AdminPage (manage registered users: view/edit/delete)
//   /admin/register  -> Registration (enroll a new face)
const path = window.location.pathname;

function Root() {
  if (path.startsWith('/admin/register')) return <Registration />;
  if (path.startsWith('/admin')) return <AdminPage />;
  return <App />;
}

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <Root />
  </StrictMode>,
);