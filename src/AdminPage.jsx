import { useState, useEffect, useCallback, useMemo } from 'react';
import { fetchAllFaces, updateFace, deleteFace } from './lib/faceRecognition';
import './AdminPage.css';

function AdminPage() {
  const [users, setUsers] = useState([]);
  const [usersLoading, setUsersLoading] = useState(true);
  const [usersError, setUsersError] = useState('');
  const [editingId, setEditingId] = useState(null);
  const [editName, setEditName] = useState('');
  const [editTitle, setEditTitle] = useState('');
  const [rowBusyId, setRowBusyId] = useState(null);
  const [search, setSearch] = useState('');

  const loadUsers = useCallback(async () => {
    setUsersLoading(true);
    setUsersError('');
    try {
      const data = await fetchAllFaces();
      setUsers(data);
    } catch (err) {
      setUsersError(err.message || 'Failed to load registered users.');
    } finally {
      setUsersLoading(false);
    }
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const startEdit = (user) => {
    setEditingId(user.id);
    setEditName(user.name);
    setEditTitle(user.title || '');
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditName('');
    setEditTitle('');
  };

  const saveEdit = async (id) => {
    if (!editName.trim()) return;
    setRowBusyId(id);
    try {
      await updateFace(id, { name: editName.trim(), title: editTitle.trim() || null });
      setUsers((prev) =>
        prev.map((u) => (u.id === id ? { ...u, name: editName.trim(), title: editTitle.trim() || null } : u))
      );
      cancelEdit();
    } catch (err) {
      setUsersError(err.message || 'Failed to update user.');
    } finally {
      setRowBusyId(null);
    }
  };

  const removeUser = async (id, name) => {
    if (!window.confirm(`Delete "${name}"? This removes their face data too.`)) return;
    setRowBusyId(id);
    try {
      await deleteFace(id);
      setUsers((prev) => prev.filter((u) => u.id !== id));
    } catch (err) {
      setUsersError(err.message || 'Failed to delete user.');
    } finally {
      setRowBusyId(null);
    }
  };

  // Client-side filter -- fine at the scale a personal-robot face table
  // will realistically reach. Matches on name or title, case-insensitive.
  const filteredUsers = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return users;
    return users.filter(
      (u) => u.name.toLowerCase().includes(q) || (u.title || '').toLowerCase().includes(q)
    );
  }, [users, search]);

  return (
    <div className="admin-container">
      <h1>Registered Users</h1>

      <div className="admin-toolbar">
        <p className="admin-subtitle admin-subtitle--inline">
          {usersLoading
            ? 'Loading...'
            : `${filteredUsers.length} of ${users.length} ${users.length === 1 ? 'person' : 'people'}`}
        </p>
        <div className="admin-toolbar-actions">
          <input
            type="text"
            placeholder="Search name or title..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="admin-input admin-input--search"
            disabled={usersLoading}
          />
          <button
            className="admin-refresh-btn"
            onClick={loadUsers}
            disabled={usersLoading}
            title="Refresh"
          >
            {usersLoading ? '...' : '⟳'}
          </button>
        </div>
      </div>

      {usersError && <p className="admin-message error">{usersError}</p>}

      {usersLoading && (
        <div className="admin-skeleton">
          {[...Array(4)].map((_, i) => (
            <div className="admin-skeleton-row" key={i} />
          ))}
        </div>
      )}

      {!usersLoading && users.length > 0 && (
        <div className="admin-table-wrap">
          <table className="admin-table">
            <thead>
              <tr>
                <th>Name</th>
                <th>Title</th>
                <th className="admin-table-actions-head"></th>
              </tr>
            </thead>
            <tbody>
              {filteredUsers.map((user) => {
                const isEditing = editingId === user.id;
                const isBusyRow = rowBusyId === user.id;
                return (
                  <tr key={user.id} className={isBusyRow ? 'admin-row-busy' : ''}>
                    {isEditing ? (
                      <>
                        <td data-label="Name">
                          <input
                            type="text"
                            value={editName}
                            onChange={(e) => setEditName(e.target.value)}
                            className="admin-input admin-input--row"
                            disabled={isBusyRow}
                            autoFocus
                          />
                        </td>
                        <td data-label="Title">
                          <input
                            type="text"
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            className="admin-input admin-input--row"
                            placeholder="Title"
                            disabled={isBusyRow}
                          />
                        </td>
                        <td className="admin-table-actions">
                          <button
                            className="admin-link-btn"
                            onClick={() => saveEdit(user.id)}
                            disabled={isBusyRow}
                          >
                            Save
                          </button>
                          <button
                            className="admin-link-btn"
                            onClick={cancelEdit}
                            disabled={isBusyRow}
                          >
                            Cancel
                          </button>
                        </td>
                      </>
                    ) : (
                      <>
                        <td data-label="Name">{user.name}</td>
                        <td data-label="Title">{user.title || '\u2014'}</td>
                        <td className="admin-table-actions">
                          <button
                            className="admin-link-btn"
                            onClick={() => startEdit(user)}
                            disabled={isBusyRow}
                          >
                            Edit
                          </button>
                          <button
                            className="admin-link-btn admin-link-btn--danger"
                            onClick={() => removeUser(user.id, user.name)}
                            disabled={isBusyRow}
                          >
                            {isBusyRow ? '...' : 'Delete'}
                          </button>
                        </td>
                      </>
                    )}
                  </tr>
                );
              })}

              {filteredUsers.length === 0 && (
                <tr>
                  <td colSpan={3} className="admin-table-empty">
                    No matches for "{search}".
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {!usersLoading && users.length === 0 && !usersError && (
        <p className="admin-subtitle">No one enrolled yet.</p>
      )}

      <a className="admin-add-link" href="/admin/register">
        + Enroll a new face
      </a>

      <a className="admin-back-link" href="/">
        &larr; Back to robot
      </a>
    </div>
  );
}

export default AdminPage;