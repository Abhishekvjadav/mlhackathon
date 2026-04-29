import React from "react";
import { Navigate, Route, Routes, Link } from "react-router-dom";
import Dashboard from "./pages/Dashboard.jsx";
import Predict from "./pages/Predict.jsx";

export default function App() {
  return (
    <div className="appShell">
      <header className="appHeader">
        <div className="brand">
          <div className="brandTitle">DoshaNet</div>
          <div className="brandSubtitle">Ayurvedic dosha prediction</div>
        </div>
        <nav className="nav">
          <Link className="navLink" to="/">
            Dashboard
          </Link>
          <Link className="navLink" to="/predict">
            Predict
          </Link>
        </nav>
      </header>

      <main className="appMain">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/predict" element={<Predict />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </div>
  );
}

