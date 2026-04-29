import React from "react";

function pct(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return `${(Number(n) * 100).toFixed(1)}%`;
}

export default function ProbabilityBar({ dosha, prob }) {
  const width = Math.max(0, Math.min(100, (Number(prob) || 0) * 100));
  return (
    <div className="probRow">
      <div className="probName">{String(dosha).toUpperCase()}</div>
      <div className="barTrack" aria-label={`${dosha} probability`}>
        <div className="barFill" style={{ width: `${width}%` }} />
      </div>
      <div className="probPct">{pct(prob)}</div>
    </div>
  );
}

