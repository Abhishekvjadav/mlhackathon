import React from "react";

export default function RemedyCard({ dosha, remedy }) {
  const r = remedy || null;

  return (
    <div>
      <div className="muted" style={{ marginBottom: 10 }}>
        {dosha ? `Recommended for: ${String(dosha).toUpperCase()}` : "Remedy details"}
      </div>

      {!r ? (
        <div className="muted">
          Run a prediction to see the full Ayurvedic guidance (herbs, formulation, diet, yoga, prevention).
        </div>
      ) : (
        <div style={{ display: "grid", gap: 10 }}>
          {r["Ayurvedic Herbs"] && (
            <div>
              <div className="muted" style={{ fontSize: 13 }}>
                Ayurvedic Herbs
              </div>
              <div style={{ fontWeight: 850 }}>{r["Ayurvedic Herbs"]}</div>
            </div>
          )}

          {r.Formulation && (
            <div>
              <div className="muted" style={{ fontSize: 13 }}>
                Formulation
              </div>
              <div style={{ fontWeight: 850 }}>{r.Formulation}</div>
            </div>
          )}

          {r["Diet and Lifestyle Recommendations"] && (
            <div>
              <div className="muted" style={{ fontSize: 13 }}>
                Diet and Lifestyle Recommendations
              </div>
              <div style={{ fontWeight: 850 }}>{r["Diet and Lifestyle Recommendations"]}</div>
            </div>
          )}

          {r["Yoga & Physical Therapy"] && (
            <div>
              <div className="muted" style={{ fontSize: 13 }}>
                Yoga & Physical Therapy
              </div>
              <div style={{ fontWeight: 850 }}>{r["Yoga & Physical Therapy"]}</div>
            </div>
          )}

          {r.Prevention && (
            <div>
              <div className="muted" style={{ fontSize: 13 }}>
                Prevention
              </div>
              <div style={{ fontWeight: 850 }}>{r.Prevention}</div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

