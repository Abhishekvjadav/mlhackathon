import React, { useEffect, useMemo, useState } from "react";
import { api } from "../api/client.js";
import FeatureSelect from "../components/FeatureSelect.jsx";
import ProbabilityBar from "../components/ProbabilityBar.jsx";
import RemedyCard from "../components/RemedyCard.jsx";

export default function Predict() {
  const [loading, setLoading] = useState(true);
  const [schema, setSchema] = useState(null);
  const [error, setError] = useState(null);

  const [formValues, setFormValues] = useState({});
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        setLoading(true);
        const res = await api.get("/api/schema");
        if (!mounted) return;
        setSchema(res.data);
        const initial = {};
        for (const f of res.data.features) {
          initial[f.name] = f.options?.[0] ?? "";
        }
        setFormValues(initial);
        setError(null);
      } catch (e) {
        if (!mounted) return;
        setError(e?.response?.data?.detail || e.message || "Failed to load schema");
      } finally {
        if (mounted) setLoading(false);
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, []);

  const featureList = useMemo(() => schema?.features ?? [], [schema]);

  async function onSubmit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    setResult(null);
    try {
      const res = await api.post("/api/predict", {
        features: formValues
      });
      setResult(res.data);
    } catch (err) {
      setError(err?.response?.data?.detail || err.message || "Prediction failed");
    } finally {
      setSubmitting(false);
    }
  }

  if (loading) {
    return (
      <div className="panel">
        <div className="panelTitle">Loading prediction form...</div>
        <div className="muted">Fetching feature schema from backend</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="panel">
        <div className="panelTitle">Error</div>
        <div className="error">{String(error)}</div>
      </div>
    );
  }

  const doshaClasses = schema?.doshaClasses || [];
  const predictedDosha = result?.predictedDosha;

  return (
    <>
      <div className="panel">
        <h2 className="panelTitle">Live Prediction</h2>
        <div className="muted">
          Choose your prakriti-related symptom traits. The backend encodes inputs and runs the DoshaNet model.
        </div>
      </div>

      <div className="panel">
        <form onSubmit={onSubmit}>
          <div className="formGrid">
            {featureList.map((f) => (
              <FeatureSelect
                key={f.name}
                name={f.name}
                options={f.options}
                value={formValues[f.name]}
                onChange={(next) => setFormValues((prev) => ({ ...prev, [f.name]: next }))}
              />
            ))}
          </div>

          <div style={{ marginTop: 16, display: "flex", justifyContent: "flex-end" }}>
            <button type="submit" disabled={submitting}>
              {submitting ? "Predicting..." : "Predict Dosha"}
            </button>
          </div>
        </form>
      </div>

      {result && (
        <>
          <div className="grid2">
            <div className="panel">
              <h3 className="panelTitle">Prediction</h3>
              <div style={{ fontSize: 22, fontWeight: 950 }}>
                {predictedDosha ? String(predictedDosha).toUpperCase() : "—"}
              </div>
              <div className="muted" style={{ marginTop: 6 }}>
                Confidence: {result?.confidence != null ? `${Number(result.confidence).toFixed(1)}%` : "—"}
              </div>

              <div style={{ marginTop: 14 }}>
                {(result?.probabilities || []).map((p) => (
                  <ProbabilityBar key={p.dosha} dosha={p.dosha} prob={p.prob} />
                ))}
              </div>
            </div>

            <div className="panel">
              <h3 className="panelTitle">Ayurvedic Remedy</h3>
              <RemedyCard dosha={result?.predictedDosha} remedy={result?.remedy} />
            </div>
          </div>
        </>
      )}
    </>
  );
}

