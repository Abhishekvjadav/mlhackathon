import React from "react";

export default function FeatureSelect({ name, options, value, onChange }) {
  return (
    <div>
      <label htmlFor={`f_${name}`}>{name}</label>
      <select
        id={`f_${name}`}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {(options || []).map((opt) => (
          <option key={opt} value={opt}>
            {String(opt)}
          </option>
        ))}
      </select>
    </div>
  );
}

