import { useState } from "react";
import "./styles.css";

const FAQ_DATA = {
  Infiltration: [
    ["What is lung infiltration?", "Pulmonary infiltration refers to substances such as fluid, inflammatory cells, or infection filling the lung tissues, appearing as opacities on X-rays."],
    ["What causes infiltration in lungs?", "Common causes include pneumonia, tuberculosis, inflammation, pulmonary edema, or immune-related conditions."],
    ["What are symptoms of infiltration?", "Cough, fever, shortness of breath, chest discomfort, and fatigue."],
    ["Is infiltration dangerous?", "Mild cases are manageable, but severe or widespread infiltration can impair breathing and become life-threatening."],
    ["What is the risk level of infiltration?", "Low in mild infections; high in severe infections, elderly patients, or untreated conditions."],
    ["How is infiltration diagnosed?", "Chest X-ray, CT scan, blood tests, and sometimes sputum analysis."],
    ["What treatment is used for infiltration?", "Antibiotics (if infection), anti-inflammatory drugs, oxygen support."],
    ["Can infiltration lead to pneumonia?", "Yes, infiltration is often a sign of pneumonia or lung infection."],
    ["What complications occur in infiltration?", "Respiratory failure, spread of infection, lung damage."],
    ["When should I consult a doctor?", "If symptoms worsen, breathing becomes difficult, or fever persists."]
  ],
  Atelectasis: [
    ["What is atelectasis?", "Collapse of part or all of a lung, reducing its ability to exchange oxygen."],
    ["What causes lung collapse?", "Mucus blockage, tumors, surgery, shallow breathing, or external pressure."],
    ["What are symptoms of atelectasis?", "Shortness of breath, rapid breathing, low oxygen levels."],
    ["Is atelectasis life-threatening?", "Small areas are not, but large collapse can be serious."],
    ["What is the severity of atelectasis?", "Depends on extent-mild (small area) to severe (whole lung)."],
    ["How is atelectasis diagnosed?", "Chest X-ray, CT scan, bronchoscopy."],
    ["What treatments help lung expansion?", "Breathing exercises, physiotherapy, removing blockages."],
    ["Can atelectasis be reversed?", "Yes, especially if treated early."],
    ["What complications can occur?", "Pneumonia, oxygen deficiency."],
    ["When should medical help be taken?", "After surgery or if breathing becomes difficult."]
  ],
  Effusion: [
    ["What is pleural effusion?", "Fluid buildup between lung and chest wall."],
    ["What causes fluid around lungs?", "Heart failure, infections, cancer, kidney/liver disease."],
    ["What are symptoms of effusion?", "Breathlessness, chest pain, cough."],
    ["How serious is pleural effusion?", "Can range from mild to life-threatening."],
    ["What is the risk level?", "High if fluid accumulates rapidly or in large volume."],
    ["How is effusion diagnosed?", "X-ray, ultrasound, CT scan, fluid analysis."],
    ["What treatments remove fluid?", "Drainage (thoracentesis), medications."],
    ["Can effusion resolve on its own?", "Sometimes, if underlying cause is mild."],
    ["What complications can occur?", "Lung compression, infection."],
    ["When is drainage required?", "When breathing is affected or fluid is large."]
  ],
  Nodule: [
    ["What is a lung nodule?", "Small round growth in lung tissue."],
    ["What causes lung nodules?", "Infections, inflammation, benign tumors, or cancer."],
    ["What symptoms do nodules show?", "Usually none; sometimes mild cough."],
    ["Are nodules cancerous?", "Most are benign; some can be malignant."],
    ["What is the risk level of nodules?", "Depends on size, growth, and patient history."],
    ["How are nodules diagnosed?", "CT scan, follow-up imaging."],
    ["What tests confirm malignancy?", "Biopsy, PET scan."],
    ["What treatments are used?", "Observation or surgery if cancer suspected."],
    ["What complications can arise?", "Cancer progression if untreated."],
    ["When should biopsy be done?", "If nodule grows or appears suspicious."]
  ],
  Pneumothorax: [
    ["What is pneumothorax?", "Air enters space around lungs causing collapse."],
    ["What causes lung collapse due to air?", "Injury, spontaneous rupture, lung disease."],
    ["What are symptoms of pneumothorax?", "Sudden chest pain, breathlessness."],
    ["Is pneumothorax an emergency?", "Yes, especially severe cases."],
    ["What is the risk level?", "High in large or tension pneumothorax."],
    ["How is pneumothorax diagnosed?", "Chest X-ray, CT scan."],
    ["What treatments are used?", "Needle aspiration, chest tube."],
    ["Can pneumothorax recur?", "Yes, especially in smokers."],
    ["What complications occur?", "Severe lung collapse, respiratory failure."],
    ["When is surgery needed?", "If recurrent or severe cases."]
  ],
  Mass: [
    ["What is a lung mass?", "Large abnormal growth in lung (>3 cm)."],
    ["What causes lung masses?", "Cancer, infections, benign tumors."],
    ["What symptoms indicate a mass?", "Persistent cough, chest pain, weight loss."],
    ["Is a lung mass cancer?", "Not always, but high suspicion."],
    ["What is the risk level?", "High, especially in smokers."],
    ["How is a mass diagnosed?", "CT scan, PET scan."],
    ["What tests confirm cancer?", "Biopsy."],
    ["What treatments are available?", "Surgery, chemotherapy, radiation."],
    ["What complications can occur?", "Cancer spread, breathing issues."],
    ["When should immediate action be taken?", "If symptoms persist or worsen rapidly."]
  ]
};

function App() {
  const [file, setFile] = useState(null);
  const [data, setData] = useState(null);

  const API_BASE = "http://localhost:5000";

  const riskVisuals = {
    Low: { className: "risk-low", note: "Currently stable. Continue observation." },
    Medium: { className: "risk-medium", note: "Requires clinical follow-up soon." },
    High: { className: "risk-high", note: "Potentially severe. Immediate medical review advised." }
  };

  const handleUpload = async () => {
    if (!file) return alert("Select an image");

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${API_BASE}/search-image`, {
      method: "POST",
      body: formData
    });

    const result = await res.json();
    if (result.error) {
      alert(result.error);
      return;
    }
    setData(result);
  };

  const getRiskLevel = (confidence) => {
    const conf = Number(confidence);
    if (conf > 75) return "High";
    if (conf > 50) return "Medium";
    return "Low";
  };

  const topConfidence = Number(data?.predictions?.[0]?.confidence || 0);
  const riskGaugeValue = Math.max(5, Math.min(95, Math.round(topConfidence)));
  const normalizedRisk = getRiskLevel(topConfidence);
  const riskMeta = riskVisuals[normalizedRisk];
  const topDisease = data?.predictions?.[0]?.disease;
  const diseaseFaq = topDisease ? FAQ_DATA[topDisease] || [] : [];
  const infoRows = [
    ["Definition", data?.knowledge?.definition],
    ["Causes", data?.knowledge?.causes],
    ["Symptoms", data?.knowledge?.symptoms],
    ["Severity", data?.knowledge?.severity],
    ["Risk Notes", data?.knowledge?.risk],
    ["Diagnosis", data?.knowledge?.diagnosis],
    ["Treatment", data?.knowledge?.treatment],
    ["Reversibility", data?.knowledge?.reversibility],
    ["Complications", data?.knowledge?.complications],
    ["When to Seek Help", data?.knowledge?.when_to_seek_help]
  ];

  const getConfidenceNote = (score) => {
    if (score >= 80) return "Very strong match - high confidence pattern.";
    if (score >= 60) return "Strong match - clinically relevant confidence.";
    if (score >= 40) return "Moderate match - needs clinical correlation.";
    return "Low-to-moderate match - review with caution.";
  };

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>X-Ray Intelligence Console</h1>
          <p>Unified Knowledge Layer - Structured Clinical Intelligence</p>
        </div>
        <nav className="menu-tabs">
          <button className="menu-btn active">Analysis</button>
        </nav>
      </header>
      <div className="card uploader">
        <input type="file" onChange={(e) => setFile(e.target.files[0])} />
        <button onClick={handleUpload}>Analyze X-Ray</button>
      </div>

      {data && (
        <>
          <div className="card hero-card">
            <h2>Similar Cases</h2>
            <div className="similar-row">
              {data.results.map((r, i) => (
                <div key={i} className="similar-card">
                  <img
                    src={`${API_BASE}/image/${r.image_name}`}
                    alt={r.image_name}
                  />
                  <p className="similarity">{(r.score * 100).toFixed(1)}% similarity</p>
                  <p className="caption">{r.category}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="split-layout">
            <div className="card">
              <h2>Predicted Diseases</h2>
              <div className="pred-list">
                {data.predictions?.length ? (
                  data.predictions.map((p, i) => (
                    <div key={i} className="pred-item">
                      <div className="pred-bar-track">
                        <div
                          className="pred-bar-fill"
                          style={{ width: `${Math.max(4, Math.min(100, p.confidence))}%` }}
                        />
                        <div className="pred-row-head">
                          <span className="pred-name">{p.disease}</span>
                          <strong className="pred-score">{p.confidence}%</strong>
                        </div>
                      </div>
                      <p className="pred-note">{getConfidenceNote(p.confidence)}</p>
                    </div>
                  ))
                ) : (
                  <p>No confident prediction available.</p>
                )}
              </div>
            </div>

            <aside className={`card risk-panel ${riskMeta.className}`}>
              <h2>Risk Level</h2>
              <div
                className="gauge-wrap"
                style={{ "--needle-rotation": `${(riskGaugeValue * 1.8) - 90}deg` }}
              >
                <div className="gauge">
                  <div className="gauge-needle" />
                  <div className="gauge-center" />
                </div>
                <div className="gauge-label">
                  <strong>{riskGaugeValue}%</strong>
                  <span>{normalizedRisk}</span>
                </div>
              </div>
              <div className="risk-badge">
                <strong>{normalizedRisk}</strong>
              </div>
              <p>{riskMeta.note}</p>
            </aside>
          </div>

          <div className="card">
            <h2>Unified Medical Insight</h2>
            <div className="info-single-column">
              {infoRows.map(([label, value]) => (
                <div key={label} className="info-row">
                  <span className="info-label">{label}</span>
                  <span className="info-value">{value || "Not available."}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h2>{topDisease ? `${topDisease} FAQ` : "Disease FAQ"}</h2>
            <p className="subtle">Showing FAQ only for the top predicted disease.</p>
            {diseaseFaq.length ? (
              <div className="faq-list-inline">
                {diseaseFaq.map(([q, a], idx) => (
                  <details key={`${topDisease}-${idx}`} className="faq-item">
                    <summary>{q}</summary>
                    <p>{a}</p>
                  </details>
                ))}
              </div>
            ) : (
              <p>No disease-specific FAQ available until prediction is generated.</p>
            )}
          </div>
        </>
      )}
    </div>
  );
}

export default App;