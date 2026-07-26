let selectedFile = null;

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));

  document.getElementById(tab + '-panel').classList.add('active');
  event.target.classList.add('active');

  hideResults();
}

// ── Text search ───────────────────────────────────────────────────────────────
async function doTextSearch() {
  const query = document.getElementById('text-input').value.trim();
  if (!query) return;

  showLoading();

  try {
    const res = await fetch('/api/text-search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });

    const data = await res.json();
    showResults(data.results, `Text Search: "${query}"`);

  } catch (e) {
    alert('Text search failed. Is Flask running?');
  }

  hideLoading();
}

// ── Image search ──────────────────────────────────────────────────────────────
async function doImageSearch() {
  if (!selectedFile) return;

  showLoading();

  const form = new FormData();
  form.append('file', selectedFile);

  try {
    const res = await fetch('/api/image-search', {
      method: 'POST',
      body: form
    });

    const data = await res.json();
    showResults(data.results, 'Image Search Results');

  } catch (e) {
    alert('Image search failed. Is Flask running?');
  }

  hideLoading();
}

// ── Diagnosis Help ────────────────────────────────────────────────────────────
async function doDiagnosis() {
  const symptoms = document.getElementById('symptoms-input').value.trim();
  if (!symptoms) return;

  showLoading();

  try {
    const res = await fetch('/diagnosis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symptoms })
    });

    const data = await res.json();
    showDiagnosisResults(data.results, symptoms);

  } catch (e) {
    alert('Diagnosis failed. Is Flask running?');
  }

  hideLoading();
}

// ── File handling ─────────────────────────────────────────────────────────────
function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) loadPreview(file);
}

function handleDrop(event) {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) loadPreview(file);
}

function loadPreview(file) {
  selectedFile = file;

  const reader = new FileReader();
  reader.onload = e => {
    const preview = document.getElementById('preview');
    preview.src = e.target.result;
    preview.style.display = 'block';

    document.getElementById('drop-label').style.display = 'none';
    document.getElementById('img-search-btn').style.display = 'inline-block';
  };

  reader.readAsDataURL(file);
}

// ── Show Image/Text Results ───────────────────────────────────────────────────
function showResults(results, title) {
  const section = document.getElementById('results-section');
  const grid = document.getElementById('results-grid');

  document.getElementById('results-title').textContent =
    `${title} — Top ${results.length} Results`;

  grid.innerHTML = results.map(r => `
    <div class="result-card">
      <img src="${r.image_url}" alt="${r.image_name}"
           onerror="this.src='https://placehold.co/200x200/0d1f35/64748b?text=X-Ray'"/>
      <div class="card-info">
        <div class="card-category">${r.category}</div>

        <div class="score-bar-bg">
          <div class="score-bar" style="width:${(r.score * 100).toFixed(0)}%"></div>
        </div>

        <div class="score-label">
          Similarity: ${(r.score * 100).toFixed(1)}%
        </div>

        <div class="card-name">${r.image_name}</div>
      </div>
    </div>
  `).join('');

  section.style.display = 'block';
}

// ── Show Diagnosis Results ────────────────────────────────────────────────────
function showDiagnosisResults(results, symptoms) {
  const section = document.getElementById('results-section');
  const grid = document.getElementById('results-grid');

  document.getElementById('results-title').textContent =
    `Diagnosis Results for "${symptoms}"`;

  grid.innerHTML = results.map(r => {
    const datasetInfo = r.dataset_info || {};
    const stats = `
      <div class="dataset-stats">
        <div class="stat-item">
          <span class="stat-label">Dataset Cases:</span>
          <span class="stat-value">${datasetInfo.count || 'N/A'}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Prevalence:</span>
          <span class="stat-value">${datasetInfo.prevalence ? datasetInfo.prevalence.toFixed(1) + '%' : 'N/A'}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Age Range:</span>
          <span class="stat-value">${datasetInfo.age_range || 'N/A'}</span>
        </div>
        <div class="stat-item">
          <span class="stat-label">Gender:</span>
          <span class="stat-value">${datasetInfo.gender_distribution ? `M: ${datasetInfo.gender_distribution.M}, F: ${datasetInfo.gender_distribution.F}` : 'N/A'}</span>
        </div>
      </div>
    `;

    return `
    <div class="result-card">
      <div class="card-info">
        <div class="card-category">${r.disease}</div>

        <div class="score-bar-bg">
          <div class="score-bar" style="width:${(r.similarity * 100).toFixed(0)}%"></div>
        </div>

        <div class="score-label">
          Match: ${(r.similarity * 100).toFixed(1)}%
        </div>

        <div class="card-name">
          Possible Condition
        </div>

        ${stats}
      </div>
    </div>
  `}).join('');

  section.style.display = 'block';
}

// ── Utility ───────────────────────────────────────────────────────────────────
function hideResults() {
  document.getElementById('results-section').style.display = 'none';
}

function showLoading() {
  document.getElementById('loading').style.display = 'block';
  document.getElementById('results-section').style.display = 'none';
}

function hideLoading() {
  document.getElementById('loading').style.display = 'none';
}

// ── Enter key support ─────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('text-input')
    .addEventListener('keydown', e => {
      if (e.key === 'Enter') doTextSearch();
    });
});