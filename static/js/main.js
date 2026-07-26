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
    const res  = await fetch('/api/text-search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query })
    });
    const data = await res.json();
    showResults(data.results, `Text Search: "${query}"`);
  } catch(e) {
    alert('Search failed. Is Flask running?');
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
    const res  = await fetch('/api/image-search', {
      method: 'POST',
      body: form
    });
    const data = await res.json();
    showResults(data.results, 'Image Search Results');
  } catch(e) {
    alert('Search failed. Is Flask running?');
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

// ── Results ───────────────────────────────────────────────────────────────────
function showResults(results, title) {
  const section = document.getElementById('results-section');
  const grid    = document.getElementById('results-grid');

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
        <div class="score-label">Similarity: ${(r.score * 100).toFixed(1)}%</div>
        <div class="card-name">${r.image_name}</div>
      </div>
    </div>
  `).join('');

  section.style.display = 'block';
}

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

// ── Enter key on text input ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('text-input')
    .addEventListener('keydown', e => {
      if (e.key === 'Enter') doTextSearch();
    });
});
