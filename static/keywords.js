let keywordLineChart = null;

document.addEventListener('DOMContentLoaded', function() {
  const trackingView = document.getElementById('tracking-view');
  const entitySelect = document.getElementById('entity-select');
  const topkSelect = document.getElementById('topk-select');
  const addKeywordButton = document.getElementById('add-keyword');
  const keywordCanvas = document.getElementById('keyword-canvas');

  if (!trackingView || !entitySelect || !addKeywordButton || !keywordCanvas || !topkSelect) return;

  function destroyChart() {
    if (keywordLineChart) {
      keywordLineChart.destroy();
      keywordLineChart = null;
    }
  }

  function buildTopKTimeSeries(data, topK = 8) {
    const years = Object.keys(data || {}).map(Number).sort((a, b) => a - b);
    const byKeyword = new Map();

    years.forEach(y => {
      (data[String(y)] || []).forEach(({ keyword, score }) => {
        if (!byKeyword.has(keyword)) byKeyword.set(keyword, new Map());
        byKeyword.get(keyword).set(y, Number(score) || 0);
      });
    });

    const totals = Array.from(byKeyword.entries()).map(([kw, m]) => {
      let sum = 0;
      m.forEach(v => sum += v);
      return { keyword: kw, total: sum };
    }).sort((a, b) => b.total - a.total);

    const selected = totals.slice(0, topK).map(o => o.keyword);

    const datasets = selected.map(kw => {
      const m = byKeyword.get(kw);
      const series = years.map(y => m.get(y) || 0);
      return {
        label: kw,
        data: series,
        tension: 0.25,
        pointRadius: 2,
        borderWidth: 2
      };
    });

    return { years: years.map(String), datasets };
  }

  function renderChart(yearLabels, datasets) {
    destroyChart();
    const ctx = keywordCanvas.getContext('2d');
    keywordLineChart = new Chart(ctx, {
      type: 'line',
      data: { labels: yearLabels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { title: { display: true, text: 'Year' } },
          y: { title: { display: true, text: 'Score' }, beginAtZero: true }
        }
      }
    });
  }

  async function loadEntities(type) {
    if (type === 'overall') {
      entitySelect.style.display = 'none';
      return;
    }
    entitySelect.style.display = '';
    entitySelect.innerHTML = '<option value="" selected disabled>Choose from list…</option>';
    try {
      const res = await fetch(`/entities?type=${encodeURIComponent(type)}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const items = data.items || [];
      items.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        entitySelect.appendChild(opt);
      });
      if (items.length) entitySelect.selectedIndex = 1;
    } catch (e) {
      console.error('loadEntities error:', e);
      entitySelect.innerHTML = '<option value="" selected disabled>Failed to load</option>';
    }
  }


  loadEntities(trackingView.value);
  trackingView.addEventListener('change', () => loadEntities(trackingView.value));

  addKeywordButton.addEventListener('click', async () => {
    const type = trackingView.value;
    const name = type === 'overall' ? '' : (entitySelect.value || '');
    if (type !== 'overall' && !name) {
      alert('Chose from the dropdown.');
      return;
    }
    
    let topK = parseInt(topkSelect.value, 10);
    if (!Number.isFinite(topK)) topK = 8;
    topK = Math.min(10, Math.max(1, topK));

    try {
      const res = await fetch('/keywords/by_year', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ type, name })
      });
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);

      const { years, datasets } = buildTopKTimeSeries(data, topK);
      if (!datasets.length) {
        destroyChart();
        return;
      }
      renderChart(years, datasets);
    } catch (err) {
      console.error(err);
      destroyChart();
    }
  });
});
