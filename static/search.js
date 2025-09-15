// static/search.js
document.addEventListener('DOMContentLoaded', function () {
  const searchForm      = document.getElementById('search-form');
  const searchInput     = document.getElementById('search-input');
  const searchResults   = document.getElementById('search-results');
  const dateRangeFilter = document.getElementById('date-range');
  const partyFilter     = document.getElementById('party-filter');
  const mpFilter        = document.getElementById('mp-filter');

  if (!searchForm || !searchInput || !searchResults) return;

  // Helpers
  function escapeRegExp(str){return str.replace(/[.*+?^${}()|[\]\\]/g,"\\$&");}
  function highlight(text, query){
    const parts=(query||"").trim().split(/\s+/).filter(Boolean);
    if(!parts.length) return text;
    const rx = new RegExp("(" + parts.map(escapeRegExp).join("|") + ")", "gi");
    return text.replace(rx, "<mark>$1</mark>");
  }

  async function populateSelect(selectEl, url, allLabel) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const items = Array.isArray(data.items) ? data.items : [];
      // reset & add "All"
      selectEl.innerHTML = "";
      const allOpt = document.createElement('option');
      allOpt.value = "all";
      allOpt.textContent = allLabel;
      selectEl.appendChild(allOpt);
      // add items
      for (const name of items) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        selectEl.appendChild(opt);
      }
    } catch (err) {
      console.error('populateSelect error:', err);
      selectEl.innerHTML = `<option value="all">${allLabel}</option>`;
    }
  }

  // Fill dropdowns from DB
  populateSelect(partyFilter, '/entities?type=party',  'All Parties');
  populateSelect(mpFilter,    '/entities?type=member', 'All Members');

  // If party/member/date changes and a query exists, rerun the search
  function reRunIfQueryExists() {
    const q = (searchInput.value || '').trim();
    if (q) searchForm.dispatchEvent(new Event('submit', {cancelable:true}));
  }
  dateRangeFilter.addEventListener('change', reRunIfQueryExists);
  partyFilter.addEventListener('change', reRunIfQueryExists);
  mpFilter.addEventListener('change', reRunIfQueryExists);

  // Submit handler
  searchForm.addEventListener('submit', function (e) {
    e.preventDefault();

    const query     = (searchInput.value || '').trim();
    const dateRange = dateRangeFilter.value; // "all" or "YYYY-YYYY"
    const party     = partyFilter.value;     // "all" or party
    const mp        = mpFilter.value;        // "all" or member

    if (!query) {
      searchResults.innerHTML = '<p class="placeholder-text">Enter search terms above to see results</p>';
      return;
    }

    searchResults.innerHTML = '<p class="placeholder-text">Searching…</p>';

    fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, dateRange, party, mp })
    })
    .then(r => {
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return r.json();
    })
    .then(data => {
      // clear box
      searchResults.innerHTML = "";

      if (!Array.isArray(data) || data.length === 0) {
        searchResults.innerHTML = `
            <p style="text-align: center; margin-top: 20px;">
              No results found.
            </p>
          `;
        return;
      }

      // render results
      for (const item of data) {
        const card = document.createElement('div');
        card.style.background = "#EBF4F6";
        card.style.borderRadius = "6px";
        card.style.boxShadow = "0 2px 10px rgba(0,0,0,0.06)";
        card.style.padding = "16px";
        card.style.margin = "10px 0";

        card.innerHTML = `
          <div style="display:flex; align-items:baseline; gap:8px; flex-wrap:wrap;">
            <h3 style="margin:0;">${item.member}</h3>
            <span>(${item.party})</span>
            <span style="opacity:.7;">– ${item.date}</span>
            <span style="margin-left:auto; font-size:.9rem; opacity:.8;">Score: ${item.score}</span>
          </div>
          <p style="margin-top:8px;">${highlight(item.speech, query)}</p>
        `;
        searchResults.appendChild(card);
      }
    })
    .catch(err => {
      console.error(err);
      searchResults.innerHTML = "<p>Error while searching.</p>";
    });
  });
});
