// static/drift.js
document.addEventListener("DOMContentLoaded", () => {
  // Selections & elements
  const typeSelect   = document.getElementById("drift-type");     // "member" | "party"
  const entitySelect = document.getElementById("drift-entity");
  const topInfo      = document.getElementById("drift-top-info");
  const btnShow      = document.getElementById("btn-drift");
  const chartCanvas  = document.getElementById("drift-chart");

  if (!typeSelect || !entitySelect || !btnShow || !chartCanvas) return;

  let chart = null;

  // Fetch and populate entities based on selected type
  async function loadEntities(){
    const t = typeSelect.value;
    entitySelect.innerHTML = '<option value="" selected disabled>Loading…</option>';
    try{
      const res = await fetch(`/entities?type=${encodeURIComponent(t)}&full=1`);
      const data = await res.json();
      const items = data.items || [];
      entitySelect.innerHTML =
        '<option value="" selected disabled>Select…</option>' +
        items.map(it => `<option value="${it.id}">${it.name}</option>`).join("");
    }catch(e){
      console.error(e);
      entitySelect.innerHTML = '<option value="" selected disabled>Failed to load</option>';
    }
  }

  // Render (or re-render) the Chart.js
  function renderChart(labels, values, title){
    if (chart) chart.destroy();
    const ctx = chartCanvas.getContext("2d");
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: title || "Topic drift",
          data: values,
          tension: 0.25,
          pointRadius: 2,
          borderWidth: 2
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: true } },
        scales: {
          x: { title: { display: true, text: "Year" } },
          y: { title: { display: true, text: "Drift (1 − cosine)" }, beginAtZero: true, suggestedMax: 1 }
        }
      }
    });
  }

  // Fetch drift data and update the chart and helper text
  async function showDrift(){
    const t  = typeSelect.value;
    const id = entitySelect.value;
    if (!t || !id){
      alert("Please choose a type and an entity."); return;
    }
    const url = `/extras/topic_drift?type=${encodeURIComponent(t)}&id=${encodeURIComponent(id)}`;
    const res = await fetch(url);
    const j   = await res.json();
    if (!res.ok){
      alert(j.error || "Error"); return;
    }

    const drifts = j.drifts || [];
    if (!drifts.length){
      // Possibly only one year available → drift not defined
      if (chart) chart.destroy();
      if (topInfo) topInfo.textContent = "Not enough years to compute drift.";
      return;
    }

    const labels = drifts.map(d => String(d.year));
    const values = drifts.map(d => Number(d.drift) || 0);
    renderChart(labels, values, `Drift: ${j.name || ""}`);
    if (topInfo) topInfo.textContent = `${j.type === "member" ? "Member" : "Party"}: ${j.name}`;
  }

  // Event bindings
  typeSelect.addEventListener("change", loadEntities);
  btnShow.addEventListener("click", showDrift);
  entitySelect.addEventListener("change", showDrift); // auto-refresh on select change

  // Initial load
  loadEntities();
});
