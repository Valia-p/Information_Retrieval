// static/drift.js
document.addEventListener("DOMContentLoaded", () => {
  const typeSelect   = document.getElementById("drift-type");     // member | party
  const entitySelect = document.getElementById("drift-entity");   // <select> γεμίζει με /entities?full=1
  const topInfo      = document.getElementById("drift-top-info"); // optional helper text
  const btnShow      = document.getElementById("btn-drift");
  const chartCanvas  = document.getElementById("drift-chart");

  if (!typeSelect || !entitySelect || !btnShow || !chartCanvas) return;

  let chart = null;

  async function loadEntities(){
    const t = typeSelect.value;
    entitySelect.innerHTML = '<option value="" selected disabled>Φόρτωση…</option>';
    try{
      const res = await fetch(`/entities?type=${encodeURIComponent(t)}&full=1`);
      const data = await res.json();
      const items = data.items || [];
      entitySelect.innerHTML = '<option value="" selected disabled>Επιλογή…</option>' +
        items.map(it => `<option value="${it.id}">${it.name}</option>`).join("");
    }catch(e){
      console.error(e);
      entitySelect.innerHTML = '<option value="" selected disabled>Αποτυχία φόρτωσης</option>';
    }
  }

  function renderChart(labels, values, title){
    if (chart) chart.destroy();
    const ctx = chartCanvas.getContext("2d");
    chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [{
          label: title || "Θεματική μετατόπιση",
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
          x: { title: { display: true, text: "Έτος" } },
          y: { title: { display: true, text: "Drift (1 − cosine)" }, beginAtZero: true, suggestedMax: 1 }
        }
      }
    });
  }

  async function showDrift(){
    const t  = typeSelect.value;
    const id = entitySelect.value;
    if (!t || !id){
      alert("Διάλεξε τύπο και οντότητα."); return;
    }
    const url = `/extras/topic_drift?type=${encodeURIComponent(t)}&id=${encodeURIComponent(id)}`;
    const res = await fetch(url);
    const j   = await res.json();
    if (!res.ok){
      alert(j.error || "Σφάλμα"); return;
    }

    const drifts = j.drifts || [];
    if (!drifts.length){
      // πιθανόν μόνο ένα έτος διαθέσιμο → δεν ορίζεται drift
      if (chart) chart.destroy();
      if (topInfo) topInfo.textContent = "Δεν υπάρχουν αρκετά έτη για υπολογισμό μετατόπισης.";
      return;
    }

    const labels = drifts.map(d => String(d.year));
    const values = drifts.map(d => Number(d.drift) || 0);
    renderChart(labels, values, `Drift: ${j.name || ""}`);
    if (topInfo) topInfo.textContent = `${j.type === "member" ? "Μέλος" : "Κόμμα"}: ${j.name}`;
  }

  typeSelect.addEventListener("change", loadEntities);
  btnShow.addEventListener("click", showDrift);
  entitySelect.addEventListener("change", showDrift); // auto-refresh on select change

  // initial
  loadEntities();
});
