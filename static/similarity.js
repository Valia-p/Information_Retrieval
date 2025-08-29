// static/similarity.js
document.addEventListener("DOMContentLoaded", () => {
  const selMember   = document.getElementById("member-select");
  const datalist    = document.getElementById("member-list");
  const inpName     = document.getElementById("name-input");
  const selTopK     = document.getElementById("sim-topk");
  const btnShow     = document.getElementById("btn-show");
  const neighborsTitle = document.getElementById("neighbors-title");
  const neighborsBody  = document.getElementById("neighbors-tbody");
  const chartCanvas    = document.getElementById("similarity-chart");
  let chart = null;

  function setTitle(n){ neighborsTitle.textContent = n ? `Neighbors of: ${n}` : "Neighbors"; }
  function renderTable(neigh){
    if (!neigh || neigh.length===0){
      neighborsBody.innerHTML = `<tr><td colspan="3" class="muted" style="text-align:center">No data</td></tr>`;
      return;
    }
    neighborsBody.innerHTML = neigh.map((r,i)=>`
      <tr>
        <td style="text-align:center;">${i+1}</td>
        <td style="padding-left:390px">${r.member}</td>
        <td style="text-align:center"><span class="badge">${r.score.toFixed(4)}</span></td>
      </tr>
    `).join("");
  }
  function renderChart(labels, scores){
    if (chart) chart.destroy();
    const ctx = chartCanvas.getContext("2d");
    const g = ctx.createLinearGradient(0,0,0,240);
    g.addColorStop(0,"rgba(33,150,243,.9)");
    g.addColorStop(.5,"rgba(33,150,243,.55)");
    g.addColorStop(1,"rgba(33,150,243,.15)");
    chart = new Chart(ctx,{
      type:"bar",
      data:{ labels, datasets:[{ data:scores, backgroundColor:g, borderColor:"rgba(33,150,243,.9)", borderWidth:1, borderRadius:8, maxBarThickness:42 }]},
      options:{ responsive:true, maintainAspectRatio:false, plugins:{legend:{display:false}}, scales:{y:{beginAtZero:true,suggestedMax:1}} }
    });
  }

  async function loadMembers(){
    const res = await fetch("/entities?type=member&full=1");
    const data = await res.json();
    const items = data.items || [];
    selMember.innerHTML = `<option value="" selected>Select a member…</option>` +
      items.map(m => `<option value="${m.id}">${m.name}</option>`).join("");
    datalist.innerHTML = items.map(m => `<option value="${m.name}"></option>`).join("");
  }

  async function showSimilar(){
    const k = parseInt(selTopK.value || "10", 10);
    const chosenId = selMember.value && selMember.value.trim();
    let url, label;

    if (chosenId){
      url = `/similarity/member?id=${encodeURIComponent(chosenId)}&k=${k}`;
      // βρες το όνομα που αντιστοιχεί στο option για τον τίτλο
      label = selMember.options[selMember.selectedIndex]?.text || "";
    }else{
      const typed = (inpName.value || "").trim();
      if (!typed) return;
      url = `/similarity/member?name=${encodeURIComponent(typed)}&k=${k}`;
      label = typed;
    }

    const res = await fetch(url);
    const j = await res.json();
    if (!res.ok){
      alert(j.error || "No results."); return;
    }
    setTitle(j.name || label);
    const labels = (j.neighbors||[]).map(d=>d.member);
    const scores = (j.neighbors||[]).map(d=>d.score);
    renderChart(labels, scores);
    renderTable(j.neighbors||[]);
  }

  btnShow?.addEventListener("click", showSimilar);
  selMember?.addEventListener("change", showSimilar);
  inpName?.addEventListener("keydown", e => { if (e.key==="Enter") showSimilar(); });

  loadMembers();
});
