// static/themes.js
// Drives the Thematic Analysis section (#thematic)
// Endpoints used:
//   GET  /themes/overview
//   GET  /themes/cluster?id=NN
//   GET  /themes/embedding2d

(function () {
  const els = {
    section: document.getElementById('thematic'),
    chartCanvas: document.getElementById('theme-chart'),
    list: document.getElementById('themes-list'),
    panel: document.getElementById('cluster-panel'),
    panelTitle: document.getElementById('cluster-panel-title'),
    kpiSize: document.getElementById('kpi-size'),
    kpiDates: document.getElementById('kpi-dates'),
    kpiAvgLen: document.getElementById('kpi-avglen'),
    kpiCohesion: document.getElementById('kpi-cohesion'),
    kwWrap: document.getElementById('cluster-panel-keywords'),
    partyBars: document.getElementById('ta-party-bars'),
    topMembers: document.getElementById('ta-top-members'),
    reprMeta: document.getElementById('ta-repr-meta'),
    reprText: document.getElementById('ta-repr-text'),
    table: document.getElementById('cluster-panel-table'),
        outliersWrap: document.getElementById('ta-outliers'),

  };

  // Inject UI upgrades
  const style = document.createElement('style');
  style.textContent = `
    .theme-card { transition: transform .15s ease, box-shadow .15s ease; cursor: pointer; }
    .theme-card:hover { transform: translateY(-3px); box-shadow: 0 10px 30px rgba(0,0,0,.09); }
    .badge { display:inline-block; padding:.3rem .55rem; border-radius:999px; background:linear-gradient(180deg,#f4f8ff,#eef4ff); border:1px solid rgba(99,102,241,.18); font-size:.85rem; margin:.2rem .25rem; }
    .ta-kpis { display:grid; grid-template-columns: repeat(4,minmax(160px,1fr)); gap: .85rem; }
    .ta-kpi { background: linear-gradient(180deg,#f7fafe,#ffffff); border:1px solid rgba(59,130,246,.18); border-radius:14px; padding:1rem 1.1rem; box-shadow: 0 1px 0 rgba(0,0,0,.02) inset; }
    .ta-kpi:nth-child(2){ background:linear-gradient(180deg,#f3fff6,#ffffff); border-color: rgba(16,185,129,.18);}
    .ta-kpi:nth-child(3){ background:linear-gradient(180deg,#fffaf3,#ffffff); border-color: rgba(245,158,11,.18);}
    .ta-kpi:nth-child(4){ background:linear-gradient(180deg,#f3f9ff,#ffffff); border-color: rgba(59,130,246,.18);}
    .ta-kpi-label{ color:#6b7280; font-weight:600; letter-spacing:.2px; }
    .ta-kpi-value{ font-weight:800; font-size:1.35rem; margin-top:.35rem; color:#1f2937; }
    .ta-keywords{ display:flex; flex-wrap:wrap; gap:.25rem; }
    .skeleton{ position:relative; overflow:hidden; background:#eef2f7; border-radius:10px; }
    .skeleton::after{ content:''; position:absolute; inset:0; transform:translateX(-100%); background:linear-gradient(90deg,transparent,rgba(255,255,255,.6),transparent); animation:shimmer 1.2s infinite; }
    @keyframes shimmer{ 100%{ transform:translateX(100%); } }
    .muted{ color:#6b7280; }
    .card.ta-accent{ border:1px solid rgba(59,130,246,.12); background:linear-gradient(180deg,#f8fbff,#ffffff); }
    .ta-embed-card{ margin-top:var(--spacing-lg); }
    @media (max-width: 900px){ .ta-kpis{ grid-template-columns: 1fr 1fr; } }
  `;
  document.head.appendChild(style);

  let overview = null;
  let chartInstance = null;
  let partyChart = null;
  let membersChart = null;
  let embedChart = null;

  const HEADER_OFFSET = 70; // for smooth jump under fixed header

  // Utils -----------------------------------------------------
  const fmt = {
    number(n) { if (n==null || Number.isNaN(n)) return '—'; return new Intl.NumberFormat('el-GR').format(n); },
    dateRange(min,max){ if(!min && !max) return '—'; if(min && max) return `${min} — ${max}`; return min||max||'—'; },
    cohesion(x){ if (x==null || Number.isNaN(x)) return '—'; return x.toFixed(3); },
    escape(s){ const d=document.createElement('div'); d.textContent=(s??'').toString(); return d.innerHTML; }
  };

  const badge = (t)=>{ const s=document.createElement('span'); s.className='badge'; s.textContent=t; return s; };
  const clearNode = (n)=>{ while(n && n.firstChild) n.removeChild(n.firstChild); };
  const skeleton = (h=20)=>{ const d=document.createElement('div'); d.className='skeleton'; d.style.height= typeof h==='number'? `${h}px`:h; return d; };

  // Overview --------------------------------------------------
  async function loadOverview(){
    try{
      const res = await fetch('/themes/overview');
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      overview = (data.clusters||[]).slice().sort((a,b)=>b.size-a.size);
      renderChart(overview);
      renderCards(overview);
      ensureEmbeddingCard();
      loadEmbedding();
    }catch(err){
      console.error('Thematic overview error:', err);
      if (els.list) els.list.innerHTML = '<div class="muted">Δεν βρέθηκαν δεδομένα θεματικών clusters.</div>';
    }
  }

  function renderChart(items){
    if (!els.chartCanvas) return;
    if (chartInstance) chartInstance.destroy();

    const labels = items.map(c=>`C${c.cluster_id}`);
    const sizes  = items.map(c=>c.size);
    const tipMap = new Map(items.map(c=>[String(c.cluster_id),(c.top_keywords||[]).join(', ')]));

    chartInstance = new Chart(els.chartCanvas.getContext('2d'),{
      type:'bar',
      data:{ labels, datasets:[{ label:'Number of Speeches', data:sizes }] },
      options:{
        responsive:true, maintainAspectRatio:false,
        plugins:{ legend:{display:false}, tooltip:{ callbacks:{
          title:(ctx)=>ctx[0]?.label||'',
          label:(ctx)=>{ const cid=(ctx.label||'').replace(/^C/,''); const kws=tipMap.get(cid)||''; const size=ctx.raw; const lines=[`Μέγεθος: ${fmt.number(size)}`]; if(kws) lines.push(`Λέξεις: ${kws}`); return lines; }
        }}},
        scales:{ x:{ ticks:{ autoSkip:true, maxRotation:0 } }, y:{ beginAtZero:true, title:{display:true,text:'Speeches'} } },
        onClick(_e,elsIdx){ if(!elsIdx?.length) return; const idx=elsIdx[0].index; const cid=items[idx]?.cluster_id; if (typeof cid==='number') openCluster(cid,true); }
      }
    });
  }

  function cardTemplate(c){
    const div=document.createElement('div');
    div.className='card theme-card'; div.tabIndex=0; div.setAttribute('role','button');
    div.setAttribute('aria-label',`Cluster ${c.cluster_id}`);
    const kw=(c.top_keywords||[]).slice(0,6);
    div.innerHTML=`
      <div class="card-header" style="display:flex; align-items:baseline; justify-content:space-between; gap:.5rem;">
        <div>
          <h3 class="card-title" style="margin-bottom:.2rem;">Cluster C${fmt.escape(c.cluster_id)}</h3>
          <div class="muted" style="font-size:.9rem;">${fmt.number(c.size)} Speeches</div>
        </div>
        <button class="primary" style="white-space:nowrap">Open</button>
      </div>
      <div class="card-body">
        <div class="ta-keywords">${kw.map(k=>`<span class=\"badge\">${fmt.escape(k)}</span>`).join(' ')||'<em class="muted">Χωρίς λέξεις</em>'}</div>
      </div>`;

    div.addEventListener('click',(e)=>{ if(e.target.closest('button')) e.preventDefault(); openCluster(c.cluster_id,true); });
    div.addEventListener('keydown',(e)=>{ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); openCluster(c.cluster_id,true);} });
    return div;
  }

  function renderCards(items){ if(!els.list) return; clearNode(els.list); const f=document.createDocumentFragment(); items.forEach(c=>f.appendChild(cardTemplate(c))); els.list.appendChild(f); }

    function renderOutliers(items){
    clearNode(els.outliersWrap);
    if (!items || items.length === 0){
      els.outliersWrap.innerHTML = '<div class="muted">No outliers were detected for this cluster.</div>';
      return;
    }
    const wrap = document.createElement('div');
    wrap.style.display = 'grid';
    wrap.style.gridTemplateColumns = 'repeat(auto-fit, minmax(280px, 1fr))';
    wrap.style.gap = '.75rem';
    wrap.className = 'outliers-grid';

    items.forEach(o => {
      const card = document.createElement('div');
      card.className = 'card outlier-card';
      card.innerHTML = `
        <div class="card-header" style="display:flex;justify-content:space-between;align-items:baseline;gap:.5rem;">
          <div>
            <div style="font-weight:600">${fmt.escape(o.member || '—')}</div>
            <div class="muted" style="font-size:.9rem">${fmt.escape(o.party || '—')} • ${fmt.escape(o.date || '—')}</div>
          </div>
          <span class="badge" title="Cosine προς centroid">sim: ${o.sim != null ? Number(o.sim).toFixed(3) : '—'}</span>
        </div>
        <div class="card-body">
          <div class="outlier-excerpt">${fmt.escape(o.excerpt || '')}</div>
          <button class="copy-btn" style="margin-top:.5rem">Copy Excerpt</button>
        </div>`;
      card.querySelector('.copy-btn')?.addEventListener('click', async (e)=>{
        try{
          await navigator.clipboard.writeText(o.excerpt || '');
          e.target.textContent = 'Copied!';
          setTimeout(()=> e.target.textContent = 'Copy Excerpt', 1200);
        }catch{}
      });
      wrap.appendChild(card);
    });

    els.outliersWrap.appendChild(wrap);
  }

  // Cluster details -------------------------------------------
  async function openCluster(clusterId, jump){
    try{
      els.panel.style.display='block'; els.panelTitle.textContent=`Cluster C${clusterId}`;
      [els.kpiSize,els.kpiDates,els.kpiAvgLen,els.kpiCohesion].forEach(el=>el.textContent='…');
      clearNode(els.kwWrap); clearNode(els.partyBars); clearNode(els.topMembers); setRepr('—','<em>—</em>'); fillTable([]);
      els.kwWrap.appendChild(skeleton(28)); els.partyBars.appendChild(skeleton(160)); els.topMembers.appendChild(skeleton(160));

      if (jump) smoothJumpTo(els.panel);

      const res=await fetch(`/themes/cluster?id=${encodeURIComponent(clusterId)}`);
      const data=await res.json(); if(data.error) throw new Error(data.error);

      // KPIs
      els.kpiSize.textContent=fmt.number(data.size);
      els.kpiDates.textContent=fmt.dateRange(data.date_min,data.date_max);
      els.kpiAvgLen.textContent=data.avg_chars?`${fmt.number(Math.round(data.avg_chars))} χαρακτ.`:'—';
      els.kpiCohesion.textContent=fmt.cohesion(data.avg_centroid_sim);

      // Keywords
      clearNode(els.kwWrap); const kws=(data.top_keywords||[]).slice(0,12);
      if(kws.length===0){ els.kwWrap.innerHTML='<em class="muted">Δεν υπάρχουν keywords</em>'; } else { kws.forEach(k=>els.kwWrap.appendChild(badge(k))); }
      clearNode(els.outliersWrap);
      els.outliersWrap.appendChild(skeleton(90));

      // Charts
      renderPartyChart(data.party_counts||{});
      renderMembersChart(data.member_top||[]);
      renderOutliers(data.outliers || []);

      // Representative
      if(data.repr){ const {member,party,date,excerpt}=data.repr; const meta=[member,party,date].filter(Boolean).join(' • '); setRepr(meta||'—', fmt.escape(excerpt||'—')); addCopyBtn(); }

      // Samples
      fillTable(data.samples||[]);
    }catch(err){ console.error('Cluster load error:',err); els.kwWrap.innerHTML='<em class="muted">Σφάλμα φόρτωσης δεδομένων</em>'; }
  }

  function smoothJumpTo(el){
    const y = el.getBoundingClientRect().top + window.pageYOffset - HEADER_OFFSET;
    window.scrollTo({ top: y, behavior: 'smooth' });
    history.replaceState(null,'', '#thematic');
  }

  function setRepr(metaHtml, textHtml){ els.reprMeta.innerHTML=metaHtml; els.reprText.innerHTML=textHtml; }

  function addCopyBtn(){ const wrap=els.reprMeta.parentElement; if(!wrap||wrap.querySelector('.copy-btn')) return; const btn=document.createElement('button'); btn.className='copy-btn'; btn.textContent='Copy Excerpt'; btn.style.marginTop='.5rem'; btn.addEventListener('click', async()=>{ try{ const text=els.reprText.innerText||els.reprText.textContent||''; await navigator.clipboard.writeText(text); btn.textContent='Copied!'; setTimeout(()=>btn.textContent='Copy Excerpt',1400);}catch{} }); wrap.appendChild(btn); }

  function renderPartyChart(counts){ clearNode(els.partyBars); const entries=Object.entries(counts).sort((a,b)=>b[1]-a[1]); if(entries.length===0){ els.partyBars.innerHTML='<div class="muted">There are no data.</div>'; return; } const canvas=document.createElement('canvas'); canvas.height=200; els.partyBars.appendChild(canvas); const labels=entries.map(e=>e[0]); const values=entries.map(e=>e[1]); if(partyChart) partyChart.destroy(); partyChart=new Chart(canvas.getContext('2d'),{ type:'bar', data:{ labels, datasets:[{ label:'Speeches', data:values }] }, options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false, plugins:{ legend:{display:false} }, scales:{ x:{ beginAtZero:true } } } }); }

  function renderMembersChart(memberTop){ clearNode(els.topMembers); if(!memberTop||memberTop.length===0){ els.topMembers.innerHTML='<li class="muted">There are no data.</li>'; return; } const canvas=document.createElement('canvas'); canvas.height=Math.max(180, memberTop.length*26); els.topMembers.appendChild(canvas); const labels=memberTop.map(m=>m.member); const values=memberTop.map(m=>m.count); if(membersChart) membersChart.destroy(); membersChart=new Chart(canvas.getContext('2d'),{ type:'bar', data:{ labels, datasets:[{ label:'Ομιλίες', data:values }] }, options:{ indexAxis:'y', responsive:true, maintainAspectRatio:false, plugins:{ legend:{display:false} }, scales:{ x:{ beginAtZero:true } } } }); }

  function fillTable(samples){ const tbody=els.table?.querySelector('tbody'); if(!tbody) return; clearNode(tbody); if(!samples||samples.length===0){ const tr=document.createElement('tr'); const td=document.createElement('td'); td.colSpan=4; td.className='muted'; td.style.textAlign='center'; td.textContent='There are no speech data.'; tr.appendChild(td); tbody.appendChild(tr); return; } samples.forEach((s,i)=>{ const tr=document.createElement('tr'); const tdIdx=document.createElement('td'); tdIdx.textContent=String(i+1); tdIdx.style.textAlign = 'center'; const tdMember=document.createElement('td'); tdMember.textContent=s.member||'—'; tdMember.style.paddingLeft="100px"; const tdParty=document.createElement('td'); tdParty.textContent=s.party||'—'; const tdDate=document.createElement('td'); tdDate.textContent=s.date||'—'; tdDate.style.textAlign = 'center'; tr.appendChild(tdIdx); tr.appendChild(tdMember); tr.appendChild(tdParty); tr.appendChild(tdDate); tbody.appendChild(tr); }); }

  // Embedding (LSI→PCA 2D) -----------------------------------
  function ensureEmbeddingCard(){
    if (!els.section || document.getElementById('embedding-card')) return;
    const card=document.createElement('div'); card.className='card ta-embed-card'; card.id='embedding-card';
    card.innerHTML=`
      <div class="card-header"><h3 class="card-title">Themes Map (2D LSI-PCA)</h3></div>
      <div class="card-body">
        <div class="muted" style="margin-bottom:.5rem; margin-left:22px; margin-top:.5rem">Each point is a speech; the color is the cluster. Click to open cluster.</div>
        <div class="chart-container" style="height: 380px"><canvas id="embedding-chart" height="320" aria-label="2D topic map"></canvas></div>
      </div>`;
    els.section.querySelector('.container').insertBefore(card, els.list);
  }

  async function loadEmbedding(){
    const canvas=document.getElementById('embedding-chart'); if(!canvas) return;
    try{
      const res=await fetch('/themes/embedding2d');
      const data=await res.json(); if(data.error) throw new Error(data.error);
      renderEmbedding(canvas, data);
    }catch(err){ console.warn('Embedding unavailable:', err); const c=document.getElementById('embedding-card'); if(c) c.style.display='none'; }
  }

  function renderEmbedding(canvas, payload){
    const ctx=canvas.getContext('2d');
    if(embedChart) embedChart.destroy();

    // Build datasets per cluster for distinct colors (Chart.js handles palette)
    const datasets = payload.series.map(s=>({
      label: 'C'+s.cluster_id,
      data: s.points.map(p=>({ x:p[0], y:p[1], _cid:s.cluster_id })),
      pointRadius: 2,
    }));

    embedChart=new Chart(ctx,{
      type:'scatter',
      data:{ datasets },
      options:{ responsive:true, maintainAspectRatio:false, plugins:{ legend:{ display:false }, tooltip:{ callbacks:{ label:(item)=>`C${item.raw._cid}: (${item.parsed.x.toFixed(2)}, ${item.parsed.y.toFixed(2)})` } } },
        onClick(_e, els){ if(!els?.length) return; const ds=els[0].datasetIndex; const cid=payload.series[ds]?.cluster_id; if (cid!=null) openCluster(cid,true); },
        scales:{ x:{ ticks:{ display:false } }, y:{ ticks:{ display:false } } }
      }
    });
  }

  // Init ------------------------------------------------------
  document.addEventListener('DOMContentLoaded', ()=>{ if(!els.section) return; loadOverview(); });
})();
