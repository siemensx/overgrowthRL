// OGRL-20260816-022/-023: dashboard front end. No build step, no
// dependencies. Polls the server at 1Hz while a run is selected and the tab
// is visible; stops polling when the tab is hidden.
//
// OGRL-20260817-028 Sec8: extended with three tabs (Training / Eval / Tapes)
// -- Training keeps everything that existed before, Eval reads
// runs/<id>/eval/*.json (evaluate.py's output), Tapes reads
// runs/<id>/tapes/ (tape.py's Tier-1 live ghost replay recordings). The 1Hz
// training poll deliberately does NOT re-render the Eval/Tapes tabs (their
// data isn't live-updating the same way, and a tape mid-playback would get
// its DOM torn out from under it by a full re-render -- see restartPolling).

const state = {
  runs: [],
  selectedRunId: null,
  manifest: null,
  metrics: [],
  episodes: [],
  events: [],
  rewardConfig: null,
  checkpoints: [],
  metricsOffset: 0,
  episodesOffset: 0,
  pollTimer: null,
  view: "training", // "training" | "eval" | "replays" | "tapes"

  // Replay panel (Training tab): renderTrainingView rebuilds this <select>
  // from scratch on every 1Hz poll tick (same as the rest of the tab), which
  // silently resets a bare <select> to its first alphabetical option every
  // time -- persisting the choice here and re-applying it in
  // renderReplayPanel is what makes a picked checkpoint actually stick.
  selectedCheckpoint: null,
  replayEpisodes: 2,

  // Eval tab
  evalList: [],
  selectedEvalStep: null,
  evalData: null,

  // Tapes tab
  tapesList: [],
  selectedTapeName: null,
  tapeMeta: null,
  tapeDecisions: [],
  tapePlaying: false,
  tapeIndex: 0,
  tapeSpeed: 1,
  tapePlayTimer: null,
  tapeAutoFollow: false,
  replaysList: [],
  selectedReplayName: null,
  checkpointCatalog: [],
  selectedMatchCheckpoint: null,
  activeMatch: null,
  matchPollTimer: null,
};

const COLORS = ["#5b9dff", "#3ecf8e", "#e8b339", "#e8583a", "#b06bd9", "#4fd1e8", "#f07ec1", "#9aa3b2", "#7ee787"];

async function fetchJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function escapeHTML(value) {
  return String(value ?? "").replace(/[&<>\"']/g, ch => ({"&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;"}[ch]));
}

function renderLiveMatch() {
  const root = document.getElementById("live-match");
  if (!root) return;
  const ready = state.checkpointCatalog.filter(item => item.status === "ready");
  const active = state.activeMatch && !["exited", "stopped", "error"].includes(state.activeMatch.phase);
  if (active) {
    const m = state.activeMatch;
    const result = m.result ? ` · ${m.result.replaceAll("_", " ")}` : "";
    const waiting = m.phase === "restart_wait" && m.restart_in !== undefined ? ` · restart ${fmtNumber(m.restart_in, 1)}s` : "";
    let panel = root.querySelector(".live-match");
    if (!panel || !panel.classList.contains("active-match")) {
      root.innerHTML = `<div class="live-match panel active-match">
        <div class="live-match-heading"><div><div class="eyebrow">LIVE ARENA</div><h2>Checkpoint match</h2></div><span class="live-dot">LIVE</span></div>
        <div class="match-score"><span>YOU <b>0</b></span><span class="versus">:</span><span>CHECKPOINT <b>0</b></span></div>
        <div class="match-status"></div>
        <button class="danger" id="stop-match">Stop match</button>
      </div>`;
      panel = root.querySelector(".live-match");
      document.getElementById("stop-match").onclick = () => stopMatch(m.match_id);
    }
    const score = panel.querySelector(".match-score");
    score.querySelector("span:nth-child(1) b").textContent = String(m.score ? m.score.you : 0);
    score.querySelector("span:nth-child(3) b").textContent = String(m.score ? m.score.checkpoint : 0);
    panel.querySelector(".match-status").textContent = `Round ${m.round || 1} · ${m.phase || "loading"}${result}${waiting}`;
    const stop = panel.querySelector("#stop-match");
    if (stop) stop.onclick = () => stopMatch(m.match_id);
    return;
  }
  const selected = state.selectedMatchCheckpoint && ready.some(item => item.id === state.selectedMatchCheckpoint) ? state.selectedMatchCheckpoint : (ready[0] && ready[0].id);
  let panel = root.querySelector(".live-match");
  if (!panel || panel.classList.contains("active-match")) {
    // Keep this subtree stable between 1Hz status polls. Replacing a native
    // <select> while its menu is open immediately closes that menu.
    root.innerHTML = `<div class="live-match panel">
      <div class="live-match-heading"><div><div class="eyebrow">LIVE ARENA</div><h2>Fight a checkpoint</h2></div><span class="match-kicker">120 Hz · 30 Hz policy</span></div>
      <div class="match-launch-row"><label for="checkpoint-select">Checkpoint</label><select id="checkpoint-select"></select><label for="level-select">Map</label><select id="level-select"></select><label for="policy-mode">Mode</label><select id="policy-mode"><option value="deterministic">Deterministic</option><option value="sampled">Sampled</option></select><button id="fight-checkpoint">Fight checkpoint</button></div>
      <div class="panel-caption" id="level-hint"></div>
      <div class="panel-caption match-overlay-hint">F8 toggles in-game diagnostics.</div>
      <div id="match-error" class="match-error"></div>
    </div>`;
    panel = root.querySelector(".live-match");
    const select = panel.querySelector("#checkpoint-select");
    select.addEventListener("change", () => { state.selectedMatchCheckpoint = select.value; });
    const levelSelect = panel.querySelector("#level-select");
    levelSelect.addEventListener("change", () => { state.selectedLevel = levelSelect.value; renderLevelHint(); });
    panel.querySelector("#fight-checkpoint").onclick = () => startMatch(select.value, panel.querySelector("#policy-mode").value, panel.querySelector("#fight-checkpoint"), levelSelect.value);
    populateLevels(levelSelect);
  }
  const select = panel.querySelector("#checkpoint-select");
  const catalogSignature = ready.map(item => `${item.id}:${item.global_step || 0}`).join("|");
  if (select.dataset.catalogSignature !== catalogSignature && document.activeElement !== select) {
    const current = select.value;
    select.replaceChildren();
    if (ready.length) {
      for (const item of ready) {
        const option = document.createElement("option");
        option.value = item.id;
        option.textContent = `${item.label} · step ${fmtNumber(item.global_step || 0, 0)}`;
        select.appendChild(option);
      }
      const next = ready.some(item => item.id === state.selectedMatchCheckpoint) ? state.selectedMatchCheckpoint : (ready.some(item => item.id === current) ? current : selected);
      select.value = next || ready[0].id;
      state.selectedMatchCheckpoint = select.value;
    } else {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No compatible checkpoints";
      select.appendChild(option);
    }
    select.disabled = ready.length === 0;
    panel.querySelector("#policy-mode").disabled = ready.length === 0;
    panel.querySelector("#fight-checkpoint").disabled = ready.length === 0;
    select.dataset.catalogSignature = catalogSignature;
  }
}

async function loadMatchState() {
  try {
    if (!state.checkpointCatalog.length) {
      const catalog = await fetchJSON("/api/checkpoint-catalog");
      state.checkpointCatalog = catalog.checkpoints || [];
    }
    const matches = await fetchJSON("/api/matches");
    const live = (matches.matches || []).filter(m => !["exited", "stopped", "error"].includes(m.phase));
    state.activeMatch = live.length ? live[live.length - 1] : null;
    renderLiveMatch();
  } catch (error) {
    const root = document.getElementById("live-match");
    if (root && !state.checkpointCatalog.length) root.innerHTML = `<div class="live-match panel"><div class="match-error">Server unavailable</div></div>`;
  }
}

function renderLevelHint() {
  const hint = document.getElementById("level-hint");
  if (!hint) return;
  const entry = (state.duelLevels || []).find(l => l.level === state.selectedLevel);
  hint.textContent = entry && entry.warn
    ? `F8 toggles in-game diagnostics. Note: ${entry.warn}.`
    : "F8 toggles in-game diagnostics.";
  hint.style.color = entry && entry.warn ? "var(--warn)" : "";
}

async function populateLevels(sel) {
  if (!state.duelLevels || !state.duelLevels.length) {
    try {
      const data = await fetchJSON("/api/duel-levels");
      state.duelLevels = data.levels || [];
    } catch (e) { state.duelLevels = []; }
  }
  sel.replaceChildren();
  if (!state.duelLevels.length) {
    const o = document.createElement("option");
    o.value = ""; o.textContent = "No duel maps found";
    sel.appendChild(o); sel.disabled = true; return;
  }
  for (const l of state.duelLevels) {
    const o = document.createElement("option");
    o.value = l.level;
    o.textContent = l.trained_on ? `${l.label} (trained)` : l.label;
    sel.appendChild(o);
  }
  // Default to a map the checkpoint actually trained on, not oval.
  const preferred = state.selectedLevel && state.duelLevels.some(l => l.level === state.selectedLevel)
    ? state.selectedLevel
    : (state.duelLevels.find(l => l.trained_on) || state.duelLevels[0]).level;
  sel.value = preferred;
  state.selectedLevel = preferred;
  renderLevelHint();
}

async function startMatch(checkpointId, policyMode, button, level) {
  const error = document.getElementById("match-error");
  if (error) error.textContent = "";
  state.selectedMatchCheckpoint = checkpointId;
  if (button) button.disabled = true;
  try {
    await fetchJSON("/api/matches", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({checkpoint_id: checkpointId, policy_mode: policyMode, level: level || state.selectedLevel})});
    await loadMatchState();
  } catch (err) {
    if (error) error.textContent = err.message;
    if (button) button.disabled = false;
  }
}

async function stopMatch(matchId) {
  await fetchJSON(`/api/matches/${encodeURIComponent(matchId)}`, {method: "DELETE"}).catch(() => {});
  await loadMatchState();
}

function fmtNumber(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "-";
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: 0 });
}

function fmtDuration(seconds) {
  if (!seconds || seconds < 0 || !Number.isFinite(seconds)) return "-";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

async function loadRunList() {
  const data = await fetchJSON("/api/runs");
  state.runs = data.runs;
  renderRunList();
}

function renderRunList() {
  const el = document.getElementById("run-list");
  el.innerHTML = "";
  for (const run of state.runs) {
    const row = document.createElement("div");
    row.className = "run-row" + (run.run_id === state.selectedRunId ? " selected" : "");
    const status = run.control_command === "pause" ? "paused" : (run.live ? "live" : (run.stale ? "stale" : (run.manifest.status || "completed")));
    const step = run.last_metric ? run.last_metric.global_step : (run.manifest.final_global_step || 0);
    const total = run.manifest.algo ? run.manifest.algo.total_timesteps : null;
    row.innerHTML = `
      <div class="run-id">${run.run_id}<span class="badge ${status}">${status}</span></div>
      <div class="run-meta">${total ? `${fmtNumber(step)} / ${fmtNumber(total)}` : fmtNumber(step)} decisions</div>
      <div class="run-meta">${run.manifest.purpose || ""}</div>
    `;
    row.onclick = () => selectRun(run.run_id);
    el.appendChild(row);
  }
}

async function selectRun(runId) {
  stopTapePlayback();
  state.selectedRunId = runId;
  state.metrics = [];
  state.episodes = [];
  state.events = [];
  state.metricsOffset = 0;
  state.episodesOffset = 0;
  state.view = "training";
  state.evalList = [];
  state.selectedEvalStep = null;
  state.evalData = null;
  state.tapesList = [];
  state.selectedTapeName = null;
  state.tapeMeta = null;
  state.tapeDecisions = [];
  state.tapeIndex = 0;
  state.tapeAutoFollow = false;
  state.replaysList = [];
  state.selectedReplayName = null;
  renderRunList();
  state.manifest = await fetchJSON(`/api/runs/${runId}`);
  // Reward config and checkpoint list are fixed for the life of a run (or at
  // least don't need 1Hz polling) -- fetch once per selection.
  state.rewardConfig = await fetchJSON(`/api/runs/${runId}/reward`).catch(() => null);
  state.checkpoints = (await fetchJSON(`/api/checkpoints`).catch(() => ({ checkpoints: [] }))).checkpoints;
  await pollRunData();
  renderMain();
  restartPolling();
}

async function pollRunData() {
  if (!state.selectedRunId) return;
  const runId = state.selectedRunId;
  const [metricsRes, episodesRes, eventsRes] = await Promise.all([
    fetchJSON(`/api/runs/${runId}/metrics?offset=${state.metricsOffset}`),
    fetchJSON(`/api/runs/${runId}/episodes?offset=${state.episodesOffset}`),
    fetchJSON(`/api/runs/${runId}/events`),
  ]);
  state.metrics.push(...metricsRes.lines);
  state.metricsOffset = metricsRes.offset;
  state.episodes.push(...episodesRes.lines);
  state.episodesOffset = episodesRes.offset;
  state.events = eventsRes.events;
}

function restartPolling() {
  if (state.pollTimer) clearInterval(state.pollTimer);
  state.pollTimer = setInterval(async () => {
    if (document.hidden || !state.selectedRunId) return;
    await loadRunList();
    await pollRunData();
    // Only the Training tab depends on this 1Hz stream -- re-rendering the
    // Eval tab does nothing useful (its data doesn't change under a poll),
    // and re-rendering the Tapes tab while a tape is mid-playback would tear
    // out the canvas/SVG nodes the playback loop is updating directly (see
    // updateTapePlayheadUI). Auto-follow is the one Tapes-tab thing that
    // DOES want this tick, handled separately below.
    if (state.view === "training") {
      renderMain();
    } else if (state.view === "tapes" && state.tapeAutoFollow) {
      await refreshTapesListAndFollow();
    }
  }, 1000);
}

document.addEventListener("visibilitychange", () => {
  // Polling itself already checks document.hidden each tick -- this listener
  // exists so resuming from a hidden tab doesn't wait a full interval.
  if (!document.hidden && state.selectedRunId) {
    pollRunData().then(() => { if (state.view === "training") renderMain(); });
  }
});

function winRateFiltered(episodes, filterFn, limit) {
  const filtered = episodes.filter(filterFn).slice(-limit);
  if (filtered.length === 0) return null;
  const won = filtered.filter(e => e.outcome === "won").length;
  const p = won / filtered.length;
  // Wilson score interval, 95%
  const n = filtered.length, z = 1.96;
  const denom = 1 + z * z / n;
  const center = (p + z * z / (2 * n)) / denom;
  const halfwidth = (z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom;
  return { p, n, lo: Math.max(0, center - halfwidth), hi: Math.min(1, center + halfwidth) };
}

function winRate(episodes) {
  return winRateFiltered(episodes, () => true, 100);
}

function renderMain() {
  const el = document.getElementById("content");
  if (!state.manifest) {
    el.innerHTML = `<div class="empty-state">Select a run on the left.</div>`;
    return;
  }
  el.innerHTML = "";
  el.appendChild(renderTabBar());
  if (state.view === "training") {
    renderTrainingView(el);
  } else if (state.view === "eval") {
    el.appendChild(renderEvalView());
  } else if (state.view === "tapes") {
    el.appendChild(renderTapesView());
  } else if (state.view === "replays") {
    el.appendChild(renderReplaysView());
  }
}

function renderTabBar() {
  const bar = document.createElement("div");
  bar.className = "tab-bar";
  const tabs = [["training", "Training"], ["eval", "Eval"], ["replays", "Replays"], ["tapes", "Legacy"]];
  for (const [key, label] of tabs) {
    const btn = document.createElement("button");
    btn.className = "tab-btn" + (state.view === key ? " active" : "");
    btn.textContent = label;
    btn.onclick = () => switchView(key);
    bar.appendChild(btn);
  }
  return bar;
}

async function switchView(view) {
  if (state.view === "tapes" && view !== "tapes") stopTapePlayback();
  state.view = view;
  if (view === "eval" && state.evalList.length === 0) {
    state.evalList = (await fetchJSON(`/api/runs/${state.selectedRunId}/eval`).catch(() => ({ evals: [] }))).evals;
    if (state.evalList.length > 0) {
      await selectEval(state.evalList[state.evalList.length - 1].global_step);
      return; // selectEval already re-renders
    }
  }
  if (view === "tapes" && state.tapesList.length === 0) {
    state.tapesList = (await fetchJSON(`/api/runs/${state.selectedRunId}/tapes`).catch(() => ({ tapes: [] }))).tapes;
  }
  if (view === "replays" && state.replaysList.length === 0) {
    state.replaysList = (await fetchJSON(`/api/runs/${state.selectedRunId}/replays`).catch(() => ({ replays: [] }))).replays;
  }
  renderMain();
}

// --- Training tab (everything that existed before, plus new panels) ---

function renderTrainingView(el) {
  const m = state.manifest;
  const metrics = state.metrics;
  const last = metrics[metrics.length - 1];
  const run = state.runs.find(r => r.run_id === state.selectedRunId);
  const live = run ? run.live : false;
  const paused = run ? run.control_command === "pause" : false;
  const entropyRef = m.entropy_random_reference || 6.9968;
  const wr = winRate(state.episodes);
  const scenario = last && last.curriculum_live ? last.curriculum_live.scenario : null;
  const topBandWr = scenario ? winRateFiltered(state.episodes, e => e.d !== null && e.d !== undefined && e.d >= scenario.d_max - 0.10001, 300) : null;

  el.appendChild(renderControls(live, paused));
  if (paused) {
    const banner = document.createElement("div");
    banner.className = "pause-banner";
    banner.innerHTML = `<span class="dot"></span> Paused -- the trainer is idle (0% CPU), engines are blocked waiting. Click Resume to continue.`;
    el.appendChild(banner);
  }
  el.appendChild(renderHero(m, last, entropyRef, wr, topBandWr, scenario, live, paused));
  el.appendChild(renderReplayPanel());
  el.appendChild(renderConfigPanel(m));
  el.appendChild(renderRewardPanel(state.rewardConfig));

  const entropyChart = renderChart("Entropy vs. random-policy reference", metrics,
    [{ key: d => d.ppo && d.ppo.entropy, label: "entropy", color: COLORS[0] }],
    { refLine: entropyRef, refLabel: "untrained-policy entropy",
      xMarker: 300000, xMarkerLabel: "300k gate (entropy < 6.0)", xMarkerColor: "#e8583a",
      markerFn: d => d.kl_spike === true, markerColor: "#e8583a" });
  const klNote = document.createElement("div");
  klNote.className = "panel-caption";
  klNote.textContent = "Red ticks at the top edge mark kl_spike updates (approx_kl > 10x target_kl that update).";
  entropyChart.appendChild(klNote);
  el.appendChild(entropyChart);

  el.appendChild(renderOutcomesChart(metrics));
  el.appendChild(renderComponentsChart(metrics));

  el.appendChild(renderCurriculumSection(metrics));
  el.appendChild(renderEmergencePanel(metrics));
  el.appendChild(renderActionStatsPanel(metrics));

  el.appendChild(renderPipelinePanel(metrics));

  const throughputChart = renderChart("Throughput (decisions/s)", metrics,
    [
      { key: d => d.perf && d.perf.steps_per_second_collection, label: "collection-only", color: COLORS[2] },
      { key: d => d.perf && d.perf.steps_per_second_cycle, label: "full cycle", color: COLORS[0] },
    ], {});
  el.appendChild(throughputChart);
  el.appendChild(renderResetPanel(metrics));

  el.appendChild(renderChart("Learning diagnostics: explained variance", metrics,
    [{ key: d => d.ppo && d.ppo.explained_variance, label: "explained_variance", color: COLORS[1] }], {}));
  el.appendChild(renderEventsPanel(state.events));
}

function renderControls(live, paused) {
  const div = document.createElement("div");
  div.className = "controls";
  const pauseBtn = document.createElement("button");
  pauseBtn.textContent = paused ? "Paused" : "Pause";
  pauseBtn.disabled = !live || paused;
  pauseBtn.onclick = () => sendControl("pause");
  const resumeBtn = document.createElement("button");
  resumeBtn.textContent = "Resume";
  resumeBtn.disabled = !live || !paused;
  resumeBtn.onclick = () => sendControl(null);
  const stopBtn = document.createElement("button");
  stopBtn.textContent = "Stop";
  stopBtn.className = "danger";
  stopBtn.disabled = !live;
  stopBtn.onclick = () => {
    if (confirm(`Stop ${state.selectedRunId}? This saves a final checkpoint and exits cleanly.`)) sendControl("stop");
  };
  div.append(pauseBtn, resumeBtn, stopBtn);
  return div;
}

async function sendControl(command) {
  await fetchJSON(`/api/runs/${state.selectedRunId}/control`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command }),
  });
  // The trainer only checks control.json once per PPO update -- an
  // immediate re-poll can legitimately still show the pre-command state.
  // Poll a few times over the next couple seconds so the UI catches up
  // without the user needing to click again to "confirm" it worked.
  for (let i = 0; i < 6; i++) {
    await new Promise(r => setTimeout(r, 400));
    await loadRunList();
    await pollRunData();
    renderMain();
    const run = state.runs.find(r => r.run_id === state.selectedRunId);
    const nowPaused = run && run.control_command === "pause";
    if (command === "pause" && nowPaused) break;
    if (command === null && run && !nowPaused) break;
    if (command === "stop" && run && !run.live) break;
  }
}

function renderHero(m, last, entropyRef, wr, topBandWr, scenario, live, paused) {
  const div = document.createElement("div");
  div.className = "hero-row";
  const algo = m.algo || {};
  const step = last ? last.global_step : (m.final_global_step || 0);
  const total = algo.total_timesteps || 0;
  const pct = total ? (100 * step / total).toFixed(1) : "-";

  const entropy = last && last.ppo ? last.ppo.entropy : null;
  let entropyClass = "";
  if (entropy !== null) {
    if (entropy >= entropyRef) entropyClass = "bad";
    else if (entropy >= entropyRef - 0.2) entropyClass = "warn";
    else entropyClass = "good";
  }

  const stepsPerSec = last && last.perf ? last.perf.steps_per_second_cycle : null;
  const resetShare = last && last.perf ? last.perf.reset_share : null;
  const remaining = total && step ? Math.max(0, total - step) : null;
  const etaSeconds = (live && !paused && stepsPerSec && remaining !== null) ? remaining / stepsPerSec : null;

  const tiles = [
    { label: "Progress", value: `${pct}%`, sub: `${fmtNumber(step)} / ${fmtNumber(total)} decisions` },
    { label: "Entropy", value: entropy !== null ? entropy.toFixed(4) : "-", sub: `reference (untrained): ${entropyRef.toFixed(4)}`, cls: entropyClass },
  ];
  if (scenario) {
    tiles.push({
      label: `Win rate (top band, d>=${(scenario.d_max - 0.10).toFixed(2)})`,
      value: topBandWr ? `${(topBandWr.p * 100).toFixed(0)}%` : "-",
      sub: topBandWr ? `95% CI [${(topBandWr.lo * 100).toFixed(0)}, ${(topBandWr.hi * 100).toFixed(0)}] n=${topBandWr.n}` : "no top-band episodes yet",
    });
    tiles.push({
      label: "d_max (scenario curriculum)",
      value: `${scenario.d_max.toFixed(2)} / ${scenario.d_max_cap.toFixed(2)}`,
      sub: `stage ${scenario.stage}, ${fmtNumber(scenario.episodes_recorded, 0)} episodes recorded`,
    });
  } else {
    tiles.push({ label: "Win rate (last 100 ep)", value: wr ? `${(wr.p * 100).toFixed(0)}%` : "-", sub: wr ? `95% CI [${(wr.lo * 100).toFixed(0)}, ${(wr.hi * 100).toFixed(0)}] n=${wr.n}` : "no episodes yet" });
  }
  tiles.push({
    label: "Throughput",
    value: stepsPerSec !== null ? fmtNumber(stepsPerSec) : "-",
    sub: `decisions/s, full cycle${resetShare !== null && resetShare !== undefined ? ` -- reset ${(resetShare * 100).toFixed(0)}% of cycle` : ""}`,
    cls: resetShare !== null && resetShare !== undefined && resetShare > 0.3 ? "warn" : "",
  });
  tiles.push({
    label: "ETA",
    value: etaSeconds !== null ? fmtDuration(etaSeconds) : "-",
    sub: live ? "at current throughput" : (paused ? "paused" : "not live"),
  });
  tiles.push({ label: "Status", value: paused ? "paused" : (live ? "live" : (m.status || "-")), sub: m.env ? `n_envs=${m.env.n_envs} act_period=${m.env.act_period}` : "", cls: paused ? "warn" : "" });

  for (const t of tiles) {
    const tile = document.createElement("div");
    tile.className = "tile" + (t.cls ? ` ${t.cls}` : "");
    tile.innerHTML = `<div class="label">${t.label}</div><div class="value">${t.value}</div><div class="sub">${t.sub}</div>`;
    div.appendChild(tile);
  }
  return div;
}

// Explicitly requested: show the EXACT current policy/config for the
// selected run, not just charts -- every hyperparameter/env/reward setting
// that manifest.json recorded at launch.
function renderConfigPanel(m) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>Exact policy config for this run</h2>`;
  const grid = document.createElement("div");
  grid.className = "config-grid";

  const groups = [
    ["Algorithm", m.algo || {}],
    ["Environment", m.env || {}],
    ["Reward", { profile: m.reward_profile }],
    ["Code", m.code || {}],
    ["Parent / resume", m.parent || {}],
  ];
  for (const [title, obj] of groups) {
    const keys = Object.keys(obj);
    if (keys.length === 0) continue;
    const group = document.createElement("div");
    group.className = "config-group";
    let rows = "";
    for (const k of keys) {
      let v = obj[k];
      if (v === null || v === undefined) v = "-";
      if (typeof v === "number") v = fmtNumber(v, 6);
      if (typeof v === "string" && v.length > 40) v = v.slice(0, 12) + "…" + v.slice(-20);
      rows += `<tr><td>${k}</td><td>${v}</td></tr>`;
    }
    group.innerHTML = `<h3>${title}</h3><table class="config-table">${rows}</table>`;
    grid.appendChild(group);
  }
  panel.appendChild(grid);
  return panel;
}

// Explicitly requested: "what the hell is reward profile 8" -- a bare
// profile-name string told the user nothing. This shows the ACTUAL numeric
// weight for every reward term this run uses, paired with the exact
// condition that triggers it, sign-coded green (positive) / red (negative).
function renderRewardPanel(rewardConfig) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>What the agent is actually rewarded/penalized for (profile: ${rewardConfig ? rewardConfig.profile : "?"})</h2>`;
  if (!rewardConfig || !rewardConfig.fields) {
    panel.innerHTML += `<div class="empty-state">Reward config unavailable.</div>`;
    return panel;
  }
  const rows = rewardConfig.fields.map(f => {
    const isOff = (f.name.includes("weight") && f.value === 0) || (f.sign === "n/a");
    const signClass = isOff ? "off" : (f.sign === "positive" ? "pos" : f.sign === "negative" ? "neg" : "");
    const signLabel = isOff ? "OFF" : (f.sign === "positive" ? "+" : f.sign === "negative" ? "−" : "");
    return `<tr class="reward-row ${signClass}">
      <td class="reward-sign">${signLabel}</td>
      <td class="reward-name">${f.name}</td>
      <td class="reward-value">${f.value}</td>
      <td class="reward-trigger">${f.trigger}</td>
    </tr>`;
  }).join("");
  panel.innerHTML += `<table class="reward-table">
    <thead><tr><th></th><th>field</th><th>value</th><th>triggers when...</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
  return panel;
}

function renderEventsPanel(events) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>Events</h2>`;
  const list = document.createElement("div");
  list.className = "events-list";
  if (events.length === 0) {
    list.innerHTML = `<div class="empty-state">No events yet.</div>`;
  }
  for (const ev of events.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "event-row";
    const t = new Date(ev.t * 1000).toLocaleTimeString();
    row.innerHTML = `<span class="event-kind">${ev.kind}</span>${ev.title}<span class="event-t">${t}</span>`;
    list.appendChild(row);
  }
  panel.appendChild(list);
  return panel;
}

function renderOutcomesChart(metrics) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>Outcomes per update (share of episodes)</h2>`;
  if (metrics.length === 0) {
    panel.innerHTML += `<div class="empty-state">No data yet.</div>`;
    return panel;
  }
  const width = Math.max(600, metrics.length * 7), height = 240, padL = 40, padR = 14, padT = 16, padB = 28;
  const plotH = height - padT - padB;
  const barGap = 1.5;
  const barW = Math.max(2, (width - padL - padR) / metrics.length - barGap);

  let gridlines = "";
  for (const frac of [0, 0.25, 0.5, 0.75, 1.0]) {
    const y = (height - padB - frac * plotH).toFixed(1);
    gridlines += `<line class="chart-gridline" x1="${padL}" y1="${y}" x2="${width - padR}" y2="${y}" />
      <text class="chart-label" x="${padL - 8}" y="${Number(y) + 3}" text-anchor="end">${(frac * 100).toFixed(0)}%</text>`;
  }

  let bars = "";
  metrics.forEach((d, i) => {
    const o = d.outcomes || { won: 0, lost: 0, timeout: 0 };
    const total = o.won + o.lost + o.timeout || 1;
    const x = padL + i * (width - padL - padR) / metrics.length;
    let y = height - padB;
    const segs = [["won", o.won, COLORS[1]], ["lost", o.lost, COLORS[3]], ["timeout", o.timeout, COLORS[7]]];
    for (const [, count, color] of segs) {
      if (count === 0) continue;
      const h = (count / total) * plotH;
      y -= h;
      bars += `<rect x="${x}" y="${y.toFixed(1)}" width="${barW.toFixed(1)}" height="${h.toFixed(1)}" fill="${color}" />`;
    }
  });
  panel.innerHTML += `
    <div class="chart-wrap"><svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      ${gridlines}
      <line class="chart-axis" x1="${padL}" y1="${height - padB}" x2="${width - padR}" y2="${height - padB}" />
      ${bars}
    </svg></div>
    <div class="legend">
      <div class="legend-item"><span class="legend-swatch" style="background:${COLORS[1]}"></span>won</div>
      <div class="legend-item"><span class="legend-swatch" style="background:${COLORS[3]}"></span>lost</div>
      <div class="legend-item"><span class="legend-swatch" style="background:${COLORS[7]}"></span>timeout</div>
    </div>`;
  return panel;
}

function renderComponentsChart(metrics) {
  if (metrics.length === 0 || !metrics.some(d => d.reward && d.reward.components)) {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<h2>Reward, decomposed (per-episode mean)</h2><div class="empty-state">No component data yet.</div>`;
    return panel;
  }
  const names = new Set();
  metrics.forEach(d => { if (d.reward && d.reward.components) Object.keys(d.reward.components).forEach(n => names.add(n)); });
  const nameList = Array.from(names).filter(n => metrics.some(d => d.reward && d.reward.components && d.reward.components[n]));
  const series = nameList.map((name, i) => ({
    key: d => d.reward && d.reward.components ? d.reward.components[name] : null,
    label: name, color: COLORS[i % COLORS.length],
  }));
  return renderChart("Reward, decomposed (per-episode mean)", metrics, series, {});
}

// --- Curriculum panel (OGRL-20260817-028 Sec3/Sec8): the environment-
// COMPOSITION curriculum (ScenarioSampler), distinct from the reward-shaping
// curriculum (closing_distance_weight/stall_penalty_weight, unchanged,
// still shown in the config panel above). ---

function renderCurriculumSection(metrics) {
  const frag = document.createDocumentFragment();
  const withScenario = metrics.filter(d => d.curriculum_live && d.curriculum_live.scenario);
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>Curriculum: environment composition (difficulty)</h2>`;
  if (withScenario.length === 0) {
    panel.innerHTML += `<div class="empty-state">No scenario-curriculum data yet (older-format run, or this run predates the ScenarioSampler).</div>`;
    frag.appendChild(panel);
    return frag;
  }
  const last = withScenario[withScenario.length - 1].curriculum_live.scenario;

  // Stalled = d_max hasn't moved across the last STALL_WINDOW updates AND
  // it's still below its cap -- the gate may be stuck, worth a human look.
  const STALL_WINDOW = 100;
  const recentTail = withScenario.slice(-STALL_WINDOW);
  const stalled = recentTail.length >= STALL_WINDOW && last.d_max < last.d_max_cap &&
                  recentTail.every(d => d.curriculum_live.scenario.d_max === last.d_max);

  let badges = `
    <div class="info-badges">
      <div class="info-badge">stage <b>${last.stage}</b></div>
      <div class="info-badge">opponents <b>${last.opponents}</b></div>
      <div class="info-badge">species_mode <b>${last.species_mode}</b></div>
      <div class="info-badge">weapons_prob <b>${fmtNumber(last.weapons_prob, 2)}</b></div>
      <div class="info-badge">episodes_recorded <b>${fmtNumber(last.episodes_recorded, 0)}</b></div>
      <div class="info-badge">d_mean_sampled <b>${last.d_mean_sampled !== null && last.d_mean_sampled !== undefined ? last.d_mean_sampled.toFixed(3) : "-"}</b></div>
    </div>`;
  if (stalled) {
    badges += `<div class="stall-banner"><span class="dot"></span> d_max has not advanced in the last ${STALL_WINDOW} updates and is still below its cap (${last.d_max_cap}) -- the advance gate may be stuck. Check the band win rates below.</div>`;
  }
  if (last.last_advance) {
    const [epIdx, oldD, newD] = last.last_advance;
    badges += `<div class="panel-caption">Last advance: at episode ${fmtNumber(epIdx, 0)}, d_max ${oldD.toFixed(2)} → ${newD.toFixed(2)}</div>`;
  }
  panel.innerHTML += badges;
  frag.appendChild(panel);

  frag.appendChild(renderChart("d_max over time (environment-composition curriculum)", withScenario,
    [
      { key: d => d.curriculum_live.scenario.d_max, label: "d_max", color: COLORS[0] },
      { key: d => d.curriculum_live.scenario.d_mean_sampled, label: "d_mean_sampled (actually-sampled difficulty)", color: COLORS[4] },
    ],
    { refLine: last.d_max_cap, refLabel: "d_max_cap" }));

  const bandPanel = document.createElement("div");
  bandPanel.className = "panel";
  bandPanel.innerHTML = `<h2>Win rate by fixed difficulty band (latest snapshot)</h2>`;
  const bandRow = document.createElement("div");
  bandRow.className = "band-bars";
  const bandEntries = Object.entries(last.band_win_rate || {});
  for (const [label, rate] of bandEntries) {
    const pct = rate !== null && rate !== undefined ? Math.round(rate * 100) : null;
    const bar = document.createElement("div");
    bar.className = "band-bar";
    bar.innerHTML = `
      <div class="band-bar-label">${label}</div>
      <div class="band-bar-track"><div class="band-bar-fill" style="width:${pct !== null ? pct : 0}%; background:${pct === null ? "var(--text-faint)" : COLORS[0]}"></div></div>
      <div class="band-bar-value">${pct !== null ? pct + "%" : "no data"}</div>`;
    bandRow.appendChild(bar);
  }
  bandPanel.appendChild(bandRow);
  frag.appendChild(bandPanel);

  // --- Opponent-count axis (OGRL-20260906-075) ---
  // The trainer has always logged curriculum_live.opponent_win_rates; nothing
  // displayed it, which is why a stalled difficulty gate went unnoticed for
  // 4575 episodes. Both curriculum axes are now visible, and the gate-health
  // banner below states outright when one has never advanced.
  const withOpp = metrics.filter(d => d.curriculum_live && d.curriculum_live.opponent_win_rates);
  if (withOpp.length) {
    const oppKeys = Object.keys(withOpp[withOpp.length - 1].curriculum_live.opponent_win_rates)
      .sort((a, b) => Number(a) - Number(b));
    frag.appendChild(renderChart("Win rate by opponent count (1v1 retention is the control)", withOpp,
      oppKeys.map((k, i) => ({
        key: d => {
          const v = d.curriculum_live.opponent_win_rates[k];
          return (v === null || v === undefined) ? null : v;
        },
        label: `1v${k}`,
        color: COLORS[i % COLORS.length],
      })), {}));

    const oppLast = withOpp[withOpp.length - 1].curriculum_live;
    const oppPanel = document.createElement("div");
    oppPanel.className = "panel";
    oppPanel.innerHTML = `<h2>Opponent curriculum (max = ${oppLast.opponents_max ?? "-"})</h2>`;
    const oppRow = document.createElement("div");
    oppRow.className = "band-bars";
    for (const k of oppKeys) {
      const rate = oppLast.opponent_win_rates[k];
      const pct = (rate !== null && rate !== undefined) ? Math.round(rate * 100) : null;
      const atMax = Number(k) === Number(oppLast.opponents_max);
      const bar = document.createElement("div");
      bar.className = "band-bar";
      bar.innerHTML = `
        <div class="band-bar-label">1v${k}${atMax ? " (at max)" : ""}</div>
        <div class="band-bar-track"><div class="band-bar-fill" style="width:${pct !== null ? pct : 0}%; background:${pct === null ? "var(--text-faint)" : COLORS[oppKeys.indexOf(k) % COLORS.length]}"></div></div>
        <div class="band-bar-value">${pct !== null ? pct + "%" : "no data"}</div>`;
      oppRow.appendChild(bar);
    }
    oppPanel.appendChild(oppRow);
    frag.appendChild(oppPanel);
  }

  // --- Gate health ---
  // A curriculum that never advances looks exactly like one that is merely
  // being careful. Say it explicitly: run21_mac sat at its d_max_start for
  // 4575 episodes because outnumbered fights were being counted toward a gate
  // meant to measure difficulty alone.
  const episodes = last.episodes_recorded || 0;
  const advanced = last.last_advance !== null && last.last_advance !== undefined;
  if (episodes > 800 && !advanced) {
    const warn = document.createElement("div");
    warn.className = "panel";
    warn.innerHTML = `<h2>Difficulty gate has never advanced</h2>
      <div class="empty-state">d_max is still at its starting value (${(last.d_max ?? 0).toFixed(2)})
      after ${episodes.toLocaleString()} episodes, and the mean sampled difficulty is
      ${(last.d_mean_sampled ?? 0).toFixed(3)}. The agent is training against opponents far weaker
      than the cap, and any win rate from this run is conditioned on that. Check that the gate is
      measuring its own axis -- outnumbered episodes must not count toward the DIFFICULTY gate
      (OGRL-20260906-075).</div>`;
    frag.insertBefore(warn, frag.firstChild);
  }

  return frag;
}

// --- Emergence panel (OGRL-20260817-028 Sec3.4/Sec8.3): "the panel to check
// in the morning" -- five behavior signatures, each (conditional prob minus
// unconditional prob), zero = no relationship. ---

const EMERGENCE_SIGNATURES = [
  { key: "parry", label: "Parry: P(grab | opp attacking) − P(grab)", sampleKey: "opp_attacking" },
  { key: "punish", label: "Punish: P(attack | opp hit-reaction) − P(attack)", sampleKey: "opp_hitreaction" },
  { key: "roll_recovery", label: "Roll recovery: P(crouch | self ragdoll) − P(crouch)", sampleKey: "self_ragdoll" },
  { key: "guard_pressure", label: "Guard pressure: P(atk | opp block broken) − P(atk | opp block healthy)", sampleKey: "opp_block_broken" },
  { key: "funnelling", label: "Funnelling: mean hostiles within 3m (steps with ≥2 hostiles visible)", sampleKey: "funnel_eligible" },
];

// Where a training cycle's wall time actually goes. Every field here has been
// logged since the vectorised trainer landed; nothing displayed it, so the
// split had to be recomputed by hand every time the question came up -- and a
// sampling-profiler guess at the same split was wrong by a factor of five
// (DEAD_ENDS.md, "sample percentages are NOT trustworthy").
//
// barrier_idle_seconds is a SUM ACROSS WORKERS, not a wall-clock fraction.
// Dividing it by cycle time without first dividing by active_workers produced a
// confident "58% of wall time is barrier idle" that was wrong by 3x. It is
// divided here so the panel cannot be misread the same way.
function renderPipelinePanel(metrics) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>Where cycle time goes (median of the last 60 updates)</h2>`;
  const rows = metrics.filter(d => d.perf && d.perf.cycle_seconds && d.perf.active_workers).slice(-60);
  if (rows.length < 5) {
    panel.innerHTML += `<div class="empty-state">Not enough perf samples yet.</div>`;
    return panel;
  }
  const med = (f) => {
    const v = rows.map(f).filter(x => typeof x === "number" && isFinite(x)).sort((a, b) => a - b);
    return v.length ? v[Math.floor(v.length / 2)] : 0;
  };
  const cycle = med(d => d.perf.cycle_seconds);
  const collect = med(d => d.perf.collection_seconds);
  const stepWall = med(d => d.perf.step_wall_seconds);
  const workers = med(d => d.perf.active_workers) || 1;
  const idlePer = med(d => d.perf.barrier_idle_seconds) / workers;
  const parts = [
    { label: "engine stepping (worker busy)", v: stepWall - idlePer, color: COLORS[0] },
    { label: "policy forward + buffering + rewards", v: collect - stepWall, color: COLORS[1] },
    { label: "barrier idle, per worker", v: idlePer, color: COLORS[3] },
    { label: "PPO update (the learner)", v: cycle - collect, color: COLORS[4] },
  ];
  const row = document.createElement("div");
  row.className = "band-bars";
  for (const p of parts) {
    const pct = cycle > 0 ? Math.max(0, Math.round((p.v / cycle) * 100)) : 0;
    const bar = document.createElement("div");
    bar.className = "band-bar";
    bar.innerHTML = `
      <div class="band-bar-label">${p.label}</div>
      <div class="band-bar-track"><div class="band-bar-fill" style="width:${pct}%; background:${p.color}"></div></div>
      <div class="band-bar-value">${pct}% &nbsp;<span style="opacity:.6">${p.v.toFixed(3)}s</span></div>`;
    row.appendChild(bar);
  }
  panel.appendChild(row);
  const note = document.createElement("div");
  note.className = "empty-state";
  note.style.marginTop = "8px";
  note.innerHTML = `cycle ${cycle.toFixed(3)}s across ${workers} active workers.
    Barrier idle is shown PER WORKER (barrier_idle_seconds is a sum across workers;
    dividing it by cycle time directly overstates it by the worker count).`;
  panel.appendChild(note);
  return panel;
}

function renderEmergencePanel(metrics) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>Emergence: strategy signatures ("the panel to check in the morning")</h2>
    <div class="panel-caption">Each line is (conditional probability − unconditional probability); zero = no relationship, a rising line = the behavior is being learned. A flat line at zero can mean "not happening" OR "not enough qualifying data yet" -- that's why each chart's sample count is printed underneath it.</div>`;
  const withData = metrics.filter(d => d.emergence);
  if (withData.length === 0) {
    panel.innerHTML += `<div class="empty-state">No emergence data yet (older-format run, or too early in training).</div>`;
    return panel;
  }
  const grid = document.createElement("div");
  grid.className = "mini-chart-grid";
  const last = withData[withData.length - 1].emergence;
  for (const sig of EMERGENCE_SIGNATURES) {
    const mini = renderMiniChart(grid, sig.label, withData,
      [{ key: d => d.emergence[sig.key], label: sig.key, color: COLORS[0] }],
      { refLine: 0, refLabel: "no relationship", height: 150, width: 440 });
    const n = last.samples ? last.samples[sig.sampleKey] : undefined;
    const note = document.createElement("div");
    note.className = "samples-note";
    note.textContent = n === undefined ? "" : `n=${fmtNumber(n, 0)} qualifying samples (latest update)`;
    mini.appendChild(note);
  }
  panel.appendChild(grid);
  return panel;
}

// --- Action statistics panel (OGRL-20260817-028 Sec8.2) ---

function renderActionStatsPanel(metrics) {
  const withData = metrics.filter(d => d.action_stats);
  if (withData.length === 0) {
    const panel = document.createElement("div");
    panel.className = "panel";
    panel.innerHTML = `<h2>Action statistics</h2><div class="empty-state">No action-stats data yet.</div>`;
    return panel;
  }
  const buttons = ["jump", "crouch", "attack", "grab", "drop", "walk"];
  const frag = document.createDocumentFragment();

  const meanChart = renderChart("Press probability -- mean, per button", withData,
    buttons.map((b, i) => ({ key: d => d.action_stats.press_prob[b], label: b, color: COLORS[i % COLORS.length] })), {});
  frag.appendChild(meanChart);

  const spreadChart = renderChart("Press probability -- SPREAD across envs (std of each env's own mean press rate)", withData,
    buttons.map((b, i) => ({ key: d => d.action_stats.press_prob_spread[b], label: b, color: COLORS[i % COLORS.length] })), {});
  const spreadNote = document.createElement("div");
  spreadNote.className = "panel-caption";
  spreadNote.textContent = "The mean alone can't tell a real controller apart from a coin flip -- only the spread across environments can (run5-9's own history: ~0.09 in the broken runs, ~0.35 once a controller emerged).";
  spreadChart.appendChild(spreadNote);
  frag.appendChild(spreadChart);

  frag.appendChild(renderChart("Continuous-move statistics", withData,
    [
      { key: d => d.action_stats.move_abs_mean, label: "move_abs_mean", color: COLORS[0] },
      { key: d => d.action_stats.move_std, label: "move_std", color: COLORS[1] },
      { key: d => d.action_stats.continuous_sigma, label: "continuous_sigma (policy's own exploration noise)", color: COLORS[2] },
    ], {}));

  return frag;
}

// --- Reset/pool overhead panel (OGRL-20260817-028 Sec8.2: "the single most
// important throughput number in this system") ---

function renderResetPanel(metrics) {
  const withData = metrics.filter(d => d.perf && d.perf.reset_seconds !== undefined && d.perf.reset_seconds !== null);
  const frag = document.createDocumentFragment();
  const shareChart = renderChart("Reset overhead: share of cycle time spent resetting", withData,
    [{ key: d => d.perf.reset_share, label: "reset_share", color: COLORS[3] }], { yMax: 1 });
  const last = withData[withData.length - 1];
  if (last) {
    const hits = last.perf.pool_hits || 0, misses = last.perf.pool_misses || 0;
    const total = hits + misses;
    const underrun = total > 0 ? (100 * misses / total).toFixed(1) : "-";
    const note = document.createElement("div");
    note.className = "panel-caption";
    note.textContent = `Env pool (latest update): ${fmtNumber(hits, 0)} hits / ${fmtNumber(misses, 0)} misses (${underrun}% underrun rate). A rising underrun rate means pre-warmed spare envs aren't keeping up with reset demand.`;
    shareChart.appendChild(note);
  }
  frag.appendChild(shareChart);
  frag.appendChild(renderChart("Reset time per cycle (seconds)", withData,
    [{ key: d => d.perf.reset_seconds, label: "reset_seconds", color: COLORS[2] }], {}));
  return frag;
}

// --- Replay launcher (fixed backend: watch.py now takes --from-run) ---

function renderReplayPanel() {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>Watch a checkpoint live</h2>
    <div class="panel-caption">Opens a real rendered engine window at 1x speed, using --from-run so it plays at this run's own level/frame_stack/act_period. This visibly slows a LIVE run down -- you'll be asked to confirm if this run is currently training.</div>`;
  const row = document.createElement("div");
  row.className = "replay-row";
  const select = document.createElement("select");
  if (state.checkpoints.length === 0) {
    const opt = document.createElement("option");
    opt.textContent = "no checkpoints found"; opt.disabled = true;
    select.appendChild(opt);
  }
  for (const c of state.checkpoints) {
    const opt = document.createElement("option");
    opt.value = c.path; opt.textContent = c.name;
    select.appendChild(opt);
  }
  // Restore the last pick (see state.selectedCheckpoint's comment) instead of
  // letting a freshly-built <select> silently fall back to its first
  // alphabetical option -- which, since run1.pt sorts before run10.pt, used
  // to mean "select run10" got reverted to run1 within one second every time.
  // First time through (nothing picked yet), default to the most RECENTLY
  // modified checkpoint rather than the alphabetically-first one -- that's
  // almost always the one you actually want to watch.
  if (state.checkpoints.length > 0) {
    const stillValid = state.checkpoints.some(c => c.path === state.selectedCheckpoint);
    if (!stillValid) {
      const newest = state.checkpoints.reduce((a, b) => (b.mtime > a.mtime ? b : a), state.checkpoints[0]);
      state.selectedCheckpoint = newest.path;
    }
    select.value = state.selectedCheckpoint;
  }
  select.onchange = () => { state.selectedCheckpoint = select.value; };
  const epInput = document.createElement("input");
  epInput.type = "number"; epInput.value = String(state.replayEpisodes); epInput.min = "1"; epInput.max = "20";
  epInput.onchange = () => { state.replayEpisodes = Number(epInput.value) || 2; };
  const btn = document.createElement("button");
  btn.textContent = "Watch live";
  btn.disabled = state.checkpoints.length === 0;
  btn.onclick = () => launchReplay(select.value, Number(epInput.value) || 2, false);
  row.append(select, epInput, btn);
  panel.appendChild(row);
  const status = document.createElement("div");
  status.className = "panel-caption";
  status.id = "replay-status";
  panel.appendChild(status);
  return panel;
}

async function launchReplay(checkpoint, episodes, force) {
  const statusEl = document.getElementById("replay-status");
  if (!checkpoint) { if (statusEl) statusEl.textContent = "No checkpoint selected."; return; }
  try {
    const res = await fetch("/api/replay", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "watch", checkpoint, episodes, run_id: state.selectedRunId, force: !!force }),
    });
    if (res.status === 409) {
      const body = await res.json().catch(() => ({}));
      if (confirm(`${body.error || "Training is currently live"}.\n\nLaunch the replay anyway? This will visibly slow the live run down.`)) {
        return launchReplay(checkpoint, episodes, true);
      }
      if (statusEl) statusEl.textContent = "Cancelled -- training is live.";
      return;
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      if (statusEl) statusEl.textContent = `Error: ${body.error || res.statusText}`;
      return;
    }
    const data = await res.json();
    if (statusEl) statusEl.textContent = `Launched job ${data.job_id} (status: ${data.status}).`;
  } catch (exc) {
    if (statusEl) statusEl.textContent = `Failed: ${exc.message}`;
  }
}

async function launchTapeReplay(tapeName, force) {
  const statusEl = document.getElementById("tape-replay-status");
  if (!tapeName) { if (statusEl) statusEl.textContent = "No tape selected."; return; }
  try {
    const res = await fetch("/api/replay", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "tape", run_id: state.selectedRunId, tape_name: tapeName, force: !!force }),
    });
    if (res.status === 409) {
      const body = await res.json().catch(() => ({}));
      if (confirm(`${body.error || "Training is currently live"}.\n\nReplay anyway? This will visibly slow the live run down.`)) {
        return launchTapeReplay(tapeName, true);
      }
      if (statusEl) statusEl.textContent = "Cancelled -- training is live.";
      return;
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      if (statusEl) statusEl.textContent = `Error: ${body.error || res.statusText}`;
      return;
    }
    const data = await res.json();
    if (statusEl) statusEl.textContent = `Launched job ${data.job_id} -- a new engine window should open shortly, `
      + `seeded to reproduce the original opponent.`;
  } catch (exc) {
    if (statusEl) statusEl.textContent = `Failed: ${exc.message}`;
  }
}

// --- Eval tab (OGRL-20260817-028 Sec6.1/Sec8) ---

function renderEvalView() {
  const container = document.createElement("div");
  const header = document.createElement("div");
  header.className = "panel";
  header.innerHTML = `<h2>Eval snapshots</h2>`;
  if (state.evalList.length === 0) {
    header.innerHTML += `<div class="empty-state">No eval results yet -- run "python3 Tools/rl/evaluate.py --from-run ${state.selectedRunId} --checkpoint &lt;path&gt;" to produce one.</div>`;
    container.appendChild(header);
    return container;
  }
  const list = document.createElement("div");
  list.className = "eval-snapshot-list";
  for (const e of state.evalList.slice().reverse()) {
    const row = document.createElement("div");
    row.className = "eval-snapshot-row" + (e.global_step === state.selectedEvalStep ? " selected" : "");
    const wr = e.overall && e.overall.win_rate !== null && e.overall.win_rate !== undefined ? `${(e.overall.win_rate * 100).toFixed(1)}%` : "-";
    row.innerHTML = `<span class="mono">step ${fmtNumber(e.global_step, 0)}</span>
      <span class="text-dim">${e.episodes || 0} ep/band × ${e.num_bands} bands, ${e.stochastic ? "stochastic" : "deterministic"}</span>
      <span class="eval-wr">overall win ${wr}</span>`;
    row.onclick = () => selectEval(e.global_step);
    list.appendChild(row);
  }
  header.appendChild(list);
  container.appendChild(header);

  if (state.evalData) {
    container.appendChild(renderEvalBandChart(state.evalData.bands || []));

    const overallPanel = document.createElement("div");
    overallPanel.className = "panel";
    const ov = state.evalData.overall || {};
    const ci = ov.win_rate_ci95 || [null, null];
    overallPanel.innerHTML = `<h2>Overall (pooled across bands -- sanity check only, see per-band chart above)</h2>
      <div class="hero-row">
        <div class="tile">
          <div class="label">Win rate</div>
          <div class="value">${ov.win_rate !== null && ov.win_rate !== undefined ? (ov.win_rate * 100).toFixed(1) + "%" : "-"}</div>
          <div class="sub">${ci[0] !== null && ci[0] !== undefined ? `95% CI [${(ci[0] * 100).toFixed(0)}, ${(ci[1] * 100).toFixed(0)}]` : ""}</div>
        </div>
        <div class="tile">
          <div class="label">Checkpoint</div>
          <div class="value" style="font-size:13px; word-break:break-all;">${(state.evalData.checkpoint || "-").split("/").pop()}</div>
          <div class="sub">global_step ${fmtNumber(state.evalData.global_step, 0)}</div>
        </div>
        <div class="tile">
          <div class="label">Env</div>
          <div class="value" style="font-size:14px;">${state.evalData.level || "-"}</div>
          <div class="sub">frame_stack=${state.evalData.frame_stack} act_period=${state.evalData.act_period}</div>
        </div>
      </div>`;
    container.appendChild(overallPanel);
  }
  return container;
}

async function selectEval(step) {
  state.selectedEvalStep = step;
  state.evalData = await fetchJSON(`/api/runs/${state.selectedRunId}/eval/${step}`).catch(() => null);
  renderMain();
}

function renderEvalBandChart(bands) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>Win rate by difficulty band (Wilson 95% CI)</h2>`;
  if (!bands || bands.length === 0) {
    panel.innerHTML += `<div class="empty-state">No band data in this eval snapshot.</div>`;
    return panel;
  }
  const width = 900, height = 280, padL = 50, padR = 14, padT = 20, padB = 44;
  const plotH = height - padT - padB;
  const n = bands.length;
  const slot = (width - padL - padR) / n;
  const yScale = p => height - padB - p * plotH;

  let gridlines = "";
  for (const frac of [0, 0.25, 0.5, 0.75, 1.0]) {
    const y = yScale(frac).toFixed(1);
    gridlines += `<line class="chart-gridline" x1="${padL}" y1="${y}" x2="${width - padR}" y2="${y}" />
      <text class="chart-label" x="${padL - 8}" y="${Number(y) + 3}" text-anchor="end">${(frac * 100).toFixed(0)}%</text>`;
  }

  function whisker(result, cx, dx, color) {
    if (!result) return "";
    const ci = result.win_rate_ci95 || [null, null];
    const [lo, hi] = ci;
    const x = cx + dx;
    let s = "";
    if (lo !== null && lo !== undefined && hi !== null && hi !== undefined) {
      const yLo = yScale(lo).toFixed(1), yHi = yScale(hi).toFixed(1);
      s += `<line x1="${x}" y1="${yLo}" x2="${x}" y2="${yHi}" stroke="${color}" stroke-width="1.5" />`;
      s += `<line x1="${x - 4}" y1="${yLo}" x2="${x + 4}" y2="${yLo}" stroke="${color}" stroke-width="1.5" />`;
      s += `<line x1="${x - 4}" y1="${yHi}" x2="${x + 4}" y2="${yHi}" stroke="${color}" stroke-width="1.5" />`;
    }
    if (result.win_rate !== null && result.win_rate !== undefined) {
      s += `<circle cx="${x}" cy="${yScale(result.win_rate).toFixed(1)}" r="4" fill="${color}" stroke="var(--panel)" stroke-width="1" />`;
    }
    return s;
  }

  let marks = "", xLabels = "";
  bands.forEach((b, i) => {
    const cx = padL + slot * (i + 0.5);
    const skill = b.normalized_skill !== null && b.normalized_skill !== undefined ? b.normalized_skill.toFixed(2) : "-";
    xLabels += `<text class="chart-label" x="${cx}" y="${height - 26}" text-anchor="middle">d=${b.band}</text>
      <text class="chart-title-label" x="${cx}" y="${height - 12}" text-anchor="middle">skill ${skill}</text>`;
    marks += whisker(b.policy, cx, -8, COLORS[0]);
    if (b.random_control) marks += whisker(b.random_control, cx, 8, COLORS[7]);
  });

  panel.innerHTML += `
    <div class="chart-wrap"><svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      ${gridlines}
      <line class="chart-axis" x1="${padL}" y1="${height - padB}" x2="${width - padR}" y2="${height - padB}" />
      ${xLabels}
      ${marks}
    </svg></div>
    <div class="legend">
      <div class="legend-item"><span class="legend-swatch" style="background:${COLORS[0]}"></span>policy</div>
      <div class="legend-item"><span class="legend-swatch" style="background:${COLORS[7]}"></span>random control</div>
    </div>`;

  let rows = `<tr><th>band (d)</th><th>policy win%</th><th>random win%</th><th>normalized skill</th><th>episodes</th></tr>`;
  for (const b of bands) {
    const p = b.policy, r = b.random_control;
    rows += `<tr>
      <td>${b.band}</td>
      <td>${p && p.win_rate !== null && p.win_rate !== undefined ? (p.win_rate * 100).toFixed(1) + "%" : "-"}</td>
      <td>${r && r.win_rate !== null && r.win_rate !== undefined ? (r.win_rate * 100).toFixed(1) + "%" : "-"}</td>
      <td>${b.normalized_skill !== undefined && b.normalized_skill !== null ? b.normalized_skill.toFixed(3) : "-"}</td>
      <td>${p ? p.episodes : "-"}</td>
    </tr>`;
  }
  const table = document.createElement("table");
  table.className = "eval-table";
  table.innerHTML = rows;
  panel.appendChild(table);
  return panel;
}

// --- Binary replay library (OGRL-20260820-044) ----------------------------

function replayStatusClass(status) {
  if (!status) return "recorded";
  if (status.startsWith("EXACT")) return "exact";
  if (status.startsWith("REFERENCE")) return "pixels";
  if (status.startsWith("DIVERGED")) return "diverged";
  if (status.startsWith("NATIVE")) return "native";
  if (status.startsWith("LEGACY")) return "legacy";
  return "recorded";
}

async function captureReplay(command, count = 1) {
  const status = document.getElementById("capture-status");
  try {
    await fetchJSON(`/api/runs/${state.selectedRunId}/replay-controls`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ command, count }),
    });
    if (status) status.textContent = command === "capture_next_loss" ? "Armed for the next loss." : "Capture armed for a future episode.";
  } catch (err) {
    if (status) status.textContent = err.message;
  }
}

async function launchNativeReplay(replayName, force = false) {
  const status = document.getElementById("native-replay-status");
  if (!replayName) return;
  try {
    const response = await fetch("/api/replay", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ kind: "native", run_id: state.selectedRunId, replay_name: replayName, force }),
    });
    const body = await response.json().catch(() => ({}));
    if (response.status === 409 && body.error && !force && confirm(`${body.error}\n\nLaunch the replay anyway?`)) {
      return launchNativeReplay(replayName, true);
    }
    if (!response.ok) {
      if (status) status.textContent = body.error || response.statusText;
      return;
    }
    if (status) status.textContent = `Engine opened · verifying ${body.job_id}`;
    const jobId = body.job_id;
    for (let attempt = 0; attempt < 180; attempt += 1) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      const job = await fetchJSON(`/api/replay/${jobId}`).catch(() => null);
      if (!job) continue;
      const result = job.result || {};
      if (result.verification === "exact_simulation_verified") {
        if (status) status.textContent = `EXACT SIMULATION VERIFIED · ${result.ticks} physics ticks`;
        return;
      }
      if (result.verification === "diverged") {
        if (status) status.textContent = `DIVERGED AT TICK ${result.tick} · engine window remains truthful but unverified`;
        return;
      }
      if (result.verification === "engine_report_missing") {
        if (status) status.textContent = "ENGINE COULD NOT VERIFY · replay process exited before its report";
        return;
      }
      if (job.status === "exited") {
        if (status) status.textContent = "Replay process exited · see the replay log for details";
        return;
      }
    }
    if (status) status.textContent = "Replay is still running; verification report will remain on disk.";
  } catch (err) {
    if (status) status.textContent = err.message;
  }
}

async function refreshReplaysList() {
  const res = await fetchJSON(`/api/runs/${state.selectedRunId}/replays`).catch(() => null);
  if (!res) return;
  state.replaysList = res.replays;
  if (state.view === "replays") renderMain();
}

function renderReplaysView() {
  const container = document.createElement("div");
  const controlPanel = document.createElement("div");
  controlPanel.className = "panel replay-command-deck";
  controlPanel.innerHTML = `<div class="eyebrow">FLIGHT RECORDER / V1</div><h2>Replay library</h2>
    <p class="panel-caption">Binary episodes are retained as recorded state until native 120 Hz proof hooks verify a simulation. Legacy JSONL remains segregated and unverifiable.</p>`;
  const controls = document.createElement("div");
  controls.className = "capture-controls";
  const next = document.createElement("button"); next.className = "button-primary"; next.textContent = "Capture next episode";
  next.onclick = () => captureReplay("capture_next");
  const loss = document.createElement("button"); loss.textContent = "Capture next loss";
  loss.onclick = () => captureReplay("capture_next_loss");
  const count = document.createElement("input"); count.type = "number"; count.min = "1"; count.max = "20"; count.value = "3"; count.title = "Number of future episodes";
  const batch = document.createElement("button"); batch.textContent = "Capture N";
  batch.onclick = () => captureReplay("capture_next_n", Math.max(1, Math.min(20, Number(count.value) || 1)));
  const refresh = document.createElement("button"); refresh.textContent = "Refresh library"; refresh.onclick = refreshReplaysList;
  const status = document.createElement("span"); status.id = "capture-status"; status.className = "panel-caption capture-status"; status.textContent = "Controls affect future episodes and the recent buffer.";
  controls.append(next, loss, count, batch, refresh, status);
  controlPanel.appendChild(controls);
  container.appendChild(controlPanel);

  const listPanel = document.createElement("div"); listPanel.className = "panel replay-library-panel";
  const sorted = state.replaysList.slice().sort((a, b) => (b.update || 0) - (a.update || 0));
  listPanel.innerHTML = `<div class="section-heading"><div><div class="eyebrow">RETAINED EVIDENCE</div><h2>Episodes <span class="count-pill">${sorted.length}</span></h2></div><div class="panel-caption">Select an episode to inspect policy, state, reward and terminal truth.</div></div>`;
  if (sorted.length === 0) {
    listPanel.innerHTML += `<div class="empty-state">No binary replays retained for this run yet.</div>`;
  } else {
    const list = document.createElement("div"); list.className = "replay-list";
    for (const replay of sorted) {
      const row = document.createElement("button");
      row.className = "replay-card" + (replay.name === state.selectedReplayName ? " selected" : "");
      row.innerHTML = `<div class="replay-card-top"><span class="replay-id">${replay.name}</span><span class="truth-badge ${replayStatusClass(replay.status)}">${replay.status || "RECORDED STATE PLAYBACK"}</span></div>
        <div class="replay-card-meta"><span>${replay.outcome || "unknown"}</span><span>update ${replay.update ?? "-"}</span><span>${replay.decision_count || replay.length_decisions || 0} decisions</span><span>${fmtNumber((replay.file_bytes || 0) / 1024, 1)} KB</span></div>`;
      row.onclick = () => selectReplay(replay.name);
      list.appendChild(row);
    }
    listPanel.appendChild(list);
  }
  container.appendChild(listPanel);
  if (state.selectedReplayName && state.tapeDecisions.length > 0) container.appendChild(renderTapePlayer());
  return container;
}

async function selectReplay(name) {
  stopTapePlayback();
  state.selectedReplayName = name;
  const data = await fetchJSON(`/api/runs/${state.selectedRunId}/replays/${name}`);
  state.selectedTapeName = name;
  state.tapeMeta = data;
  state.tapeDecisions = data.decisions || data.visual_states || [];
  state.tapeIndex = 0;
  renderMain();
}

// --- Legacy Tapes tab (OGRL-20260817-028): 2D top-down viewer ------------

async function refreshTapesListAndFollow() {
  const res = await fetchJSON(`/api/runs/${state.selectedRunId}/tapes`).catch(() => null);
  if (!res) return;
  state.tapesList = res.tapes;
  const sorted = state.tapesList.slice().sort((a, b) => b.update - a.update);
  if (sorted.length > 0 && sorted[0].name !== state.selectedTapeName) {
    await selectTape(sorted[0].name);
  } else if (state.view === "tapes") {
    renderMain(); // refresh the list (new tapes may have appeared even if the newest name is unchanged, e.g. ties)
  }
}

function renderTapesView() {
  const container = document.createElement("div");
  const listPanel = document.createElement("div");
  listPanel.className = "panel";
  listPanel.innerHTML = `<div class="section-heading"><div><div class="eyebrow">COMPATIBILITY ARCHIVE</div><h2>Legacy tapes</h2></div><span class="truth-badge legacy">LEGACY / UNVERIFIABLE</span></div>
    <p class="panel-caption">These JSONL summaries preserve the old viewer, but the original 120 Hz simulation state was never recorded.</p>`;
  if (state.tapesList.length === 0) {
    listPanel.innerHTML += `<div class="empty-state">No tapes recorded yet for this run.</div>`;
    container.appendChild(listPanel);
    return container;
  }

  const followBtn = document.createElement("button");
  followBtn.textContent = state.tapeAutoFollow ? "Auto-follow: ON" : "Auto-follow latest";
  if (state.tapeAutoFollow) followBtn.style.borderColor = "var(--accent)";
  followBtn.onclick = async () => {
    state.tapeAutoFollow = !state.tapeAutoFollow;
    if (state.tapeAutoFollow) await refreshTapesListAndFollow();
    else renderMain();
  };
  listPanel.appendChild(followBtn);

  const list = document.createElement("div");
  list.className = "tape-list";
  const sorted = state.tapesList.slice().sort((a, b) => b.update - a.update);
  for (const t of sorted) {
    const row = document.createElement("div");
    row.className = "tape-row" + (t.name === state.selectedTapeName ? " selected" : "");
    const tags = (t.reasons || []).map(r => `<span class="tape-tag ${r}">${r}</span>`).join("");
    row.innerHTML = `
      <div class="tape-row-main"><span class="mono">update ${t.update} · w${t.worker}</span> ${tags}<span class="truth-badge legacy">legacy</span></div>
      <div class="tape-row-sub">outcome <b class="${t.outcome}">${t.outcome}</b> · reward ${fmtNumber(t.reward_total, 2)} ·
        d=${t.difficulty !== null && t.difficulty !== undefined ? t.difficulty.toFixed(2) : "-"} · ${t.length_decisions} decisions</div>`;
    row.onclick = () => selectTape(t.name);
    list.appendChild(row);
  }
  listPanel.appendChild(list);
  container.appendChild(listPanel);

  if (state.tapeDecisions.length > 0) {
    container.appendChild(renderTapePlayer());
  }
  return container;
}

async function selectTape(name) {
  stopTapePlayback();
  state.selectedTapeName = name;
  state.tapeMeta = state.tapesList.find(t => t.name === name) || null;
  const text = await fetch(`/api/runs/${state.selectedRunId}/tapes/${name}`).then(r => r.text());
  state.tapeDecisions = text.split("\n").filter(l => l.trim()).map(l => JSON.parse(l));
  state.tapeIndex = 0;
  renderMain();
}

// Body-frame convention used below is not a guess -- it mirrors
// Source/Main/rl_observation.cpp's ToEgocentric()/MakeSelfFrame() exactly:
// frame.forward = flattened (yaw-only) facing, frame.right = (forward.z,
// -forward.x), and every rel/fwd field is (dot(v, right), v.y, dot(v,
// forward)). So in tape data, index 0 of a 2-tuple is the "right" axis and
// index 1 (or index 2 of a 3-tuple pos) is the "forward" axis, always
// relative to self's CURRENT facing that decision.
function worldToEgoXZ(dx, dz, selfFwd) {
  const fx = selfFwd[0], fz = selfFwd[1];
  return { x: dx * fz - dz * fx, z: dx * fx + dz * fz };
}

const STATE_COLORS = ["#5b9dff", "#9aa3b2", "#e8583a", "#e8b339", "#b06bd9"]; // movement, ground, attack, hit_reaction, ragdoll
const STATE_NAMES = ["movement", "ground", "attack", "hit_reaction", "ragdoll"];
const TAPE_SCALE = 24; // px per meter
const TAPE_TRAIL = 15; // decisions of fading trail

function drawBodyOnCanvas(ctx, x, y, dirX, dirY, fillColor, ringColor, hp, label) {
  const size = 11;
  ctx.save();
  ctx.translate(x, y);
  const angle = Math.atan2(dirX, -dirY);
  ctx.rotate(angle);
  ctx.fillStyle = fillColor;
  ctx.beginPath();
  ctx.moveTo(0, -size);
  ctx.lineTo(size * 0.65, size * 0.75);
  ctx.lineTo(-size * 0.65, size * 0.75);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
  ctx.save();
  ctx.translate(x, y);
  ctx.strokeStyle = ringColor;
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(0, 0, size + 4, 0, Math.PI * 2); ctx.stroke();
  ctx.restore();
  if (hp !== undefined && hp !== null) {
    const hpFrac = Math.max(0, Math.min(1, hp));
    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x, y, size + 9, 0, Math.PI * 2); ctx.stroke();
    ctx.strokeStyle = hpFrac > 0.5 ? "#3ecf8e" : hpFrac > 0.2 ? "#e8b339" : "#e8583a";
    ctx.beginPath(); ctx.arc(x, y, size + 9, -Math.PI / 2, -Math.PI / 2 + hpFrac * Math.PI * 2); ctx.stroke();
  }
  if (label) {
    ctx.fillStyle = "#9aa3b2"; ctx.font = "10px monospace"; ctx.textAlign = "center";
    ctx.fillText(label, x, y + size + 22);
  }
}

function drawTapeFrame(canvas, idx) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width, h = canvas.height;
  ctx.fillStyle = "#0b0d11";
  ctx.fillRect(0, 0, w, h);
  const decisions = state.tapeDecisions;
  const d = decisions[idx];
  if (!d) return;
  const cx = w / 2, cy = h / 2;

  ctx.strokeStyle = "rgba(255,255,255,0.07)";
  ctx.lineWidth = 1;
  for (let r = 3; r <= 15; r += 3) {
    ctx.beginPath(); ctx.arc(cx, cy, r * TAPE_SCALE, 0, Math.PI * 2); ctx.stroke();
  }

  // Self's own trail: reproject past ABSOLUTE self.pos into THIS frame's
  // egocentric view (self.pos is world-space per tape.py; self.fwd is also
  // world-space here, unlike entity fwd/rel which are already egocentric).
  for (let k = Math.max(0, idx - TAPE_TRAIL); k < idx; k++) {
    const pd = decisions[k];
    if (!pd || !pd.self) continue;
    const dx = pd.self.pos[0] - d.self.pos[0], dz = pd.self.pos[2] - d.self.pos[2];
    const ego = worldToEgoXZ(dx, dz, d.self.fwd);
    const age = (idx - k) / TAPE_TRAIL;
    ctx.fillStyle = `rgba(91,157,255,${(1 - age) * 0.5})`;
    ctx.beginPath(); ctx.arc(cx + ego.x * TAPE_SCALE, cy - ego.z * TAPE_SCALE, 2.5, 0, Math.PI * 2); ctx.fill();
  }
  // Opponent trails: approximate -- each historical frame's OWN egocentric
  // rel (not reprojected into the current frame, since we don't have their
  // absolute positions). Some jitter from self's own motion/turning between
  // decisions is expected; at 30Hz decision rate it reads as a trail, not
  // noise.
  for (let k = Math.max(0, idx - TAPE_TRAIL); k < idx; k++) {
    const pd = decisions[k];
    if (!pd) continue;
    const age = (idx - k) / TAPE_TRAIL;
    for (const e of (pd.ents || [])) {
      ctx.fillStyle = e.ally ? `rgba(91,157,255,${(1 - age) * 0.3})` : `rgba(232,88,58,${(1 - age) * 0.3})`;
      ctx.beginPath(); ctx.arc(cx + e.rel[0] * TAPE_SCALE, cy - e.rel[2] * TAPE_SCALE, 2, 0, Math.PI * 2); ctx.fill();
    }
  }

  for (const e of (d.ents || [])) {
    const x = cx + e.rel[0] * TAPE_SCALE, y = cy - e.rel[2] * TAPE_SCALE;
    const stateColor = STATE_COLORS[e.state] || "#9aa3b2";
    const ringColor = e.ally ? "#5b9dff" : "#e8583a"; // blue=ally, red=hostile
    drawBodyOnCanvas(ctx, x, y, e.fwd[0], -e.fwd[1], stateColor, ringColor, e.hp, `${e.ally ? "ally" : "hostile"} ${STATE_NAMES[e.state] || ""}`);
  }
  // self, always drawn at center pointing "up"
  drawBodyOnCanvas(ctx, cx, cy, 0, -1, STATE_COLORS[d.self.state] || "#5b9dff", "#e6e9ef", d.self.hp, "self " + (STATE_NAMES[d.self.state] || ""));

  ctx.fillStyle = "#5f6879"; ctx.font = "10px monospace"; ctx.textAlign = "left";
  ctx.fillText(`t=${fmtNumber(d.t, 2)}s  d=${d.d !== null && d.d !== undefined ? d.d.toFixed(2) : "-"}  rings @ 3m`, 8, h - 10);
  ctx.textAlign = "right";
  ctx.fillText("blue ring = ally, red ring = hostile, fill = state", w - 8, h - 10);
}

let tapePianoGeom = null;
let tapeRewardGeom = null;

function buildTapePianoRollSVG(decisions) {
  const laneNames = ["move_x", "move_y", "jump", "crouch", "attack", "grab", "drop", "walk"];
  const width = Math.max(600, decisions.length * 3), laneH = 22, padL = 74, padR = 10, padT = 4;
  const height = padT + laneNames.length * laneH + 4;
  const n = decisions.length;
  let svg = "";
  laneNames.forEach((name, li) => {
    const y0 = padT + li * laneH;
    svg += `<text class="chart-label" x="${padL - 8}" y="${y0 + laneH / 2 + 3}" text-anchor="end">${name}</text>`;
    svg += `<line class="chart-gridline" x1="${padL}" y1="${(y0 + laneH).toFixed(1)}" x2="${width - padR}" y2="${(y0 + laneH).toFixed(1)}" />`;
    if (li < 2) {
      const yMid = y0 + laneH / 2;
      let path = "";
      decisions.forEach((d, i) => {
        const x = padL + (n > 1 ? i / (n - 1) : 0) * (width - padL - padR);
        const v = Math.max(-1, Math.min(1, d.act[li] || 0));
        const yv = yMid - v * (laneH / 2 - 2);
        path += `${i === 0 ? "M" : "L"}${x.toFixed(1)},${yv.toFixed(1)} `;
      });
      svg += `<line x1="${padL}" y1="${yMid.toFixed(1)}" x2="${width - padR}" y2="${yMid.toFixed(1)}" stroke="var(--border)" stroke-width="1" stroke-dasharray="2 3" />`;
      svg += `<path d="${path}" fill="none" stroke="${COLORS[li]}" stroke-width="1.2" />`;
    } else {
      const barW = Math.max(1, (width - padL - padR) / n);
      let rects = "";
      decisions.forEach((d, i) => {
        if ((d.act[li] || 0) > 0.5) {
          const x = padL + i * (width - padL - padR) / n;
          rects += `<rect x="${x.toFixed(1)}" y="${y0 + 2}" width="${barW.toFixed(1)}" height="${laneH - 4}" fill="${COLORS[li % COLORS.length]}" opacity="0.85" />`;
        }
      });
      svg += rects;
    }
  });
  return { svg, width, height, padL, padR };
}

function drawTapePianoRoll() {
  const container = document.getElementById("tape-piano");
  if (!container) return;
  const built = buildTapePianoRollSVG(state.tapeDecisions);
  tapePianoGeom = built;
  container.innerHTML = `<div class="panel-caption" style="margin-bottom:6px;">Action piano-roll (8 lanes: 2 continuous move axes, 6 discrete buttons)</div>
    <div class="chart-wrap"><svg class="chart" viewBox="0 0 ${built.width} ${built.height}" preserveAspectRatio="none" style="height:${built.height}px;">
      ${built.svg}
      <line id="tape-piano-playhead" x1="0" y1="0" x2="0" y2="${built.height}" stroke="#e6e9ef" stroke-width="1.5" />
    </svg></div>`;
}

function buildTapeRewardStripSVG(decisions) {
  const width = Math.max(600, decisions.length * 3), height = 140, padL = 74, padR = 10, padT = 10, padB = 10;
  const plotH = height - padT - padB;
  const n = decisions.length;
  const names = new Set();
  decisions.forEach(d => Object.keys(d.rew || {}).forEach(k => names.add(k)));
  const nameList = Array.from(names);
  let maxAbs = 0;
  decisions.forEach(d => {
    let pos = 0, neg = 0;
    for (const k of nameList) { const v = (d.rew && d.rew[k]) || 0; if (v > 0) pos += v; else neg += v; }
    maxAbs = Math.max(maxAbs, pos, -neg);
  });
  maxAbs = maxAbs || 1;
  const yZero = padT + plotH / 2;
  const barW = Math.max(1, (width - padL - padR) / n);
  let bars = "";
  decisions.forEach((d, i) => {
    const x = padL + i * (width - padL - padR) / n;
    let posY = yZero, negY = yZero;
    nameList.forEach((name, ni) => {
      const v = (d.rew && d.rew[name]) || 0;
      const color = COLORS[ni % COLORS.length];
      if (v > 0) {
        const hgt = (v / maxAbs) * (plotH / 2);
        bars += `<rect x="${x.toFixed(1)}" y="${(posY - hgt).toFixed(1)}" width="${barW.toFixed(1)}" height="${hgt.toFixed(1)}" fill="${color}" />`;
        posY -= hgt;
      } else if (v < 0) {
        const hgt = (-v / maxAbs) * (plotH / 2);
        bars += `<rect x="${x.toFixed(1)}" y="${negY.toFixed(1)}" width="${barW.toFixed(1)}" height="${hgt.toFixed(1)}" fill="${color}" />`;
        negY += hgt;
      }
    });
  });
  const axisSvg = `<line class="chart-gridline" x1="${padL}" y1="${yZero.toFixed(1)}" x2="${width - padR}" y2="${yZero.toFixed(1)}" />
    <text class="chart-label" x="${padL - 8}" y="${(yZero + 3).toFixed(1)}" text-anchor="end">0</text>
    <text class="chart-label" x="${padL - 8}" y="${(padT + 8).toFixed(1)}" text-anchor="end">+${maxAbs.toFixed(2)}</text>
    <text class="chart-label" x="${padL - 8}" y="${(height - padB).toFixed(1)}" text-anchor="end">-${maxAbs.toFixed(2)}</text>`;
  return { svgBody: axisSvg + bars, width, height, padL, padR, nameList };
}

function drawTapeRewardStrip() {
  const container = document.getElementById("tape-reward-strip");
  if (!container) return;
  const built = buildTapeRewardStripSVG(state.tapeDecisions);
  tapeRewardGeom = built;
  container.innerHTML = `<div class="panel-caption" style="margin-bottom:6px;">Reward components (stacked, per decision)</div>
    <div class="chart-wrap"><svg class="chart" viewBox="0 0 ${built.width} ${built.height}" preserveAspectRatio="none" style="height:${built.height}px;">
      ${built.svgBody}
      <line id="tape-reward-playhead" x1="0" y1="0" x2="0" y2="${built.height}" stroke="#e6e9ef" stroke-width="1.5" />
    </svg></div>
    <div class="legend">${built.nameList.map((nm, i) => `<div class="legend-item"><span class="legend-swatch" style="background:${COLORS[i % COLORS.length]}"></span>${nm}</div>`).join("")}</div>`;
}

function renderTapePlayer() {
  const meta = state.tapeMeta || {};
  const binaryReplay = meta.container === "ogreplay";
  const decisionCount = meta.length_decisions || meta.decision_count || state.tapeDecisions.length;
  const panel = document.createElement("div");
  panel.className = "panel tape-player";
  const tags = (meta.reasons || []).map(r => `<span class="tape-tag ${r}">${r}</span>`).join(" ");
  panel.innerHTML = `<div class="replay-player-header"><div><div class="eyebrow">${binaryReplay ? "RECORDED STATE INSPECTOR" : "LEGACY DECISION VIEW"}</div><h2>update ${meta.update ?? "-"} · worker ${meta.worker ?? "-"} · <span class="${meta.outcome}">${meta.outcome || "unknown"}</span> · reward ${fmtNumber(meta.reward_total, 2)} · d=${meta.difficulty !== null && meta.difficulty !== undefined ? Number(meta.difficulty).toFixed(2) : "-"} ${tags}</h2></div><span class="truth-badge ${replayStatusClass(meta.status)}">${meta.status || "LEGACY / UNVERIFIABLE"}</span></div>`;

  const truthRow = document.createElement("div");
  truthRow.className = "replay-row truth-row";
  const truthText = document.createElement("span");
  truthText.className = "panel-caption";
  truthText.textContent = binaryReplay
    ? ((meta.native_tick_count || 0) > 0
      ? "The local engine will replay every recorded applied action at 120 Hz and verify its native state chain."
      : "Browser inspection only. Engine launch is unavailable for this recorded-state trace.")
    : "Legacy JSONL contains policy-decision summaries only and cannot verify the original simulation.";
  truthRow.appendChild(truthText);
  if (!binaryReplay) {
    const engineReplayBtn = document.createElement("button");
    engineReplayBtn.textContent = "Open legacy engine replay";
    engineReplayBtn.onclick = () => launchTapeReplay(state.selectedTapeName, false);
    truthRow.appendChild(engineReplayBtn);
  } else if ((meta.native_tick_count || 0) > 0) {
    const engineReplayBtn = document.createElement("button");
    engineReplayBtn.className = "button-primary";
    engineReplayBtn.textContent = "Launch engine replay · verify exactness";
    engineReplayBtn.onclick = () => launchNativeReplay(state.selectedReplayName);
    truthRow.appendChild(engineReplayBtn);
    const launchStatus = document.createElement("span");
    launchStatus.id = "native-replay-status";
    launchStatus.className = "panel-caption";
    launchStatus.textContent = "Ready: native tick trace present.";
    truthRow.appendChild(launchStatus);
  }
  panel.appendChild(truthRow);

  const canvasWrap = document.createElement("div");
  canvasWrap.className = "tape-canvas-wrap";
  const canvas = document.createElement("canvas");
  canvas.width = 560; canvas.height = 560;
  canvas.className = "tape-canvas";
  canvasWrap.appendChild(canvas);
  panel.appendChild(canvasWrap);

  const controls = document.createElement("div");
  controls.className = "tape-controls";
  const playBtn = document.createElement("button");
  playBtn.textContent = state.tapePlaying ? "Pause" : "Play";
  playBtn.onclick = () => {
    if (state.tapePlaying) { stopTapePlayback(); playBtn.textContent = "Play"; }
    else { startTapePlayback(); playBtn.textContent = "Pause"; }
  };
  const stepBackBtn = document.createElement("button");
  stepBackBtn.textContent = "◀ frame";
  stepBackBtn.onclick = () => { stopTapePlayback(); playBtn.textContent = "Play"; state.tapeIndex = Math.max(0, state.tapeIndex - 1); updateTapePlayheadUI(); };
  const stepFwdBtn = document.createElement("button");
  stepFwdBtn.textContent = "frame ▶";
  stepFwdBtn.onclick = () => { stopTapePlayback(); playBtn.textContent = "Play"; state.tapeIndex = Math.min(state.tapeDecisions.length - 1, state.tapeIndex + 1); updateTapePlayheadUI(); };
  const speedSelect = document.createElement("select");
  for (const s of [0.25, 0.5, 1, 2, 4]) {
    const opt = document.createElement("option");
    opt.value = String(s); opt.textContent = `${s}x`;
    if (s === state.tapeSpeed) opt.selected = true;
    speedSelect.appendChild(opt);
  }
  speedSelect.onchange = () => {
    state.tapeSpeed = Number(speedSelect.value);
    if (state.tapePlaying) { stopTapePlayback(); startTapePlayback(); }
  };
  const scrub = document.createElement("input");
  scrub.type = "range"; scrub.min = "0"; scrub.max = String(Math.max(0, state.tapeDecisions.length - 1)); scrub.value = String(state.tapeIndex);
  scrub.className = "tape-scrub";
  scrub.id = "tape-scrub";
  scrub.oninput = () => { stopTapePlayback(); playBtn.textContent = "Play"; state.tapeIndex = Number(scrub.value); updateTapePlayheadUI(); };
  const frameLabel = document.createElement("span");
  frameLabel.className = "tape-frame-label mono";
  frameLabel.id = "tape-frame-label";
  frameLabel.textContent = `${state.tapeIndex + 1} / ${decisionCount}`;
  controls.append(playBtn, stepBackBtn, stepFwdBtn, speedSelect, scrub, frameLabel);
  panel.appendChild(controls);

  const piano = document.createElement("div");
  piano.className = "tape-piano";
  piano.id = "tape-piano";
  panel.appendChild(piano);

  const rewardStrip = document.createElement("div");
  rewardStrip.className = "tape-reward-strip";
  rewardStrip.id = "tape-reward-strip";
  panel.appendChild(rewardStrip);

  // Mount happens after this element is attached to the DOM by the caller --
  // defer the draws one tick so getElementById can find the nodes above.
  requestAnimationFrame(() => {
    drawTapePianoRoll();
    drawTapeRewardStrip();
    drawTapeFrame(canvas, state.tapeIndex);
  });
  return panel;
}

function updateTapePlayheadUI() {
  const idx = state.tapeIndex;
  const n = state.tapeDecisions.length;
  const frac = n > 1 ? idx / (n - 1) : 0;
  if (tapePianoGeom) {
    const x = tapePianoGeom.padL + frac * (tapePianoGeom.width - tapePianoGeom.padL - tapePianoGeom.padR);
    const line = document.getElementById("tape-piano-playhead");
    if (line) { line.setAttribute("x1", x); line.setAttribute("x2", x); }
  }
  if (tapeRewardGeom) {
    const x = tapeRewardGeom.padL + frac * (tapeRewardGeom.width - tapeRewardGeom.padL - tapeRewardGeom.padR);
    const line = document.getElementById("tape-reward-playhead");
    if (line) { line.setAttribute("x1", x); line.setAttribute("x2", x); }
  }
  const scrub = document.getElementById("tape-scrub");
  if (scrub) scrub.value = String(idx);
  const label = document.getElementById("tape-frame-label");
  if (label) label.textContent = `${idx + 1} / ${n}`;
  const canvas = document.querySelector(".tape-canvas");
  if (canvas) drawTapeFrame(canvas, idx);
}

function startTapePlayback() {
  if (state.tapePlaying || state.tapeDecisions.length === 0) return;
  state.tapePlaying = true;
  const intervalMs = 33.333 / state.tapeSpeed; // 1x = 30Hz decision rate
  state.tapePlayTimer = setInterval(() => {
    state.tapeIndex++;
    if (state.tapeIndex >= state.tapeDecisions.length) {
      state.tapeIndex = state.tapeDecisions.length - 1;
      stopTapePlayback();
      const playBtn = document.querySelector(".tape-controls button");
      if (playBtn) playBtn.textContent = "Play";
      updateTapePlayheadUI();
      return;
    }
    updateTapePlayheadUI();
  }, intervalMs);
}

function stopTapePlayback() {
  if (state.tapePlayTimer) { clearInterval(state.tapePlayTimer); state.tapePlayTimer = null; }
  state.tapePlaying = false;
}

// --- Chart-drawing core (shared by full-width panels and the emergence
// grid's mini-charts) ---

function niceTicks(min, max, count) {
  const range = max - min || 1;
  const rough = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const norm = rough / mag;
  const step = (norm >= 5 ? 5 : norm >= 2 ? 2 : 1) * mag;
  const ticks = [];
  for (let v = Math.ceil(min / step) * step; v <= max + step * 1e-6; v += step) ticks.push(v);
  return ticks;
}

// Bucket-mean downsampling: a run at millions of decisions produces
// thousands of per-update points, and a chart is only a few hundred pixels
// wide -- found 2026-08-17 (user report: "graphs are so mushy") that
// run10's emergence mini-charts were plotting ~13 raw points PER PIXEL
// (5675 updates into a 440px chart), which is indistinguishable from solid
// noise no matter how real the underlying trend is. Bucketing by point
// INDEX (not by x-value) into a fixed number of buckets and averaging both
// x and y per bucket preserves the trend shape while keeping the point
// count something a chart line can actually resolve. Below maxPoints this
// is a no-op (returns pts unchanged).
function downsamplePoints(pts, maxPoints) {
  if (pts.length <= maxPoints) return pts;
  const bucketSize = pts.length / maxPoints;
  const out = [];
  for (let b = 0; b < maxPoints; b++) {
    const start = Math.floor(b * bucketSize), end = Math.floor((b + 1) * bucketSize);
    const slice = pts.slice(start, Math.max(start + 1, end));
    if (slice.length === 0) continue;
    let sx = 0, sy = 0;
    for (const [x, y] of slice) { sx += x; sy += y; }
    out.push([sx / slice.length, sy / slice.length]);
  }
  return out;
}

// Builds the inner chart markup (SVG + legend) and returns everything
// attachHoverInteraction needs. Both renderChart (full-width panel) and
// renderMiniChart (grid cell) call this -- one implementation of axis/tick/
// gridline/path math, not two.
function buildLineChart(metrics, series, opts) {
  const width = opts.width || 900, height = opts.height || 240, padL = 52, padR = 14, padT = 16, padB = 28;
  const xs = metrics.map(d => d.global_step);
  const xMin = Math.min(...xs), xMax = Math.max(...xs, xMin + 1);

  // ~1.5 raw points per pixel of plot width is about the density a line
  // chart can still resolve as a shape rather than a fuzzed band -- see
  // downsamplePoints's comment.
  const maxPoints = Math.max(50, Math.round((width - padL - padR) * 1.5));

  let allVals = [];
  let wasDownsampled = false;
  const seriesPoints = series.map(s => {
    const pts = metrics.map(d => [d.global_step, s.key(d)]).filter(([, v]) => v !== null && v !== undefined && !Number.isNaN(v));
    if (pts.length > maxPoints) wasDownsampled = true;
    const downsampled = downsamplePoints(pts, maxPoints);
    allVals.push(...downsampled.map(p => p[1]));
    return { ...s, pts: downsampled };
  });
  if (opts.refLine !== undefined) allVals.push(opts.refLine);
  let yMin = Math.min(...allVals, 0), yMax = Math.max(...allVals, 1);
  if (opts.yMin !== undefined) yMin = Math.min(yMin, opts.yMin);
  if (opts.yMax !== undefined) yMax = Math.max(yMax, opts.yMax);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const yPad = (yMax - yMin) * 0.1;
  yMin -= yPad; yMax += yPad;

  const xScale = x => padL + (width - padL - padR) * (x - xMin) / Math.max(1, xMax - xMin);
  const yScale = y => height - padB - (height - padT - padB) * (y - yMin) / (yMax - yMin);

  // Recessive horizontal gridlines at "nice" values, with a value label on
  // each -- this is the single biggest lever against a chart reading as
  // amateurish: bare axes force the eye to guess intermediate values.
  const yTicks = niceTicks(yMin, yMax, 4);
  let gridlines = "";
  for (const t of yTicks) {
    const y = yScale(t).toFixed(1);
    gridlines += `<line class="chart-gridline" x1="${padL}" y1="${y}" x2="${width - padR}" y2="${y}" />
      <text class="chart-label" x="${padL - 8}" y="${Number(y) + 3}" text-anchor="end">${t.toFixed(Math.abs(t) < 10 ? 2 : 0)}</text>`;
  }
  const xTicks = niceTicks(xMin, xMax, 5);
  let xTickLabels = "";
  for (const t of xTicks) {
    if (t < xMin || t > xMax) continue;
    const x = xScale(t).toFixed(1);
    xTickLabels += `<text class="chart-label" x="${x}" y="${height - 8}" text-anchor="middle">${fmtNumber(t, 0)}</text>`;
  }

  let paths = "";
  for (const s of seriesPoints) {
    if (s.pts.length === 0) continue;
    const d = s.pts.map(([x, y], i) => `${i === 0 ? "M" : "L"}${xScale(x).toFixed(1)},${yScale(y).toFixed(1)}`).join(" ");
    paths += `<path class="chart-line" d="${d}" stroke="${s.color}" />`;
  }
  let refLineSvg = "";
  if (opts.refLine !== undefined) {
    const y = yScale(opts.refLine).toFixed(1);
    refLineSvg = `<path class="chart-ref-line" d="M${padL},${y} L${width - padR},${y}" stroke="${opts.refLineColor || "#9aa3b2"}" />
      <text class="chart-title-label" x="${width - padR}" y="${Number(y) - 6}" text-anchor="end">${opts.refLabel || "reference"} (${opts.refLine.toFixed(4)})</text>`;
  }
  let markerSvg = "";
  if (opts.xMarker !== undefined && opts.xMarker >= xMin && opts.xMarker <= xMax) {
    const x = xScale(opts.xMarker).toFixed(1);
    markerSvg = `<path class="chart-ref-line" d="M${x},${padT} L${x},${height - padB}" stroke="${opts.xMarkerColor || "#e8583a"}" />
      <text class="chart-title-label" x="${x}" y="${padT + 10}" text-anchor="middle" fill="${opts.xMarkerColor || "#e8583a"}">${opts.xMarkerLabel || ""}</text>`;
  }
  // Per-point marker ticks (e.g. kl_spike flags) -- a short vertical tick at
  // the top edge for every metric row where opts.markerFn(d) is true.
  let pointMarkersSvg = "";
  if (opts.markerFn) {
    for (const d of metrics) {
      if (opts.markerFn(d)) {
        const x = xScale(d.global_step).toFixed(1);
        pointMarkersSvg += `<path class="chart-spike-marker" d="M${x},${padT} L${x},${padT + 8}" stroke="${opts.markerColor || "#e8583a"}" />`;
      }
    }
  }

  const html = `
    <div class="chart-wrap">
      <svg class="chart" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
        ${gridlines}
        <line class="chart-axis" x1="${padL}" y1="${height - padB}" x2="${width - padR}" y2="${height - padB}" />
        ${xTickLabels}
        ${refLineSvg}
        ${markerSvg}
        ${pointMarkersSvg}
        ${paths}
        <line class="hover-crosshair" x1="0" y1="${padT}" x2="0" y2="${height - padB}" style="display:none" />
        <g class="hover-dots"></g>
        <rect class="hover-capture" x="${padL}" y="${padT}" width="${width - padL - padR}" height="${height - padT - padB}" fill="transparent" />
      </svg>
      <div class="chart-tooltip" style="display:none"></div>
    </div>
    <div class="legend">${series.map(s => `<div class="legend-item"><span class="legend-swatch" style="background:${s.color}"></span>${s.label}</div>`).join("")}</div>
  `;
  return { html, seriesPoints, xScale, yScale, padL, width, padR, wasDownsampled };
}

function renderChart(title, metrics, series, opts) {
  const panel = document.createElement("div");
  panel.className = "panel";
  panel.innerHTML = `<h2>${title}</h2>`;
  if (metrics.length === 0) {
    panel.innerHTML += `<div class="empty-state">No data yet.</div>`;
    return panel;
  }
  const built = buildLineChart(metrics, series, opts);
  panel.innerHTML += built.html;
  attachHoverInteraction(panel, built.seriesPoints, built.xScale, built.yScale, built.padL, built.width, built.padR, built.wasDownsampled);
  if (built.wasDownsampled) {
    const note = document.createElement("div");
    note.className = "panel-caption";
    note.textContent = `Downsampled for readability (${metrics.length} updates plotted as bucket averages) -- hover values are per-bucket means, not single updates.`;
    panel.appendChild(note);
  }
  return panel;
}

// Lighter-weight variant for grid cells (emergence panel): no full .panel
// wrapper/h2, just an h3 + chart, appended straight into `container`.
function renderMiniChart(container, title, metrics, series, opts) {
  const wrap = document.createElement("div");
  wrap.className = "mini-chart";
  wrap.innerHTML = `<h3>${title}</h3>`;
  if (metrics.length === 0) {
    wrap.innerHTML += `<div class="empty-state">No data yet.</div>`;
    container.appendChild(wrap);
    return wrap;
  }
  const built = buildLineChart(metrics, series, opts);
  wrap.innerHTML += built.html;
  attachHoverInteraction(wrap, built.seriesPoints, built.xScale, built.yScale, built.padL, built.width, built.padR, built.wasDownsampled);
  container.appendChild(wrap);
  return wrap;
}

// Crosshair + tooltip on hover -- shows the exact value at the cursor's x
// position for every series in the chart. seriesPoints are already in DATA
// space; xScale/yScale map to the SVG viewBox, and the SVG's own CTM maps
// viewBox space to screen pixels, so mouse events (screen space) go through
// both to find the nearest data point.
function attachHoverInteraction(panel, seriesPoints, xScale, yScale, padL, width, padR, wasDownsampled) {
  const svg = panel.querySelector("svg.chart");
  const capture = panel.querySelector(".hover-capture");
  const crosshair = panel.querySelector(".hover-crosshair");
  const dotsGroup = panel.querySelector(".hover-dots");
  const tooltip = panel.querySelector(".chart-tooltip");
  if (!svg || !capture) return;

  const allX = Array.from(new Set(seriesPoints.flatMap(s => s.pts.map(p => p[0])))).sort((a, b) => a - b);
  if (allX.length === 0) return;

  function handleMove(evt) {
    const rect = svg.getBoundingClientRect();
    const svgX = ((evt.clientX - rect.left) / rect.width) * (width);
    // xScale is a plain closure (no inverse) -- find the nearest actual data
    // point by running every known x through the same forward scale instead
    // of inverting it.
    let nearest = allX[0], nearestDist = Infinity;
    for (const x of allX) {
      const d = Math.abs(xScale(x) - svgX);
      if (d < nearestDist) { nearest = x; nearestDist = d; }
    }
    const crossX = xScale(nearest);
    crosshair.setAttribute("x1", crossX); crosshair.setAttribute("x2", crossX);
    crosshair.style.display = "";

    let dots = "";
    let rows = "";
    for (const s of seriesPoints) {
      const pt = s.pts.find(p => p[0] === nearest);
      if (!pt) continue;
      const cy = yScale(pt[1]);
      dots += `<circle cx="${crossX}" cy="${cy}" r="3.5" fill="${s.color}" stroke="var(--panel)" stroke-width="1.5" />`;
      rows += `<div class="tooltip-row"><span class="tooltip-swatch" style="background:${s.color}"></span>${s.label}: <b>${fmtNumber(pt[1], 4)}</b></div>`;
    }
    dotsGroup.innerHTML = dots;
    const xLabel = wasDownsampled ? `~step ${fmtNumber(nearest, 0)} (bucket avg)` : `step ${fmtNumber(nearest, 0)}`;
    tooltip.innerHTML = `<div class="tooltip-x">${xLabel}</div>${rows}`;
    tooltip.style.display = "";
    const tipX = Math.min(rect.width - 160, Math.max(0, (evt.clientX - rect.left) + 12));
    const tipY = Math.max(0, (evt.clientY - rect.top) - 10);
    tooltip.style.left = `${tipX}px`;
    tooltip.style.top = `${tipY}px`;
  }
  function handleLeave() {
    crosshair.style.display = "none";
    dotsGroup.innerHTML = "";
    tooltip.style.display = "none";
  }
  capture.addEventListener("mousemove", handleMove);
  capture.addEventListener("mouseleave", handleLeave);
}

loadRunList();
loadMatchState();
state.matchPollTimer = setInterval(() => { if (!document.hidden) loadMatchState(); }, 1000);
setInterval(() => { if (!document.hidden) loadRunList(); }, 3000);
