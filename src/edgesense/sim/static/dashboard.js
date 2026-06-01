// Dashboard frontend: connects to /ws, drives Chart.js, sends control commands.

const MAX_POINTS = 240;
const PRIMARY_CHANNELS_DEFAULTS = {
  metropt: ["TP2", "Oil_temperature", "Motor_current", "Reservoirs"],
  hydraulic: ["PS1", "TS1", "EPS1", "CE"],
  cmapss: ["sensor_2", "sensor_3", "sensor_4", "sensor_7"],
};
const PRIMARY_CHANNELS = PRIMARY_CHANNELS_DEFAULTS.metropt;

const els = {
  sourceSelect: document.getElementById("source-select"),
  calibInput: document.getElementById("calib-input"),
  speedSelect: document.getElementById("speed-select"),
  btnStart: document.getElementById("btn-start"),
  btnPause: document.getElementById("btn-pause"),
  btnStop: document.getElementById("btn-stop"),
  phasePill: document.getElementById("phase-pill"),
  phaseName: document.getElementById("phase-name"),
  phaseDetail: document.getElementById("phase-detail"),
  progressBar: document.getElementById("progress-bar"),
  wsStatus: document.getElementById("ws-status"),
  sensorLabels: document.getElementById("sensor-labels"),
  healthValue: document.getElementById("health-value"),
  healthDetail: document.getElementById("health-detail"),
  healthArc: document.getElementById("health-arc"),
  alertPill: document.getElementById("alert-pill"),
  alertText: document.getElementById("alert-text"),
  srcName: document.getElementById("src-name"),
  phaseMeta: document.getElementById("phase-meta"),
  scoreMeta: document.getElementById("score-meta"),
  thresholdMeta: document.getElementById("threshold-meta"),
  timeMeta: document.getElementById("time-meta"),
  failuresList: document.getElementById("failures-list"),
  failuresHint: document.getElementById("failures-hint"),
  rulBlock: document.getElementById("rul-block"),
  rulPredPrimary: document.getElementById("rul-pred-primary"),
  rulPredUnit: document.getElementById("rul-pred-unit"),
  rulPredSecondary: document.getElementById("rul-pred-secondary"),
  rulTruePrimary: document.getElementById("rul-true-primary"),
  rulTrueUnit: document.getElementById("rul-true-unit"),
  rulTrueSecondary: document.getElementById("rul-true-secondary"),
  rulUnitInfo: document.getElementById("rul-unit-info"),
};

let currentPhase = "idle";

const palette = {
  text: "#e6edf6",
  muted: "#8593a8",
  grid: "#1c2741",
  accent: "#4cc9f0",
  threshold: "#f3722c",
};

const sensorCharts = [];
let scoreChart = null;
let ws = null;
let primaryChannels = PRIMARY_CHANNELS.slice();

function setStatus(state, text) {
  els.wsStatus.dataset.state = state;
  els.wsStatus.textContent = text;
}

function makeLineChart(canvas, label, color) {
  return new Chart(canvas, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label,
          data: [],
          borderColor: color,
          backgroundColor: color,
          borderWidth: 1.4,
          pointRadius: 0,
          tension: 0.15,
        },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: { ticks: { color: palette.muted, font: { size: 10 } }, grid: { color: palette.grid } },
      },
      plugins: {
        legend: {
          display: true,
          position: "top",
          align: "start",
          labels: { color: palette.text, font: { size: 11 }, boxWidth: 8 },
        },
      },
    },
  });
}

function initCharts() {
  if (typeof Chart === "undefined") {
    setStatus("disconnected", "Chart.js failed to load");
    console.error("Chart.js is undefined — CDN may have failed.");
    return;
  }
  // Destroy any pre-existing chart on each canvas (defensive against re-init).
  for (let i = 0; i < 4; i++) {
    const canvas = document.getElementById(`sensor-${i}`);
    const existing = Chart.getChart?.(canvas);
    if (existing) existing.destroy();
    sensorCharts[i] = makeLineChart(canvas, primaryChannels[i] ?? "—", palette.accent);
  }
  const scoreCanvas = document.getElementById("score-chart");
  const existing = Chart.getChart?.(scoreCanvas);
  if (existing) existing.destroy();
  scoreChart = new Chart(scoreCanvas, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Score (smoothed)", data: [], borderColor: palette.accent, borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
        { label: "Threshold", data: [], borderColor: palette.threshold, borderWidth: 1.2, borderDash: [6, 4], pointRadius: 0 },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: { ticks: { color: palette.muted }, grid: { color: palette.grid } },
      },
      plugins: { legend: { display: false } },
    },
  });
  els.sensorLabels.textContent = primaryChannels.slice(0, 4).join(" · ");
}

function resetChartData() {
  sensorCharts.forEach((c) => {
    if (!c) return;
    c.data.labels.length = 0;
    c.data.datasets[0].data.length = 0;
    c.update("none");
  });
  if (scoreChart) {
    scoreChart.data.labels.length = 0;
    scoreChart.data.datasets[0].data.length = 0;
    scoreChart.data.datasets[1].data.length = 0;
    scoreChart.update("none");
  }
}

function pushSensor(i, x, y) {
  const chart = sensorCharts[i];
  if (!chart) return;
  chart.data.labels.push(x);
  chart.data.datasets[0].data.push(y);
  while (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
}

function pushScorePoint(x, score, threshold) {
  if (!scoreChart) return;
  scoreChart.data.labels.push(x);
  scoreChart.data.datasets[0].data.push(score);
  scoreChart.data.datasets[1].data.push(threshold ?? null);
  while (scoreChart.data.labels.length > MAX_POINTS) {
    scoreChart.data.labels.shift();
    scoreChart.data.datasets[0].data.shift();
    scoreChart.data.datasets[1].data.shift();
  }
}

function updateHealth(value) {
  if (value === null || value === undefined) {
    els.healthValue.textContent = "—";
    els.healthArc.setAttribute("stroke", palette.muted);
    return;
  }
  const v = Math.max(0, Math.min(100, value));
  els.healthValue.textContent = v.toFixed(0);
  const total = 282;
  const dash = (v / 100) * total;
  els.healthArc.setAttribute("stroke-dasharray", `${dash} ${total - dash}`);
  let color = "#06d6a0";
  if (v < 70) color = "#f7b500";
  if (v < 30) color = "#ef476f";
  els.healthArc.setAttribute("stroke", color);
}

function updateAlert(level) {
  els.alertPill.className = `alert-pill ${level}`;
  els.alertText.textContent =
    level === "alert" ? "ALERT — anomaly above threshold" :
    level === "warn"  ? "Watch — score elevated" :
                        "All clear";
}

function updatePhase(name, progress, detail) {
  if (name) {
    els.phaseName.textContent = name;
    els.phasePill.dataset.phase = name;
    els.phaseMeta.textContent = name;
    if (name !== currentPhase) {
      currentPhase = name;
      refreshFailureButtons();
    }
  }
  if (detail !== undefined && detail !== null) els.phaseDetail.textContent = detail;
  if (typeof progress === "number") {
    els.progressBar.style.width = `${Math.max(0, Math.min(1, progress)) * 100}%`;
  }
}

function refreshFailureButtons() {
  const enabled = currentPhase === "inferring";
  els.failuresHint.textContent = enabled
    ? "click a row to jump 10 minutes before the labeled failure starts"
    : `jumps available once the model is inferring (now: ${currentPhase})`;
  els.failuresList.querySelectorAll(".btn-jump").forEach((b) => (b.disabled = !enabled));
}

async function loadFailures(sourceName) {
  try {
    const path = sourceName ? `/failures?source=${encodeURIComponent(sourceName)}` : "/failures";
    const r = await fetch(path);
    const data = await r.json();
    els.failuresList.innerHTML = "";
    (data.failures || []).forEach((f) => {
      const li = document.createElement("li");
      const left = document.createElement("span");
      left.className = "label";
      const tag = document.createElement("span");
      tag.className = "source-tag" + (f.source === "audit" ? " audit" : "");
      tag.textContent = f.source === "audit" ? "audit" : "logged";
      left.textContent = f.label + " ";
      left.appendChild(tag);
      const btn = document.createElement("button");
      btn.className = "btn-jump";
      btn.textContent = "▶ jump";
      btn.dataset.failureId = String(f.id);
      btn.addEventListener("click", () => jumpToFailure(f.id, f.label));
      li.appendChild(left);
      li.appendChild(btn);
      els.failuresList.appendChild(li);
    });
    refreshFailureButtons();
  } catch (err) {
    console.error("Failed to load failures", err);
  }
}

async function jumpToFailure(failureId, label) {
  const res = await api("/jump", { failure_id: failureId });
  if (res.status === "ok") {
    console.log("Jumped to failure", failureId, label, res.index);
  } else if (res.status === "rejected") {
    setStatus("disconnected", res.detail || "jump rejected");
    setTimeout(() => setStatus("connected", "connected"), 1500);
  }
}

function renderRul(rulCycles, rulHours, cycleLabel, primaryEl, unitEl, secondaryEl) {
  if (rulCycles === null || rulCycles === undefined) {
    primaryEl.textContent = "—";
    unitEl.textContent = "";
    secondaryEl.textContent = "";
    return;
  }
  const cycles = Number(rulCycles);
  const cycleSuffix = Math.abs(cycles - 1) < 1e-6 ? cycleLabel : `${cycleLabel}s`;
  if (rulHours === null || rulHours === undefined) {
    primaryEl.textContent = cycles.toFixed(0);
    unitEl.textContent = ` ${cycleSuffix}`;
    secondaryEl.textContent = "";
    return;
  }
  const hours = Number(rulHours);
  if (hours < 24) {
    primaryEl.textContent = hours.toFixed(1);
    unitEl.textContent = " hours";
  } else if (hours < 60 * 24) {
    primaryEl.textContent = (hours / 24).toFixed(1);
    unitEl.textContent = " days";
  } else {
    primaryEl.textContent = (hours / (24 * 30)).toFixed(1);
    unitEl.textContent = " months";
  }
  secondaryEl.textContent = `≈ ${cycles.toFixed(0)} ${cycleSuffix}`;
}

let throttleCounter = 0;
function handleEvent(ev) {
  if (ev.kind === "phase") {
    updatePhase(ev.phase, ev.progress, ev.detail);
    return;
  }
  if (ev.kind === "reading") {
    const time = ev.elapsed_simulated_seconds ?? 0;

    // Push every event into the data buffers; redraw at most every 4th tick
    // so a flood of 60+/sec events doesn't pin the main thread.
    primaryChannels.slice(0, 4).forEach((name, i) => {
      const value = ev.features?.[name];
      if (value !== undefined) pushSensor(i, time, value);
    });
    if (ev.score !== null && ev.score !== undefined) {
      pushScorePoint(time, ev.score, ev.threshold);
    }

    throttleCounter = (throttleCounter + 1) % 4;
    if (throttleCounter === 0) {
      sensorCharts.forEach((c) => c && c.update("none"));
      scoreChart && scoreChart.update("none");
    }

    updateHealth(ev.health ?? null);
    updateAlert(ev.alert_level ?? "ok");
    if (els.rulBlock && !els.rulBlock.hidden) {
      const cycleLabel = ev.cycle_label || "cycle";
      renderRul(
        ev.rul_pred,
        ev.rul_pred_hours,
        cycleLabel,
        els.rulPredPrimary,
        els.rulPredUnit,
        els.rulPredSecondary,
      );
      renderRul(
        ev.true_rul,
        ev.rul_true_hours,
        cycleLabel,
        els.rulTruePrimary,
        els.rulTrueUnit,
        els.rulTrueSecondary,
      );
      if (ev.unit_id !== undefined) {
        els.rulUnitInfo.textContent =
          `unit ${ev.unit_id} · cycle ${ev.unit_cycle ?? "?"}`;
      }
    }
    els.scoreMeta.textContent =
      ev.score !== null && ev.score !== undefined ? ev.score.toFixed(3) : "—";
    els.thresholdMeta.textContent =
      ev.threshold !== null && ev.threshold !== undefined ? ev.threshold.toFixed(3) : "—";
    els.timeMeta.textContent = formatSimTime(ev.elapsed_simulated_seconds ?? 0);
    if (ev.phase) updatePhase(ev.phase, undefined, undefined);
    els.healthDetail.textContent =
      ev.phase === "calibrating" ? "calibration in progress" :
      ev.phase === "training"    ? "training model …" :
      ev.phase === "inferring"   ? "monitoring live" :
                                   "awaiting calibration";
  }
}

function formatSimTime(seconds) {
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const parts = [];
  if (days) parts.push(`${days}d`);
  if (hours || days) parts.push(`${hours}h`);
  parts.push(`${minutes}m`);
  return parts.join(" ");
}

function openSocket() {
  if (ws) {
    try { ws.close(); } catch (_) {}
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  setStatus("connecting", "connecting…");
  ws.onopen = () => setStatus("connected", "connected");
  ws.onerror = (e) => {
    console.error("WebSocket error", e);
    setStatus("disconnected", "ws error");
  };
  ws.onclose = () => {
    setStatus("disconnected", "disconnected — retrying");
    setTimeout(openSocket, 1000);
  };
  ws.onmessage = (msg) => {
    try { handleEvent(JSON.parse(msg.data)); }
    catch (err) { console.error("Bad event", err, msg.data); }
  };
}

async function api(path, payload) {
  const opts = payload === undefined
    ? { method: "POST" }
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
  const r = await fetch(path, opts);
  if (!r.ok) {
    console.error(`${path} failed`, r.status);
    setStatus("disconnected", `${path} ${r.status}`);
  }
  return r.json().catch(() => ({}));
}

let sourceCatalog = [];

async function loadSources() {
  try {
    const r = await fetch("/sources");
    const data = await r.json();
    sourceCatalog = data.sources || [];
    els.sourceSelect.innerHTML = "";
    sourceCatalog.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.name;
      opt.textContent = s.display_name + (s.available === "false" ? " (coming soon)" : "");
      if (s.available === "false") opt.disabled = true;
      els.sourceSelect.appendChild(opt);
    });
    // Apply defaults for the initial selection.
    applySourceDefaults(els.sourceSelect.value);
  } catch (err) {
    console.error("Failed to load sources", err);
  }
}

function applySourceDefaults(sourceName) {
  const spec = sourceCatalog.find((s) => s.name === sourceName);
  if (spec) {
    els.calibInput.value = spec.suggested_calibration;
    const unit = spec.natural_unit || "samples";
    const label = els.calibInput.parentElement.querySelector("span");
    if (label) label.textContent = `Calibration ${unit}`;
    // Show the RUL panel only for sources that produce RUL predictions.
    if (els.rulBlock) {
      els.rulBlock.hidden = !(spec.output_kind && spec.output_kind.includes("rul"));
    }
  }
  primaryChannels = (PRIMARY_CHANNELS_DEFAULTS[sourceName] || PRIMARY_CHANNELS).slice();
  initCharts();
  loadFailures(sourceName);
}

els.btnStart.addEventListener("click", async () => {
  resetChartData();
  const result = await api("/start", {
    source: els.sourceSelect.value,
    speed: Number(els.speedSelect.value),
    calibration_samples: Number(els.calibInput.value),
  });
  console.log("/start ->", result);
  els.srcName.textContent = els.sourceSelect.selectedOptions[0]?.textContent ?? "";
});
els.btnPause.addEventListener("click", () => api("/pause"));
els.btnStop.addEventListener("click", () => api("/stop"));
els.speedSelect.addEventListener("change", () =>
  api("/speed", { speed: Number(els.speedSelect.value) })
);
els.sourceSelect.addEventListener("change", () => applySourceDefaults(els.sourceSelect.value));

window.addEventListener("DOMContentLoaded", () => {
  initCharts();
  loadSources();
  // loadFailures() is invoked via applySourceDefaults once sources are loaded.
  openSocket();
});
