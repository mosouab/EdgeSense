// Dashboard frontend: connects to /ws, drives Chart.js, sends control commands.

const MAX_POINTS = 240;
const PRIMARY_CHANNELS = ["TP2", "Oil_temperature", "Motor_current", "Reservoirs"];

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
};

const sensorCharts = [];
let scoreChart = null;
let ws = null;
let primaryChannels = PRIMARY_CHANNELS.slice();

const palette = {
  text: "#e6edf6",
  muted: "#8593a8",
  grid: "#1c2741",
  accent: "#4cc9f0",
  threshold: "#f3722c",
};

function makeLineChart(canvas, label, color) {
  return new Chart(canvas, {
    type: "line",
    data: { labels: [], datasets: [{ label, data: [], borderColor: color, backgroundColor: color, borderWidth: 1.4, pointRadius: 0, tension: 0.15 }] },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: { ticks: { color: palette.muted, font: { size: 10 } }, grid: { color: palette.grid } },
      },
      plugins: {
        legend: { display: true, position: "top", align: "start", labels: { color: palette.text, font: { size: 11 }, boxWidth: 8 } },
      },
    },
  });
}

function initSensorCharts() {
  sensorCharts.length = 0;
  for (let i = 0; i < 4; i++) {
    const canvas = document.getElementById(`sensor-${i}`);
    const channel = primaryChannels[i] ?? "—";
    sensorCharts.push(makeLineChart(canvas, channel, palette.accent));
  }
  els.sensorLabels.textContent = primaryChannels.slice(0, 4).join(" · ");
}

function initScoreChart() {
  const ctx = document.getElementById("score-chart");
  scoreChart = new Chart(ctx, {
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
}

function pushPoint(chart, x, y) {
  chart.data.labels.push(x);
  chart.data.datasets[0].data.push(y);
  if (chart.data.labels.length > MAX_POINTS) {
    chart.data.labels.shift();
    chart.data.datasets[0].data.shift();
  }
}

function pushScorePoint(x, score, threshold) {
  scoreChart.data.labels.push(x);
  scoreChart.data.datasets[0].data.push(score);
  scoreChart.data.datasets[1].data.push(threshold ?? null);
  if (scoreChart.data.labels.length > MAX_POINTS) {
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
  // Arc length of the semicircle path is ~282
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
  els.alertText.textContent = (
    level === "alert" ? "ALERT — anomaly above threshold" :
    level === "warn" ? "Watch — score elevated" :
    "All clear"
  );
}

function updatePhase(name, progress, detail) {
  els.phaseName.textContent = name;
  els.phaseDetail.textContent = detail ?? "";
  els.phasePill.dataset.phase = name;
  els.progressBar.style.width = `${Math.max(0, Math.min(1, progress ?? 0)) * 100}%`;
  els.phaseMeta.textContent = name;
}

function handleEvent(ev) {
  if (ev.kind === "phase") {
    updatePhase(ev.phase, ev.progress, ev.detail);
    if (ev.phase === "inferring" && ev.detail?.startsWith("threshold")) {
      // Reset score chart so threshold line appears cleanly
    }
    return;
  }
  if (ev.kind === "reading") {
    const time = ev.elapsed_simulated_seconds ?? Date.now();
    primaryChannels.slice(0, 4).forEach((name, i) => {
      const value = ev.features?.[name];
      if (value !== undefined) pushPoint(sensorCharts[i], time, value);
      sensorCharts[i].update("none");
    });

    if (ev.score !== null && ev.score !== undefined) {
      pushScorePoint(time, ev.score, ev.threshold);
      scoreChart.update("none");
    }
    updateHealth(ev.health ?? null);
    updateAlert(ev.alert_level ?? "ok");
    els.scoreMeta.textContent = ev.score !== null && ev.score !== undefined ? ev.score.toFixed(3) : "—";
    els.thresholdMeta.textContent = ev.threshold !== null && ev.threshold !== undefined ? ev.threshold.toFixed(3) : "—";
    els.timeMeta.textContent = formatSimTime(ev.elapsed_simulated_seconds ?? 0);
    if (ev.phase) updatePhase(ev.phase, undefined, undefined); // keep phase pill in sync without resetting detail
    els.healthDetail.textContent = ev.phase === "calibrating" ? "calibration in progress" :
                                   ev.phase === "training" ? "training model …" :
                                   ev.phase === "inferring" ? "monitoring live" :
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
  if (ws) ws.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (msg) => {
    try {
      handleEvent(JSON.parse(msg.data));
    } catch (err) {
      console.error("Bad event", err, msg.data);
    }
  };
  ws.onclose = () => setTimeout(openSocket, 1000);
}

async function api(path, payload) {
  const opts = payload === undefined ? { method: "POST" }
                                     : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) };
  const r = await fetch(path, opts);
  return r.json();
}

async function loadSources() {
  const r = await fetch("/sources");
  const data = await r.json();
  els.sourceSelect.innerHTML = "";
  data.sources.forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s.name;
    opt.textContent = s.display_name + (s.available === "false" ? " (coming soon)" : "");
    if (s.available === "false") opt.disabled = true;
    els.sourceSelect.appendChild(opt);
  });
}

els.btnStart.addEventListener("click", async () => {
  initSensorCharts();
  initScoreChart();
  await api("/start", {
    source: els.sourceSelect.value,
    speed: Number(els.speedSelect.value),
    calibration_samples: Number(els.calibInput.value),
  });
  els.srcName.textContent = els.sourceSelect.selectedOptions[0]?.textContent ?? "";
});
els.btnPause.addEventListener("click", () => api("/pause"));
els.btnStop.addEventListener("click", () => api("/stop"));
els.speedSelect.addEventListener("change", () => api("/speed", { speed: Number(els.speedSelect.value) }));

initSensorCharts();
initScoreChart();
loadSources();
openSocket();
