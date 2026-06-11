// EdgeSense Operator Console — Vue 3 app.
// All UI; no application logic. REST + WebSocket contract is identical
// to the previous build.

import {
  createApp,
  reactive,
  ref,
  computed,
  onMounted,
  onUnmounted,
  watch,
  nextTick,
} from "vue";

/* ─────────────────────────────────────────────────────────
   API client
   ───────────────────────────────────────────────────────── */

async function jsonRequest(path, opts = {}) {
  const init = { ...opts };
  if (init.body && typeof init.body !== "string") {
    init.headers = { "Content-Type": "application/json", ...(init.headers || {}) };
    init.body = JSON.stringify(init.body);
  }
  const r = await fetch(path, init);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json().catch(() => ({}));
}

const api = {
  sources: () => jsonRequest("/sources"),
  failures: (source) =>
    jsonRequest(`/failures${source ? `?source=${encodeURIComponent(source)}` : ""}`),
  start: (body) => jsonRequest("/start", { method: "POST", body }),
  stop: () => jsonRequest("/stop", { method: "POST" }),
  pause: () => jsonRequest("/pause", { method: "POST" }),
  setSpeed: (speed) => jsonRequest("/speed", { method: "POST", body: { speed } }),
  jump: (failureId) =>
    jsonRequest("/jump", { method: "POST", body: { failure_id: failureId } }),
  feedback: (body) => jsonRequest("/feedback", { method: "POST", body }),
  recalibrate: () => jsonRequest("/recalibrate", { method: "POST" }),
  revert: () => jsonRequest("/revert", { method: "POST" }),
};

/* ─────────────────────────────────────────────────────────
   Constants & helpers
   ───────────────────────────────────────────────────────── */

const MAX_TRACE = 240;
const PRIMARY_CHANNELS = {
  metropt: ["TP2", "Oil_temperature", "Motor_current", "Reservoirs"],
  hydraulic: ["PS1", "TS1", "EPS1", "CE"],
  cmapss: ["sensor_2", "sensor_3", "sensor_4", "sensor_7"],
};
const SPEED_OPTIONS = [
  { value: 1, label: "1× — real-time" },
  { value: 10, label: "10× — 1 sample/s" },
  { value: 60, label: "60× — 1 min/s" },
  { value: 600, label: "600× — 10 min/s" },
  { value: 3000, label: "3000× — 50 min/s" },
  { value: 5000, label: "5000× — 1.4 h/s" },
];

const PALETTE = {
  text: "#e6ecf2",
  muted: "#7a8497",
  grid: "rgba(148, 172, 204, 0.08)",
  accent: "#06b6d4",
  threshold: "#fb923c",
};

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;" }[c]),
  );
}

function formatSimTime(seconds) {
  if (!seconds && seconds !== 0) return "—";
  const days = Math.floor(seconds / 86400);
  const hours = Math.floor((seconds % 86400) / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const parts = [];
  if (days) parts.push(`${days}d`);
  if (hours || days) parts.push(`${hours}h`);
  parts.push(`${minutes}m`);
  return parts.join(" ");
}

function formatDuration(seconds) {
  if (seconds == null) return { value: "—", unit: "" };
  if (seconds < 60) return { value: seconds.toFixed(0), unit: "seconds" };
  if (seconds < 3600) return { value: (seconds / 60).toFixed(0), unit: "minutes" };
  if (seconds < 86400) return { value: (seconds / 3600).toFixed(1), unit: "hours" };
  if (seconds < 60 * 86400)
    return { value: (seconds / 86400).toFixed(1), unit: "days" };
  return { value: (seconds / (30 * 86400)).toFixed(1), unit: "months" };
}

function normalizeAlertLevel(level) {
  const v = String(level || "ok").toLowerCase();
  if (v === "critical" || v === "high" || v === "alert") return "alert";
  if (v === "medium" || v === "warn") return "warn";
  return "ok";
}

function alertText(level) {
  if (level === "alert") return "ALARM — anomaly above threshold";
  if (level === "warn") return "WATCH — score elevated";
  return "CLEAR — all systems nominal";
}

/* ─────────────────────────────────────────────────────────
   Chart.js instance factories (managed imperatively by the root
   component — no per-event reactive watchers, redrawn in an rAF loop)
   ───────────────────────────────────────────────────────── */

function makeSparkChart(canvas) {
  return new Chart(canvas, {
    type: "line",
    data: { labels: [], datasets: [{ data: [], borderColor: PALETTE.accent, borderWidth: 1.4, pointRadius: 0, tension: 0.25, fill: false }] },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: { x: { display: false }, y: { display: false } },
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    },
  });
}

function makeScoreChartInstance(canvas) {
  return new Chart(canvas, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        { label: "Score", data: [], borderColor: PALETTE.accent, backgroundColor: "rgba(6, 182, 212, 0.05)", borderWidth: 1.6, pointRadius: 0, tension: 0.18, fill: true },
        { label: "Threshold", data: [], borderColor: PALETTE.threshold, borderWidth: 1.2, borderDash: [6, 4], pointRadius: 0, fill: false },
      ],
    },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: {
          ticks: { color: PALETTE.muted, font: { size: 10, family: "JetBrains Mono" } },
          grid: { color: PALETTE.grid },
          border: { color: PALETTE.grid },
        },
      },
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
    },
  });
}

/* ─────────────────────────────────────────────────────────
   Gauge — semicircle health % display
   ───────────────────────────────────────────────────────── */

const HealthGauge = {
  props: { value: { type: Number, default: null } },
  setup(props) {
    const arcLength = 282;
    const dashArray = computed(() => {
      if (props.value == null) return `0 ${arcLength}`;
      const v = Math.max(0, Math.min(100, props.value));
      const dash = (v / 100) * arcLength;
      return `${dash} ${arcLength - dash}`;
    });
    const color = computed(() => {
      if (props.value == null) return "var(--text-faint)";
      if (props.value < 30) return "var(--alert)";
      if (props.value < 70) return "var(--warn)";
      return "var(--ok)";
    });
    const display = computed(() =>
      props.value == null ? "—" : Math.round(props.value).toString(),
    );
    return { arcLength, dashArray, color, display };
  },
  template: `
    <div class="gauge">
      <svg viewBox="0 0 200 110" aria-hidden="true">
        <path d="M10,100 A90,90 0 0,1 190,100" fill="none" stroke="var(--border)" stroke-width="10" stroke-linecap="round"/>
        <path d="M10,100 A90,90 0 0,1 190,100" fill="none" :stroke="color" stroke-width="10" stroke-linecap="round" :stroke-dasharray="dashArray"/>
      </svg>
      <div class="gauge-value">{{ display }}<em>%</em></div>
    </div>
  `,
};

/* ─────────────────────────────────────────────────────────
   Root App
   ───────────────────────────────────────────────────────── */

const App = {
  components: { HealthGauge },
  setup() {
    /* ── state ─────────────────────────────────────────── */

    const state = reactive({
      // catalogue
      sources: [],
      failures: [],

      // form
      activeSource: "metropt",
      calibSamples: 60000,
      speed: 3000,

      // connection / lifecycle
      wsState: "connecting",
      phase: "idle",
      phaseDetail: "awaiting start",
      progress: 0,
      paused: false,

      // live values (updated at most once per animation frame, not per event)
      score: null,
      threshold: null,
      health: null,
      alertLevel: "ok",
      elapsedSeconds: 0,
      sensorLatest: {}, // {channel: latest value} — only the displayed number

      // synthesis
      contributors: [],
      forecast: null,
      diagnosis: null,
      currentEpisodeId: null,

      // operator feedback
      feedback: {
        pending: null,   // null | "confirmed" | "false_positive"
        note: "",
        toast: "",
        toastKind: "ok",
      },

      // Layer-2 adaptation
      adaptation: { extra_healthy: 0, extra_cap: 0, snapshots: [], current_snapshot: null },
      recalibrating: false,

      // work-request modal
      workRequest: {
        open: false,
        stage: "preview", // "preview" | "submitting" | "submitted" | "error"
        loading: false,
        preview: null,    // built WorkRequest payload
        submitted: null,  // { request, cmms_ref, submitted_at, storage_path, ... }
        error: "",
      },
    });

    const ws = ref(null);

    /* ── non-reactive render buffers ───────────────────────
       High-frequency data (chart points + the latest reading scalars) is
       kept OUT of Vue reactivity. The WS handler writes here cheaply; a
       single rAF loop throttles all reactive updates + chart redraws to one
       animation frame, so event rate (hundreds/sec at high speed) never
       drives hundreds of re-renders/sec. */
    const traceBuffers = {}; // {channel: number[]} — plain arrays
    const scoreBuf = [];     // smoothed scores
    const thrBuf = [];       // threshold aligned with scoreBuf
    let pendingReading = null; // latest reading event, for batched scalar flush
    let streamDirty = false;
    let sensorCharts = [];   // Chart[] (non-reactive)
    let scoreChartInst = null;
    let rafHandle = null;
    let lastFlush = 0;
    const FLUSH_MS = 33;     // ~30 fps cap

    /* ── derived ───────────────────────────────────────── */

    const sourceSpec = computed(() =>
      state.sources.find((s) => s.name === state.activeSource),
    );
    const primaryChannels = computed(
      () => PRIMARY_CHANNELS[state.activeSource] || ["—", "—", "—", "—"],
    );
    const calibrationUnit = computed(() => sourceSpec.value?.natural_unit || "samples");
    const isRunning = computed(() => state.phase !== "idle" && state.phase !== "finished");
    const canStart = computed(() => !isRunning.value && state.wsState === "connected");
    const canJump = computed(() => state.phase === "inferring");

    const normalizedAlert = computed(() => normalizeAlertLevel(state.alertLevel));
    const alertMessage = computed(() => alertText(normalizedAlert.value));

    const channelLabels = computed(() => {
      const map = {};
      const desc = sourceSpec.value?.feature_descriptions; // not currently exposed via /sources
      // Server already attaches `label` to each contributor; here we just provide
      // a fallback display name for the sensor cells.
      primaryChannels.value.forEach((c) => {
        map[c] = c.replace(/_/g, " ");
      });
      return map;
    });

    /* ── render buffers + throttled flush ──────────────── */

    function resetStreams() {
      for (const k of Object.keys(traceBuffers)) delete traceBuffers[k];
      scoreBuf.length = 0;
      thrBuf.length = 0;
      pendingReading = null;
      streamDirty = false;
      state.sensorLatest = {};
      state.contributors = [];
      state.forecast = null;
      state.diagnosis = null;
      state.score = null;
      state.threshold = null;
      state.health = null;
      state.alertLevel = "ok";
      state.currentEpisodeId = null;
      state.feedback.pending = null;
      state.feedback.note = "";
      state.adaptation = { extra_healthy: 0, extra_cap: 0, snapshots: [], current_snapshot: null };
      // Clear the chart canvases immediately.
      sensorCharts.forEach((c) => {
        if (!c) return;
        c.data.labels = [];
        c.data.datasets[0].data = [];
        c.update("none");
      });
      if (scoreChartInst) {
        scoreChartInst.data.labels = [];
        scoreChartInst.data.datasets[0].data = [];
        scoreChartInst.data.datasets[1].data = [];
        scoreChartInst.update("none");
      }
    }

    function buildCharts() {
      sensorCharts.forEach((c) => c && c.destroy());
      sensorCharts = [];
      primaryChannels.value.forEach((_, i) => {
        const canvas = document.getElementById(`spark-${i}`);
        if (canvas) sensorCharts[i] = makeSparkChart(canvas);
      });
      const sc = document.getElementById("score-canvas");
      if (scoreChartInst) scoreChartInst.destroy();
      scoreChartInst = sc ? makeScoreChartInstance(sc) : null;
    }

    function flush() {
      // 1) batched scalar update (one reactive mutation set per frame)
      const ev = pendingReading;
      if (ev) {
        if (typeof ev.score === "number") {
          state.score = ev.score;
          state.threshold = ev.threshold;
        }
        if (typeof ev.health === "number") state.health = ev.health;
        if (ev.alert_level) state.alertLevel = ev.alert_level;
        if (ev.phase) state.phase = ev.phase;
        if (typeof ev.elapsed_simulated_seconds === "number")
          state.elapsedSeconds = ev.elapsed_simulated_seconds;
        if (Array.isArray(ev.contributors)) state.contributors = ev.contributors;
        if (ev.forecast !== undefined) state.forecast = ev.forecast;
        if (ev.diagnosis !== undefined) state.diagnosis = ev.diagnosis;
        state.currentEpisodeId = typeof ev.episode_id === "string" ? ev.episode_id : null;
        // latest sensor cell numbers
        const latest = {};
        primaryChannels.value.forEach((ch) => {
          const buf = traceBuffers[ch];
          if (buf && buf.length) latest[ch] = buf[buf.length - 1];
        });
        state.sensorLatest = latest;
      }
      // 2) redraw charts directly from plain buffers
      sensorCharts.forEach((c, i) => {
        if (!c) return;
        const ch = primaryChannels.value[i];
        const buf = traceBuffers[ch] || [];
        c.data.labels = buf.map((_, k) => k);
        c.data.datasets[0].data = buf;
        c.update("none");
      });
      if (scoreChartInst) {
        scoreChartInst.data.labels = scoreBuf.map((_, k) => k);
        scoreChartInst.data.datasets[0].data = scoreBuf;
        scoreChartInst.data.datasets[1].data = thrBuf;
        scoreChartInst.update("none");
      }
    }

    function renderLoop(ts) {
      if (streamDirty && ts - lastFlush >= FLUSH_MS) {
        lastFlush = ts;
        streamDirty = false;
        flush();
      }
      rafHandle = requestAnimationFrame(renderLoop);
    }

    /* ── WebSocket ─────────────────────────────────────── */

    function openSocket() {
      if (ws.value) {
        try { ws.value.close(); } catch (_) {}
      }
      state.wsState = "connecting";
      const proto = location.protocol === "https:" ? "wss" : "ws";
      const sock = new WebSocket(`${proto}://${location.host}/ws`);
      sock.onopen = () => (state.wsState = "connected");
      sock.onerror = () => (state.wsState = "disconnected");
      sock.onclose = () => {
        state.wsState = "disconnected";
        setTimeout(openSocket, 1000);
      };
      sock.onmessage = (msg) => {
        let ev;
        try {
          ev = JSON.parse(msg.data);
        } catch (e) {
          return;
        }
        handleEvent(ev);
      };
      ws.value = sock;
    }

    function handleEvent(ev) {
      // Phase events are rare and drive the progress bar — apply immediately.
      if (ev.kind === "phase") {
        state.phase = ev.phase || "idle";
        state.phaseDetail = ev.detail || "";
        state.progress = typeof ev.progress === "number" ? ev.progress : state.progress;
        return;
      }
      if (ev.kind === "feedback") {
        const fp = ev.verdict === "false_positive";
        let msg = fp ? "Recorded — alert dismissed" : "Recorded — fault confirmed";
        if (fp && ev.collected && ev.collected.added)
          msg += ` (+${ev.collected.added} window${ev.collected.added === 1 ? "" : "s"} learned)`;
        showToast(msg, fp ? "ok" : "warn");
        if (fp) state.currentEpisodeId = null;
        if (ev.adaptation) state.adaptation = ev.adaptation;
        return;
      }
      if (ev.kind === "recalibrated") {
        if (ev.adaptation) state.adaptation = ev.adaptation;
        showToast(`Recalibrated on ${ev.n_extra} dismissed windows — threshold ${Number(ev.threshold).toFixed(3)}`, "ok");
        return;
      }
      if (ev.kind === "reverted") {
        if (ev.adaptation) state.adaptation = ev.adaptation;
        showToast("Reverted to the previous model", "warn");
        return;
      }
      if (ev.kind !== "reading") return;

      // Reading events are high-frequency: write to plain buffers only, mark
      // dirty, and let the rAF loop flush to the UI at most once per frame.
      const features = ev.features || {};
      primaryChannels.value.forEach((channel) => {
        const v = features[channel];
        if (typeof v !== "number") return;
        let buf = traceBuffers[channel];
        if (!buf) buf = traceBuffers[channel] = [];
        buf.push(v);
        if (buf.length > MAX_TRACE) buf.shift();
      });
      if (typeof ev.score === "number") {
        scoreBuf.push(ev.score);
        thrBuf.push(ev.threshold);
        if (scoreBuf.length > MAX_TRACE) {
          scoreBuf.shift();
          thrBuf.shift();
        }
      }
      pendingReading = ev;
      streamDirty = true;
    }

    /* ── REST control ──────────────────────────────────── */

    async function loadSources() {
      try {
        const data = await api.sources();
        state.sources = data.sources || [];
        const spec = state.sources.find((s) => s.name === state.activeSource);
        if (spec) state.calibSamples = spec.suggested_calibration ?? state.calibSamples;
      } catch (e) {
        console.error("loadSources", e);
      }
    }

    async function loadFailures() {
      try {
        const data = await api.failures(state.activeSource);
        state.failures = data.failures || [];
      } catch (e) {
        console.error("loadFailures", e);
      }
    }

    async function start() {
      resetStreams();
      try {
        await api.start({
          source: state.activeSource,
          speed: Number(state.speed),
          calibration_samples: Number(state.calibSamples),
        });
      } catch (e) {
        console.error("start", e);
      }
    }

    async function stop() {
      try { await api.stop(); } catch (e) { console.error("stop", e); }
      state.phase = "idle";
      state.phaseDetail = "stopped";
    }

    async function togglePause() {
      try {
        const res = await api.pause();
        state.paused = res?.status === "paused";
      } catch (e) {
        console.error("pause", e);
      }
    }

    async function applySpeed() {
      try {
        await api.setSpeed(Number(state.speed));
      } catch (e) {
        console.error("speed", e);
      }
    }

    async function jumpTo(failureId) {
      try {
        await api.jump(failureId);
      } catch (e) {
        console.error("jump", e);
      }
    }

    /* ── work-request modal ────────────────────────────── */

    function workRequestPayload(previewOnly) {
      return {
        diagnosis: state.diagnosis || {},
        contributors: state.contributors || [],
        forecast: state.forecast,
        asset_id: state.activeSource,
        asset_label: sourceSpec.value?.display_name || state.activeSource,
        score: state.score,
        threshold: state.threshold,
        elapsed_simulated_seconds: state.elapsedSeconds,
        preview_only: previewOnly,
      };
    }

    async function openWorkRequest() {
      state.workRequest.open = true;
      state.workRequest.stage = "preview";
      state.workRequest.loading = true;
      state.workRequest.preview = null;
      state.workRequest.submitted = null;
      state.workRequest.error = "";
      try {
        const res = await jsonRequest("/work_request", {
          method: "POST",
          body: workRequestPayload(true),
        });
        if (res.status === "preview") {
          state.workRequest.preview = res.request;
        } else {
          state.workRequest.stage = "error";
          state.workRequest.error = res.detail || "Failed to build preview.";
        }
      } catch (e) {
        state.workRequest.stage = "error";
        state.workRequest.error = String(e);
      } finally {
        state.workRequest.loading = false;
      }
    }

    async function submitWorkRequest() {
      state.workRequest.stage = "submitting";
      state.workRequest.loading = true;
      try {
        const res = await jsonRequest("/work_request", {
          method: "POST",
          body: workRequestPayload(false),
        });
        if (res.status === "submitted") {
          state.workRequest.submitted = res;
          state.workRequest.stage = "submitted";
        } else {
          state.workRequest.stage = "error";
          state.workRequest.error = res.detail || "Submission failed.";
        }
      } catch (e) {
        state.workRequest.stage = "error";
        state.workRequest.error = String(e);
      } finally {
        state.workRequest.loading = false;
      }
    }

    function closeWorkRequest() {
      state.workRequest.open = false;
    }

    const canRaiseWorkRequest = computed(() => {
      const u = state.diagnosis?.urgency;
      return u && u !== "info" && u !== "low";
    });

    /* ── operator feedback ─────────────────────────────── */

    let toastTimer = null;
    function showToast(text, kind = "ok") {
      state.feedback.toast = text;
      state.feedback.toastKind = kind;
      if (toastTimer) clearTimeout(toastTimer);
      toastTimer = setTimeout(() => (state.feedback.toast = ""), 3500);
    }

    function askFeedback(verdict) {
      state.feedback.pending = verdict;
      state.feedback.note = "";
    }

    function cancelFeedback() {
      state.feedback.pending = null;
      state.feedback.note = "";
    }

    async function sendFeedback() {
      const episodeId = state.currentEpisodeId;
      const verdict = state.feedback.pending;
      if (!episodeId || !verdict) {
        cancelFeedback();
        return;
      }
      const note = state.feedback.note;
      cancelFeedback();
      try {
        const res = await api.feedback({ episode_id: episodeId, verdict, note });
        if (res.status !== "recorded") {
          showToast(res.detail || "Feedback failed", "warn");
        }
        // success toast is driven by the broadcast 'feedback' event so every
        // connected console sees it; no extra toast here.
      } catch (e) {
        console.error("feedback", e);
        showToast("Feedback request failed", "warn");
      }
    }

    const canRecalibrate = computed(() => state.adaptation.extra_healthy > 0 && !state.recalibrating);
    const canRevert = computed(() => (state.adaptation.snapshots || []).length > 1 && !state.recalibrating);

    async function recalibrate() {
      if (!canRecalibrate.value) return;
      state.recalibrating = true;
      try {
        const res = await api.recalibrate();
        if (res.status !== "recalibrated") showToast(res.detail || "Recalibration failed", "warn");
      } catch (e) {
        console.error("recalibrate", e);
        showToast("Recalibration request failed", "warn");
      } finally {
        state.recalibrating = false;
      }
    }

    async function revert() {
      if (!canRevert.value) return;
      try {
        const res = await api.revert();
        if (res.status !== "reverted") showToast(res.detail || "Revert failed", "warn");
      } catch (e) {
        console.error("revert", e);
        showToast("Revert request failed", "warn");
      }
    }

    /* ── source-change side effects ───────────────────── */

    watch(
      () => state.activeSource,
      (name) => {
        const spec = state.sources.find((s) => s.name === name);
        if (spec) state.calibSamples = spec.suggested_calibration ?? state.calibSamples;
        loadFailures();
        // Channel set changed → the sensor canvases are re-rendered; rebuild
        // the chart instances against the new DOM nodes on the next tick.
        resetStreams();
        nextTick(buildCharts);
      },
    );

    /* ── lifecycle ─────────────────────────────────────── */

    onMounted(async () => {
      await loadSources();
      await loadFailures();
      await nextTick();
      buildCharts();
      rafHandle = requestAnimationFrame(renderLoop);
      openSocket();
    });

    onUnmounted(() => {
      if (rafHandle) cancelAnimationFrame(rafHandle);
      sensorCharts.forEach((c) => c && c.destroy());
      if (scoreChartInst) scoreChartInst.destroy();
      if (ws.value) try { ws.value.close(); } catch (_) {}
    });

    /* ── forecast computed ─────────────────────────────── */

    const forecastDisplay = computed(() => {
      const f = state.forecast;
      if (!f) {
        return {
          status: "warming_up",
          statusText: "awaiting calibration",
          value: "—",
          unit: "",
          band: "",
        };
      }
      if (f.status === "warming_up") {
        return {
          status: "warming_up",
          statusText: `collecting (${f.samples ?? 0})`,
          value: "—",
          unit: "",
          band: "",
        };
      }
      if (f.status === "above_threshold") {
        return {
          status: "above_threshold",
          statusText: "score already above threshold",
          value: "0",
          unit: "alert now",
          band: "",
        };
      }
      if (f.status === "stable") {
        const slope = f.slope_per_day ?? 0;
        return {
          status: "stable",
          statusText: "no upward trend detected",
          value: ">",
          unit: "stable",
          band: `slope ${slope >= 0 ? "+" : "−"}${Math.abs(slope).toFixed(3)} score/day`,
        };
      }
      // trending_up
      const main = formatDuration(Math.max(0, f.time_to_alert_seconds ?? 0));
      let band = "";
      if (f.time_to_alert_low_seconds != null && f.time_to_alert_high_seconds != null) {
        const lo = formatDuration(Math.max(0, f.time_to_alert_low_seconds));
        const hi = formatDuration(Math.max(0, f.time_to_alert_high_seconds));
        band = `95% band: ${lo.value} ${lo.unit} → ${hi.value} ${hi.unit}`;
      }
      return {
        status: "trending_up",
        statusText: "rising trend — alert projected",
        value: main.value,
        unit: main.unit,
        band,
      };
    });

    /* ── exposed to template ───────────────────────────── */

    return {
      state,
      sourceSpec,
      primaryChannels,
      channelLabels,
      calibrationUnit,
      isRunning,
      canStart,
      canJump,
      canRaiseWorkRequest,
      normalizedAlert,
      alertMessage,
      forecastDisplay,
      SPEED_OPTIONS,
      // actions
      start,
      stop,
      togglePause,
      applySpeed,
      jumpTo,
      openWorkRequest,
      submitWorkRequest,
      closeWorkRequest,
      askFeedback,
      cancelFeedback,
      sendFeedback,
      canRecalibrate,
      canRevert,
      recalibrate,
      revert,
      // formatters
      formatSimTime,
    };
  },

  template: `
  <div class="app">

    <!-- SYSTEM BAR -->
    <header class="sysbar">
      <div class="sys-brand">
        <div class="sys-brand-mark">ES</div>
        <div>
          <div class="sys-brand-name">EdgeSense</div>
          <div class="sys-brand-sub">Operator Console</div>
        </div>
      </div>
      <div class="sys-indicators">
        <div class="sys-indicator">
          <span>Mode</span>
          <span class="sys-indicator-value">{{ (sourceSpec && sourceSpec.display_name) || state.activeSource }}</span>
        </div>
        <div class="sys-indicator">
          <span>Asset state</span>
          <span class="sys-indicator-value">{{ state.phase.toUpperCase() }}</span>
        </div>
        <div class="sys-indicator">
          <span>Alert</span>
          <span class="led"
                :class="{
                  'led--ok': normalizedAlert === 'ok',
                  'led--warn': normalizedAlert === 'warn',
                  'led--alert': normalizedAlert === 'alert',
                  'led--pulse': normalizedAlert === 'alert',
                }"></span>
        </div>
      </div>
    </header>

    <!-- CONTROLS -->
    <section class="controlbar">
      <label class="field">
        <span class="field-label">Asset</span>
        <select v-model="state.activeSource" :disabled="isRunning">
          <option v-for="s in state.sources" :key="s.name" :value="s.name" :disabled="s.available === 'false'">
            {{ s.display_name }}{{ s.available === 'false' ? ' (coming soon)' : '' }}
          </option>
        </select>
      </label>
      <label class="field">
        <span class="field-label">Calibration {{ calibrationUnit }}</span>
        <input type="number" v-model.number="state.calibSamples" min="500" step="500" :disabled="isRunning" />
      </label>
      <label class="field">
        <span class="field-label">Simulation speed</span>
        <select v-model.number="state.speed" @change="applySpeed">
          <option v-for="opt in SPEED_OPTIONS" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </label>
      <div class="adaptation-group" v-if="isRunning">
        <span class="adaptation-label">
          Learned <strong>{{ state.adaptation.extra_healthy }}</strong>/{{ state.adaptation.extra_cap }} windows
          <span class="adaptation-ver" v-if="state.adaptation.snapshots && state.adaptation.snapshots.length">v{{ state.adaptation.snapshots.length - 1 }}</span>
        </span>
        <button class="btn" :disabled="!canRecalibrate" @click="recalibrate">
          {{ state.recalibrating ? 'Recalibrating…' : 'Recalibrate' }}
        </button>
        <button class="btn" :disabled="!canRevert" @click="revert">Revert</button>
      </div>
      <div class="actions">
        <button class="btn btn--primary" :disabled="!canStart" @click="start">Start</button>
        <button class="btn" :data-paused="state.paused" :disabled="!isRunning" @click="togglePause">{{ state.paused ? 'Resume' : 'Pause' }}</button>
        <button class="btn" :disabled="!isRunning" @click="stop">Stop</button>
      </div>
    </section>

    <!-- PHASE STRIP -->
    <section class="phase-strip">
      <span class="phase-strip-label">Phase</span>
      <span class="phase-tag" :data-phase="state.phase">
        <span class="led"
              :class="{
                'led--info': state.phase === 'calibrating',
                'led--warn': state.phase === 'training',
                'led--ok': state.phase === 'inferring',
                'led--alert': state.phase === 'failed',
                'led--pulse': state.phase === 'training' || state.phase === 'calibrating',
              }"></span>
        {{ state.phase.toUpperCase() }}
      </span>
      <div class="progress-bar" :aria-valuenow="state.progress * 100">
        <div class="progress-fill"
             :class="{ 'progress-fill--indeterminate': state.phase === 'training' }"
             :style="{ width: (state.progress * 100) + '%' }"></div>
      </div>
      <span class="phase-detail">{{ state.phaseDetail || '—' }}</span>
      <span class="tx-pill" :data-state="state.wsState">
        <span class="led" :class="{
          'led--ok': state.wsState === 'connected',
          'led--info': state.wsState === 'connecting',
          'led--alert': state.wsState === 'disconnected',
          'led--pulse': state.wsState !== 'connected',
        }"></span>
        TX {{ state.wsState }}
      </span>
    </section>

    <!-- DIAGNOSIS BANNER -->
    <section class="diagnosis" :data-urgency="(state.diagnosis && state.diagnosis.urgency) || 'info'">
      <div>
        <span class="urgency-pill" :data-urgency="(state.diagnosis && state.diagnosis.urgency) || 'info'">
          <span class="led" :class="{
            'led--ok': state.diagnosis?.urgency === 'low',
            'led--warn': state.diagnosis?.urgency === 'medium',
            'led--alert': state.diagnosis?.urgency === 'high' || state.diagnosis?.urgency === 'critical',
            'led--pulse': state.diagnosis?.urgency === 'critical',
          }"></span>
          {{ (state.diagnosis && state.diagnosis.urgency_label) || 'awaiting calibration' }}
        </span>
        <h2 class="diagnosis-root">{{ (state.diagnosis && state.diagnosis.root_cause) || 'EdgeSense — awaiting calibration' }}</h2>
        <ul class="diagnosis-evidence" v-if="state.diagnosis && state.diagnosis.evidence && state.diagnosis.evidence.length">
          <li v-for="(e, i) in state.diagnosis.evidence" :key="i">{{ e }}</li>
        </ul>
      </div>
      <aside class="action-block">
        <div class="action-label">Recommended action</div>
        <div class="action-body">
          {{ (state.diagnosis && state.diagnosis.recommended_action) || 'Start a simulation to begin monitoring.' }}
        </div>
        <div v-if="state.diagnosis && state.diagnosis.matched_rule" class="action-rule">Rule: {{ state.diagnosis.matched_rule }}</div>
        <div class="action-buttons">
          <button v-if="canRaiseWorkRequest"
                  class="btn btn--primary action-cta"
                  @click="openWorkRequest"
                  title="Build a CMMS work request from this diagnosis">
            File work request ↗
          </button>
          <template v-if="state.currentEpisodeId">
            <button class="btn action-cta" @click="askFeedback('confirmed')"
                    title="Confirm this is a real fault">✓ Confirm fault</button>
            <button class="btn action-cta" @click="askFeedback('false_positive')"
                    title="Mark as a false alarm and dismiss">✕ False positive</button>
          </template>
        </div>
        <div v-if="state.feedback.pending" class="feedback-confirm">
          <span class="feedback-confirm-label">
            {{ state.feedback.pending === 'confirmed' ? 'Confirm fault' : 'Dismiss as false positive' }} — optional note:
          </span>
          <input class="feedback-note" v-model="state.feedback.note"
                 placeholder="e.g. idle regime, known sensor drift…"
                 @keyup.enter="sendFeedback" />
          <div class="feedback-confirm-actions">
            <button class="btn action-cta" @click="cancelFeedback">Cancel</button>
            <button class="btn btn--primary action-cta" @click="sendFeedback">Send</button>
          </div>
        </div>
      </aside>
    </section>

    <div v-if="state.feedback.toast" class="feedback-toast" :data-kind="state.feedback.toastKind">
      {{ state.feedback.toast }}
    </div>

    <!-- WORKSPACE -->
    <main class="workspace">

      <!-- LEFT COL -->
      <div class="col">

        <!-- Sensor traces -->
        <section class="panel">
          <header class="panel-head">
            <span class="panel-title">Live sensor traces</span>
            <span class="panel-head-aux">{{ primaryChannels.join(' · ') }}</span>
          </header>
          <div class="panel-body">
            <div class="sensors-grid">
              <div v-for="(ch, i) in primaryChannels" :key="i" class="sensor-cell">
                <div class="sensor-name">{{ channelLabels[ch] || ch }}</div>
                <div class="sensor-value">
                  {{ state.sensorLatest[ch] != null ? state.sensorLatest[ch].toFixed(2) : '—' }}
                </div>
                <div class="sensor-spark"><canvas :id="'spark-' + i"></canvas></div>
              </div>
            </div>
          </div>
        </section>

        <!-- Score + attribution -->
        <section class="panel">
          <header class="panel-head">
            <span class="panel-title">Anomaly score</span>
            <span class="panel-head-aux">{{ state.score == null ? '—' : state.score.toFixed(3) }}</span>
          </header>
          <div class="panel-body">
            <div class="score-chart-wrap"><canvas id="score-canvas"></canvas></div>
            <div class="score-legend">
              <span><span class="score-legend-marker"></span>Score (smoothed)</span>
              <span><span class="score-legend-marker score-legend-marker--threshold"></span>Threshold</span>
            </div>
          </div>
          <div class="panel-section">
            <header class="panel-head" style="border: none; padding: 0 0 8px;">
              <span class="panel-title">What's driving it</span>
              <span class="panel-head-aux">top {{ Math.min(state.contributors.length, 5) }}</span>
            </header>
            <div class="attribution-list">
              <div v-if="state.contributors.length === 0" class="attribution-empty">awaiting inference…</div>
              <div v-for="c in state.contributors.slice(0, 5)" :key="c.name" class="attribution-row">
                <div class="attribution-row-main">
                  <div class="attribution-channel">
                    <span>{{ c.label || c.name }}</span>
                    <span class="attribution-tag" v-if="c.label && c.label !== c.name">{{ c.name }}</span>
                  </div>
                  <div class="attribution-bar"><span :style="barStyle(c, state.contributors)"></span></div>
                  <div class="attribution-delta" :class="{ neg: c.delta_pct < 0 }">
                    {{ c.delta_pct >= 0 ? '+' : '−' }}{{ Math.abs(c.delta_pct).toFixed(1) }} pts
                  </div>
                </div>
                <div v-if="c.action" class="attribution-action">{{ c.action }}</div>
              </div>
            </div>
          </div>
        </section>

      </div>

      <!-- RIGHT COL -->
      <div class="col">

        <!-- Health -->
        <section class="panel">
          <header class="panel-head">
            <span class="panel-title">Health</span>
            <span class="panel-head-aux">{{ state.health == null ? '—' : Math.round(state.health) + '%' }}</span>
          </header>
          <div class="health-panel-body">
            <HealthGauge :value="state.health" />
            <div class="gauge-caption">
              {{
                state.phase === 'calibrating' ? 'calibration in progress' :
                state.phase === 'training'    ? 'fitting model' :
                state.phase === 'inferring'   ? 'monitoring live' :
                'awaiting calibration'
              }}
            </div>
          </div>
        </section>

        <!-- Alert + Forecast + Metrics + Jump -->
        <section class="panel">
          <header class="panel-head">
            <span class="panel-title">Alert state</span>
          </header>
          <div class="panel-body">
            <div class="alert-pill" :data-level="normalizedAlert">
              <span class="led" :class="{
                'led--ok': normalizedAlert === 'ok',
                'led--warn': normalizedAlert === 'warn',
                'led--alert': normalizedAlert === 'alert',
                'led--pulse': normalizedAlert === 'alert',
              }"></span>
              {{ alertMessage }}
            </div>

            <div class="forecast-block">
              <div style="display:flex; justify-content: space-between; align-items: baseline;">
                <span class="forecast-status" :data-state="forecastDisplay.status">
                  <span class="led" :class="{
                    'led--ok': forecastDisplay.status === 'stable',
                    'led--warn': forecastDisplay.status === 'trending_up',
                    'led--alert': forecastDisplay.status === 'above_threshold',
                  }"></span>
                  {{ forecastDisplay.statusText }}
                </span>
              </div>
              <div class="forecast-value">{{ forecastDisplay.value }}<em>{{ forecastDisplay.unit }}</em></div>
              <div class="forecast-band" v-if="forecastDisplay.band">{{ forecastDisplay.band }}</div>
              <div class="forecast-caption">label-free trend extrapolation</div>
            </div>

            <div class="metrics">
              <div class="metric">
                <span class="metric-label">Score</span>
                <span class="metric-value">{{ state.score == null ? '—' : state.score.toFixed(3) }}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Threshold</span>
                <span class="metric-value">{{ state.threshold == null ? '—' : state.threshold.toFixed(3) }}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Source</span>
                <span class="metric-value">{{ state.activeSource }}</span>
              </div>
              <div class="metric">
                <span class="metric-label">Sim time</span>
                <span class="metric-value">{{ formatSimTime(state.elapsedSeconds) }}</span>
              </div>
            </div>
          </div>
          <div class="panel-section">
            <header class="panel-head" style="border: none; padding: 0 0 8px;">
              <span class="panel-title">Jump to event</span>
              <span class="panel-head-aux">{{ state.failures.length }} markers</span>
            </header>
            <p class="jump-hint">{{ canJump ? 'click a row to skip to ~10 min before the labelled event' : 'available once inferring' }}</p>
            <ul class="jump-list">
              <li v-for="f in state.failures" :key="f.id" class="jump-item">
                <span class="jump-name">
                  {{ f.label }}
                  <span class="jump-source-tag" :class="{ 'jump-source-tag--audit': f.source === 'audit' }">{{ f.source === 'audit' ? 'audit' : 'logged' }}</span>
                </span>
                <button class="jump-btn" :disabled="!canJump" @click="jumpTo(f.id)">▶ jump</button>
              </li>
            </ul>
          </div>
        </section>

      </div>
    </main>

    <!-- WORK-REQUEST MODAL -->
    <div v-if="state.workRequest.open" class="modal-backdrop" @click.self="closeWorkRequest">
      <div class="modal" role="dialog" aria-modal="true">
        <header class="modal-head">
          <div>
            <div class="modal-eyebrow">CMMS Work Request</div>
            <h3 class="modal-title">
              <template v-if="state.workRequest.stage === 'submitted'">Work request submitted</template>
              <template v-else-if="state.workRequest.stage === 'submitting'">Submitting…</template>
              <template v-else-if="state.workRequest.stage === 'error'">Submission failed</template>
              <template v-else>Review work request</template>
            </h3>
          </div>
          <button class="modal-close" @click="closeWorkRequest" aria-label="Close">×</button>
        </header>

        <div class="modal-body">
          <!-- LOADING -->
          <div v-if="state.workRequest.loading && !state.workRequest.preview && !state.workRequest.submitted"
               class="modal-empty">building preview…</div>

          <!-- ERROR -->
          <div v-else-if="state.workRequest.stage === 'error'" class="modal-error">
            <div class="modal-error-title">
              <span class="led led--alert led--pulse"></span>
              Could not submit
            </div>
            <pre class="modal-pre">{{ state.workRequest.error }}</pre>
          </div>

          <!-- SUBMITTED -->
          <div v-else-if="state.workRequest.stage === 'submitted' && state.workRequest.submitted" class="modal-success">
            <div class="modal-success-row">
              <span class="led led--ok"></span>
              <span>Filed via <code>{{ state.workRequest.submitted.request.metadata?.edgesense_source ? 'MockCmmsClient' : 'CMMS' }}</code>.</span>
            </div>
            <dl class="modal-kv">
              <div><dt>CMMS reference</dt><dd>{{ state.workRequest.submitted.cmms_ref }}</dd></div>
              <div><dt>Submitted at</dt><dd>{{ state.workRequest.submitted.submitted_at }}</dd></div>
              <div v-if="state.workRequest.submitted.storage_path"><dt>Persisted to</dt><dd>{{ state.workRequest.submitted.storage_path }}</dd></div>
            </dl>
            <details class="modal-details">
              <summary>Show submitted payload</summary>
              <pre class="modal-pre">{{ JSON.stringify(state.workRequest.submitted.request, null, 2) }}</pre>
            </details>
          </div>

          <!-- PREVIEW -->
          <template v-else-if="state.workRequest.preview">
            <div class="wr-summary">
              <span class="urgency-pill" :data-urgency="state.workRequest.preview.urgency">
                P{{ state.workRequest.preview.priority }} · {{ state.workRequest.preview.urgency.toUpperCase() }}
              </span>
              <h4 class="wr-title">{{ state.workRequest.preview.title }}</h4>
              <div class="wr-asset">
                {{ state.workRequest.preview.asset_label }}
                <span class="attribution-tag">{{ state.workRequest.preview.asset_id }}</span>
              </div>
            </div>

            <dl class="modal-kv">
              <div><dt>Work type</dt><dd>{{ state.workRequest.preview.work_type }}</dd></div>
              <div><dt>Category</dt><dd>{{ state.workRequest.preview.category }}</dd></div>
              <div><dt>Requested by</dt><dd>{{ state.workRequest.preview.requested_by }}</dd></div>
              <div><dt>Reference</dt><dd>{{ state.workRequest.preview.external_id }}</dd></div>
              <div v-if="state.workRequest.preview.failure_mode"><dt>Failure mode</dt><dd>{{ state.workRequest.preview.failure_mode }}</dd></div>
            </dl>

            <div class="wr-section-label">Description</div>
            <pre class="modal-pre">{{ state.workRequest.preview.description }}</pre>

            <details class="modal-details">
              <summary>Show metadata + contributors</summary>
              <pre class="modal-pre">{{ JSON.stringify(state.workRequest.preview.metadata, null, 2) }}</pre>
            </details>
          </template>
        </div>

        <footer class="modal-foot">
          <button class="btn" @click="closeWorkRequest">
            {{ state.workRequest.stage === 'submitted' ? 'Close' : 'Cancel' }}
          </button>
          <button v-if="state.workRequest.stage === 'preview' && state.workRequest.preview"
                  class="btn btn--primary"
                  :disabled="state.workRequest.loading"
                  @click="submitWorkRequest">
            File work request
          </button>
        </footer>
      </div>
    </div>

  </div>
  `,

  methods: {
    barStyle(c, list) {
      const maxAbs = Math.max(...list.map((x) => Math.abs(x.delta_pct ?? 0)), 1);
      const width = Math.min(100, (Math.max(0, c.delta_pct) / maxAbs) * 100);
      return { width: width + "%" };
    },
  },
};

createApp(App).mount("#app");
