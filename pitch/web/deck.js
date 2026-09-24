/* EdgeSense pitch deck runtime: stage scaling, navigation, presenter view, pace timer.
   No dependencies, works from file:// (the presenter window talks to the deck via postMessage). */
(() => {
  "use strict";

  const W = 1920, H = 1080;
  const deck = document.getElementById("deck");
  const slides = Array.from(deck.querySelectorAll(":scope > .slide"));
  const hud = document.getElementById("hud");
  const isPresenter = location.hash === "#presenter";
  const durations = slides.map((s) => Number(s.dataset.duration) || 20);
  const ends = durations.map((_, i) => durations.slice(0, i + 1).reduce((a, b) => a + b, 0));
  const TOTAL = ends[ends.length - 1];

  let cur = 0;
  let peer = isPresenter ? window.opener : null;
  let black = false;

  /* ---------- cover: a compressor-like pressure trace that leaves its learned envelope ---------- */

  function rng(seed) {
    return () => {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function drawCoverTrace(svg) {
    const rand = rng(7);
    const y0 = 850, A = 40, P = 58, onset = 1330;
    const top = y0 - A - 14, bottom = y0 + 12;
    const smooth = (t) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
    let phase = 0, d = "", flag = null;
    for (let x = -20; x <= 1940; x += 3) {
      const k = smooth((x - onset) / 300);
      phase += 3 / (P * (1 - 0.4 * k));
      const f = phase % 1;
      const v = f < 0.8 ? f / 0.8 : 1 - (f - 0.8) / 0.2;
      const y = y0 - 55 * k - A * (1 + 0.4 * k) * v + (rand() - 0.5) * 4;
      d += (d ? " L" : "M") + x + " " + y.toFixed(1);
      if (!flag && x > onset && y < top - 34) flag = { x, y };
    }
    svg.querySelector(".envelope").setAttribute("d", `M0 ${top} H1920 V${bottom} H0 Z`);
    svg.querySelector(".envelope-edge").setAttribute("d", `M0 ${top} H1920 M0 ${bottom} H1920`);
    const trace = svg.querySelector(".trace");
    trace.setAttribute("d", d);
    trace.style.setProperty("--len", Math.ceil(trace.getTotalLength()));
    if (flag) {
      const g = svg.querySelector(".flag-g");
      g.querySelectorAll("circle").forEach((c) => { c.setAttribute("cx", flag.x); c.setAttribute("cy", flag.y); });
      const label = g.querySelector(".flag-label");
      label.setAttribute("x", flag.x - 44);
      label.setAttribute("y", flag.y - 34);
      label.setAttribute("text-anchor", "end");
    }
  }
  deck.querySelectorAll(".cover-trace").forEach((svg) => {
    if (svg.querySelector(".trace")) drawCoverTrace(svg);
  });

  /* ---------- stage scaling ---------- */

  function fit(el, box) {
    const r = box.getBoundingClientRect();
    return Math.min(r.width / W, r.height / H);
  }
  function layout() {
    if (isPresenter) {
      document.querySelectorAll(".p-frame").forEach((f) => {
        const d = f.querySelector(".deck");
        if (d) d.style.setProperty("--ps", fit(d, f));
      });
    } else {
      deck.style.setProperty("--s", Math.min(innerWidth / W, innerHeight / H));
    }
  }
  addEventListener("resize", layout);

  /* ---------- navigation ---------- */

  function show(i, { silent = false } = {}) {
    i = Math.max(0, Math.min(slides.length - 1, i | 0));
    slides.forEach((s, k) => s.classList.toggle("is-active", k === i));
    cur = i;
    hud.textContent = `${i + 1} / ${slides.length}`;
    if (!isPresenter && location.hash !== `#${i + 1}`) history.replaceState(null, "", `#${i + 1}`);
    if (!silent) send({ type: "goto", index: i });
    if (isPresenter) {
      if (i > 0) timer.autostart();
      renderPresenter();
    }
  }

  function send(msg) {
    if (peer && !peer.closed) peer.postMessage({ ...msg, src: "edgesense" }, "*");
  }

  addEventListener("message", (e) => {
    const m = e.data;
    if (!m || m.src !== "edgesense") return;
    if (m.type === "hello") { peer = e.source; send({ type: "goto", index: cur }); }
    if (m.type === "goto") show(m.index, { silent: true });
    if (m.type === "black") setBlack(m.on, true);
  });

  function openPresenter() {
    if (isPresenter) return;
    const url = location.href.split("#")[0] + "#presenter";
    peer = window.open(url, "edgesense-presenter", "popup,width=1400,height=860");
  }

  function setBlack(on, fromPeer) {
    black = on;
    document.body.classList.toggle("is-black", on && !isPresenter);
    const b = document.getElementById("p-black");
    if (b) b.textContent = on ? "Écran noir : oui" : "Écran noir";
    if (!fromPeer) send({ type: "black", on });
  }

  function toggleFullscreen() {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen?.();
  }

  addEventListener("keydown", (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    switch (e.key) {
      case "ArrowRight": case "ArrowDown": case "PageDown": case " ": case "Enter": case "n": case "N":
        show(cur + 1); break;
      case "ArrowLeft": case "ArrowUp": case "PageUp": case "Backspace":
        show(cur - 1); break;
      case "Home": show(0); break;
      case "End": show(slides.length - 1); break;
      case "f": case "F": toggleFullscreen(); break;
      case "p": case "P": openPresenter(); break;
      case "b": case "B": case ".": setBlack(!black); break;
      case "t": case "T": timer.toggle(); break;
      case "r": case "R": timer.reset(); break;
      case "h": case "H": case "?": document.body.classList.toggle("show-help"); break;
      case "Escape": document.body.classList.remove("show-help"); return;
      default: return;
    }
    e.preventDefault();
  });

  /* ---------- pace timer (presenter only) ---------- */

  const timer = {
    t0: 0, acc: 0, running: false,
    elapsed() { return (this.acc + (this.running ? Date.now() - this.t0 : 0)) / 1000; },
    toggle() {
      if (!isPresenter) return;
      if (this.running) { this.acc += Date.now() - this.t0; this.running = false; }
      else { this.t0 = Date.now(); this.running = true; }
      tick();
    },
    reset() { if (!isPresenter) return; this.acc = 0; this.running = false; tick(); },
    autostart() { if (!this.running && this.acc === 0) this.toggle(); },
  };
  const mmss = (s) => {
    const v = Math.round(Math.abs(s));
    return `${Math.floor(v / 60)}:${String(v % 60).padStart(2, "0")}`;
  };

  /* ---------- presenter view ---------- */

  let P = null;

  function buildPresenter() {
    document.body.classList.add("is-presenter");
    document.title = "Présentateur · " + document.title;
    const root = document.createElement("div");
    root.className = "presenter";
    root.innerHTML = `
      <div class="p-bar">
        <div class="p-clock"><span id="p-elapsed">0:00</span> <small>/ ${mmss(TOTAL)}</small></div>
        <div class="p-pace ok" id="p-pace">Prêt</div>
        <div class="p-meter"><i id="p-fill"></i><b id="p-mark"></b></div>
        <div class="p-slideno" id="p-no"></div>
        <button class="p-btn" id="p-toggle">Démarrer</button>
        <button class="p-btn" id="p-reset">Zéro</button>
        <button class="p-btn" id="p-black">Écran noir</button>
      </div>
      <div class="p-main">
        <div class="p-frame p-frame--cur"></div>
        <div class="p-keys">→ / Espace : suivante · ← : précédente · T : chrono · R : zéro · B : écran noir</div>
      </div>
      <div class="p-side">
        <div><div class="p-label" style="margin-bottom:8px">Suivante</div><div class="p-frame p-frame--next"><div class="deck" id="p-next"></div></div></div>
        <div class="p-notes" id="p-notes"></div>
      </div>`;
    document.body.appendChild(root);
    root.querySelector(".p-frame--cur").appendChild(deck);
    P = {
      elapsed: root.querySelector("#p-elapsed"), pace: root.querySelector("#p-pace"),
      fill: root.querySelector("#p-fill"), mark: root.querySelector("#p-mark"),
      no: root.querySelector("#p-no"), toggle: root.querySelector("#p-toggle"),
      next: root.querySelector("#p-next"), notes: root.querySelector("#p-notes"),
    };
    P.toggle.addEventListener("click", () => timer.toggle());
    root.querySelector("#p-reset").addEventListener("click", () => timer.reset());
    root.querySelector("#p-black").addEventListener("click", () => setBlack(!black));
    setInterval(tick, 250);
  }

  function renderPresenter() {
    if (!P) return;
    const s = slides[cur];
    P.no.textContent = `${cur + 1} / ${slides.length} · ${s.dataset.label || ""}`;
    const notes = s.querySelector(".notes");
    const time = `<span class="p-time">${mmss(ends[cur] - durations[cur])} → ${mmss(ends[cur])} · ${durations[cur]} s</span>`;
    P.notes.innerHTML = time + (notes ? notes.innerHTML : "<p>(pas de notes)</p>");
    P.next.replaceChildren();
    const nxt = slides[cur + 1];
    if (nxt) {
      const c = nxt.cloneNode(true);
      c.classList.add("is-active");
      P.next.appendChild(c);
    }
    tick();
    layout();
  }

  function tick() {
    if (!P) return;
    const t = timer.elapsed();
    const end = ends[cur];
    P.elapsed.textContent = mmss(t);
    P.toggle.textContent = timer.running ? "Pause" : t > 0 ? "Reprendre" : "Démarrer";
    P.fill.style.width = `${Math.min(100, (t / TOTAL) * 100)}%`;
    P.mark.style.left = `${(end / TOTAL) * 100}%`;
    let cls = "ok", txt = `Dans les temps · cible ${mmss(end)}`;
    if (t === 0) txt = `Prêt · cible ${mmss(end)}`;
    else if (t > end + 3) { cls = "late"; txt = `En retard de ${mmss(t - end)}`; }
    else if (t > end - 4) { cls = "warn"; txt = `À conclure · cible ${mmss(end)}`; }
    P.pace.className = `p-pace ${cls}`;
    P.pace.textContent = txt;
  }

  /* ---------- boot ---------- */

  if (isPresenter) {
    buildPresenter();
    show(0, { silent: true });
    send({ type: "hello" });
  } else {
    const fromHash = () => {
      const n = parseInt(location.hash.slice(1), 10);
      if (Number.isFinite(n) && n - 1 !== cur) show(n - 1);
    };
    const n = parseInt(location.hash.slice(1), 10);
    show(Number.isFinite(n) ? n - 1 : 0, { silent: true });
    addEventListener("hashchange", fromHash);
  }
  layout();
  document.fonts?.ready.then(layout);
})();
