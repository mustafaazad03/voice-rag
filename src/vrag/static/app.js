/* Voxcite console.
   No framework: one page, three files, no build step. Everything talks to the
   JSON API documented in the README. */

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );

const EXAMPLES = [
  "What is a corporation?",
  "Boiling point of water",
  "কৰ্পোৰেচন কি?",
  "How does incorporation work?",
];

/* --------------------------------------------------------------------------- *
   Chrome
 * --------------------------------------------------------------------------- */
function buildExamples() {
  const host = $("examples");
  EXAMPLES.forEach((text) => {
    const b = document.createElement("button");
    b.className = "btn btn--tiny";
    b.type = "button";
    b.textContent = text;
    b.onclick = () => {
      $("q").value = text;
      ask();
    };
    host.appendChild(b);
  });
}

function status(text, { busy = false, bad = false } = {}) {
  $("status").innerHTML =
    (busy ? '<span class="spinner" aria-hidden="true"></span>' : "") +
    `<span${bad ? ' style="color:var(--pink)"' : ""}>${esc(text)}</span>`;
}

/* --------------------------------------------------------------------------- *
   Network
 * --------------------------------------------------------------------------- */
async function post(url, body, isForm) {
  const res = await fetch(
    url,
    isForm
      ? { method: "POST", body }
      : {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
  );
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data?.error?.message || `${res.status} ${res.statusText}`);
  return data;
}

/* --------------------------------------------------------------------------- *
   Answer rendering
 * --------------------------------------------------------------------------- */
const STATUS_COPY = {
  answered: "Answered",
  insufficient_evidence: "Not enough evidence",
  off_topic: "Outside the corpus",
  blocked: "Refused",
};

function render(r) {
  const budget = window.__budget || 200;

  const stats = [
    `<span class="stat ${r.latency_ms <= budget ? "stat--good" : "stat--warn"}">pipeline <b>${r.latency_ms.toFixed(1)} ms</b></span>`,
    r.total_ms != null ? `<span class="stat">with stt <b>${r.total_ms.toFixed(0)} ms</b></span>` : "",
    r.confidence ? `<span class="stat">confidence <b>${r.confidence.overall.toFixed(2)}</b></span>` : "",
    r.grounding
      ? `<span class="stat ${r.grounding.grounded ? "stat--good" : "stat--bad"}">grounding <b>${r.grounding.score.toFixed(2)}</b></span>`
      : "",
    r.cached ? `<span class="stat">cached</span>` : "",
    r.transcription
      ? `<span class="stat">${esc(r.transcription.provider)}${r.transcription.fallback_used ? " (fallback)" : ""} <b>${r.transcription.latency_ms.toFixed(0)} ms</b></span>`
      : "",
    ...(r.degraded || []).map((d) => `<span class="stat stat--warn">${esc(d)}</span>`),
  ]
    .filter(Boolean)
    .join("");

  const cites = (r.citations || [])
    .map(
      (c) => `<div class="cite">
        <span class="cite__marker">${esc(c.marker)}</span>
        <div>${esc(c.quote)}
          <span class="cite__meta">${esc(c.doc_id)} · score ${c.score}</span>
        </div>
      </div>`
    )
    .join("");

  const answerHtml = esc(r.answer).replace(/\[(\d+)\]/g, "<mark>[$1]</mark>");

  $("answer").innerHTML = `<div class="answer">
    <span class="answer__ribbon" data-status="${esc(r.status)}">
      ${esc(STATUS_COPY[r.status] || r.status)}
    </span>
    <p class="answer__text">${answerHtml}</p>
    ${stats ? `<div class="stats">${stats}</div>` : ""}
    ${
      r.transcription
        ? `<div class="heard">heard <b>“${esc(r.transcription.text)}”</b> · ${esc(r.transcription.language_code || "auto-detected")}</div>`
        : ""
    }
    ${cites ? `<div class="cites">${cites}</div>` : ""}
    ${timeline(r.stage_ms)}
  </div>`;

  status(`${STATUS_COPY[r.status] || r.status} in ${r.latency_ms.toFixed(1)} ms.`, {
    bad: r.status === "blocked",
  });
}

function timeline(spans) {
  const entries = Object.entries(spans || {}).sort((a, b) => b[1] - a[1]);
  if (!entries.length) return "";
  const max = entries[0][1] || 1;
  const bars = entries
    .map(
      ([name, ms]) => `<div class="bar">
        <span class="bar__name">${esc(name)}</span>
        <span class="bar__track"><span class="bar__fill ${ms === max ? "bar__fill--hot" : ""}"
              style="width:${Math.max(2, (ms / max) * 100).toFixed(1)}%"></span></span>
        <span class="bar__ms">${ms.toFixed(2)} ms</span>
      </div>`
    )
    .join("");
  return `<details class="timeline">
    <summary>Stage timings</summary>
    <div class="bars">${bars}</div>
  </details>`;
}

/* --------------------------------------------------------------------------- *
   Health + the live P50 chip
 * --------------------------------------------------------------------------- */
async function loadHealth() {
  try {
    const h = await (await fetch("/api/v1/health")).json();
    const chain = h.stt_chain || [];
    $("txt-stt").textContent = chain.length ? chain.join(" → ") : "no stt keys";
    $("dot-health").className = "dot " + (h.status === "ok" ? "dot--ok" : "dot--warn");
    if (h.index) {
      $("chip-index").textContent = `${h.index.size.toLocaleString()} chunks`;
      $("chip-strategy").textContent = h.index.strategy;
    }
    const cfg = await (await fetch("/api/v1/config")).json();
    window.__budget = cfg.budget_total_ms || 200;
    if (!chain.length) {
      $("mic").disabled = true;
      $("mic").title = "Set SARVAM_API_KEY or ELEVENLABS_API_KEY to enable voice";
    }
  } catch {
    $("txt-stt").textContent = "api unreachable";
    $("dot-health").className = "dot dot--warn";
  }
}

/* --------------------------------------------------------------------------- *
   Text query
 * --------------------------------------------------------------------------- */
async function ask() {
  const query = $("q").value.trim();
  if (!query) return;
  $("ask").disabled = true;
  status("Retrieving…", { busy: true });
  try {
    render(await post("/api/v1/query", { query }));
  } catch (e) {
    status(e.message, { bad: true });
  } finally {
    $("ask").disabled = false;
  }
}

/* --------------------------------------------------------------------------- *
   Voice — click to toggle, or push-to-talk on a held key (WhisperFlow style).
 * --------------------------------------------------------------------------- */
let recorder = null;
let chunks = [];
let audioCtx = null;
let rafId = 0;
let cancelled = false;
let starting = false;
let pendingStop = null;

function drawWave(analyser) {
  const canvas = $("wave");
  const ctx = canvas.getContext("2d");
  const buf = new Uint8Array(analyser.frequencyBinCount);
  const paint = () => {
    canvas.width = canvas.clientWidth;
    analyser.getByteTimeDomainData(buf);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ff0080";
    ctx.beginPath();
    const slice = canvas.width / buf.length;
    for (let i = 0; i < buf.length; i++) {
      const y = (buf[i] / 128) * (canvas.height / 2);
      i === 0 ? ctx.moveTo(0, y) : ctx.lineTo(i * slice, y);
    }
    ctx.stroke();
    rafId = requestAnimationFrame(paint);
  };
  paint();
}

function teardownAudio() {
  cancelAnimationFrame(rafId);
  $("wave").classList.remove("is-live");
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
}

function setRecordingUi(on) {
  $("mic").classList.toggle("is-recording", on);
  $("mic").setAttribute("aria-pressed", String(on));
  $("mic-label").textContent = on ? "Stop" : "Record";
  $("ptt").classList.toggle("is-live", on);
}

const isRecording = () => recorder && recorder.state === "recording";

async function startRecording() {
  if (isRecording() || starting) return;
  if ($("mic").disabled) {
    return status("Voice is off — set SARVAM_API_KEY or ELEVENLABS_API_KEY.", { bad: true });
  }
  starting = true;
  cancelled = false;

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    starting = false;
    return status("Microphone permission denied.", { bad: true });
  }

  chunks = [];
  recorder = new MediaRecorder(stream);
  recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);

  recorder.onstop = async () => {
    stream.getTracks().forEach((t) => t.stop());
    teardownAudio();
    setRecordingUi(false);

    if (cancelled) return status("Cancelled.");
    // A tap rather than a hold produces a few bytes of silence; the STT providers
    // reject that with a confusing error, so stop before we get there.
    const blob = new Blob(chunks, { type: recorder.mimeType });
    if (blob.size < 2000) return status("Too short — hold Space a little longer.");

    status("Transcribing…", { busy: true });
    const form = new FormData();
    form.append("audio", blob, "speech.webm");
    try {
      const r = await post("/api/v1/voice/query", form, true);
      $("q").value = r.transcription?.text || "";
      render(r);
    } catch (e) {
      status(e.message, { bad: true });
    }
  };

  // MediaRecorder.start() throws on some browser/codec combinations. Recover
  // rather than leaving the microphone open behind a half-updated UI.
  try {
    audioCtx = new AudioContext();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    $("wave").classList.add("is-live");
    drawWave(analyser);
    recorder.start();
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop());
    teardownAudio();
    recorder = null;
    starting = false;
    pendingStop = null;
    return status(`Could not start recording: ${e.name}`, { bad: true });
  }

  setRecordingUi(true);
  status("Listening… release to send.");
  starting = false;

  // getUserMedia is async, so a quick tap can release the key before we get
  // here. Honour the stop that arrived while we were starting up, otherwise the
  // microphone stays open with nobody holding the key.
  if (pendingStop) {
    const { discard } = pendingStop;
    pendingStop = null;
    stopRecording({ discard });
  }
}

function stopRecording({ discard = false } = {}) {
  if (starting) {
    pendingStop = { discard };
    return;
  }
  // Do not touch `cancelled` unless there is a live recording to cancel: the
  // keyup that follows an Esc would otherwise clear the flag and the discarded
  // audio would be uploaded anyway.
  if (!isRecording()) return;
  cancelled = discard;
  recorder.stop();
}

/* --------------------------------------------------------------------------- *
   Hotkeys
     hold Space   push to talk (when not typing in the field)
     hold Alt/⌥   push to talk from anywhere, including mid-typing
     ⌘K / Ctrl+K  focus the field
     ⌘⏎ / Ctrl+⏎  ask from anywhere
     Esc          cancel a recording, or blur the field
 * --------------------------------------------------------------------------- */
const typing = () => document.activeElement === $("q");

// keydown repeats while held; only the first one should start the recorder.
let heldKey = null;

document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    $("q").focus();
    $("q").select();
    return;
  }
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    ask();
    return;
  }
  if (e.key === "Escape") {
    if (isRecording() || starting) {
      heldKey = null;
      stopRecording({ discard: true });
    } else {
      $("q").blur();
    }
    return;
  }

  const pttSpace = e.code === "Space" && !typing() && !e.metaKey && !e.ctrlKey;
  const pttAlt = e.key === "Alt";
  if (!pttSpace && !pttAlt) return;
  // Space would otherwise scroll the page while held.
  e.preventDefault();
  if (heldKey || e.repeat) return;
  heldKey = pttAlt ? "Alt" : "Space";
  startRecording();
});

document.addEventListener("keyup", (e) => {
  const released = e.key === "Alt" ? "Alt" : e.code === "Space" ? "Space" : null;
  if (!released || released !== heldKey) return;
  heldKey = null;
  stopRecording();
});

// A lost window (alt-tab, or the OS eating the modifier) must not leave the
// microphone open forever.
window.addEventListener("blur", () => {
  if (heldKey) {
    heldKey = null;
    stopRecording();
  }
});

/* --------------------------------------------------------------------------- *
   Boot
 * --------------------------------------------------------------------------- */
buildExamples();
loadHealth();

$("ask").onclick = ask;
$("mic").onclick = () => (isRecording() ? stopRecording() : startRecording());
$("q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.metaKey && !e.ctrlKey) ask();
});
