/* The playground page: the microphone (or an audio file) to the model over a WebSocket,
   each question a row whose answer is drawn as it arrives. The protocol is README.md
   beside the server. Nothing here stores audio. */
(() => {
  'use strict';

  const SR = 16000;
  const WINDOW_S = 30;
  const KEEP_S = 10 * 60;
  const MAX_TYPED = 6;
  const ENROLL_S = 5;
  // Raise the version when the defaults change, so every visitor starts from them once.
  const STORE = 'crele.playground.questions.v4';
  const UI_FONT = '"Avenir Next", "Avenir", "Helvetica Neue", sans-serif';
  // A fixed palette, in order: the two voice
  // signals take the first two, typed questions the rest, one each while they live.
  const COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'];
  const SIGNALS = [
    { id: 'user', text: 'You are speaking', color: COLORS[0] },
    { id: 'other', text: 'Someone else is speaking', color: COLORS[1] },
  ];

  // The tap: every input mixed to mono, at 16 kHz, posted in 80 ms chunks.
  const WORKLET = `
class CreleTap extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = 1280; this.buf = new Float32Array(this.size); this.n = 0;
    this.ratio = sampleRate / 16000; this.acc = 0; this.cnt = 0; this.pos = 0;
    this.port.onmessage = (e) => {
      if (e.data !== 'flush') return;
      const rest = this.buf.slice(0, this.n); this.n = 0;
      this.port.postMessage({ flushed: true, pcm: rest }, [rest.buffer]);
    };
  }
  emit(x) {
    this.buf[this.n++] = x;
    if (this.n === this.size) {
      const chunk = this.buf; this.buf = new Float32Array(this.size); this.n = 0;
      this.port.postMessage({ pcm: chunk }, [chunk.buffer]);
    }
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const len = input[0].length, k = input.length;
    for (let i = 0; i < len; i++) {
      let x = 0;
      for (let c = 0; c < k; c++) x += input[c][i];
      x /= k;
      if (this.ratio === 1) { this.emit(x); continue; }
      this.acc += x; this.cnt += 1; this.pos += 1;   // another rate: a box average
      while (this.pos >= this.ratio) { this.emit(this.acc / this.cnt); this.acc = 0; this.cnt = 0; this.pos -= this.ratio; }
    }
    return true;
  }
}
registerProcessor('crele-tap', CreleTap);
`;

  const S = {
    ws: null, conn: 'wait', server: null, firstReady: true,
    typed: [], nextId: 1, signalDials: { user: {}, other: {} }, open: new Set(),
    live: false, stopping: false, cap: null, source: null, fileName: null, fileDur: 0,
    samplesSent: 0, lastChunkWall: 0, frozenEnd: 0,
    lanes: new Map(), input: [],
    enroll: { recording: false, left: 0, pcm: null },
  };

  const $ = (sel) => document.querySelector(sel);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
  const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

  function toast(text, err = false, ms = 5000) {
    const t = document.createElement('div');
    t.className = `toast${err ? ' err' : ''}`;
    t.textContent = text;
    $('#toasts').append(t);
    setTimeout(() => t.remove(), ms);
  }

  // ------------------------------------------------------------------ questions

  // The event settings, by the names voice-activity detectors use: the answer turns yes
  // when above `threshold` for `start_secs`, and back to no when below `stop_threshold`
  // for `stop_secs`.
  const KNOBS = ['threshold', 'start_secs', 'stop_threshold', 'stop_secs'];
  const KNOB_DEFAULT = { threshold: 0.7, start_secs: 0.12, stop_threshold: 0.35, stop_secs: 0.3 };
  function pickDials(o) {
    const d = {};
    for (const k of KNOBS) if (Number.isFinite(o?.[k])) d[k] = o[k];
    return d;
  }
  function save() {
    try {
      localStorage.setItem(STORE, JSON.stringify({
        questions: S.typed.map((q) => ({ text: q.text, ...q.dials })),
        signals: S.signalDials,
      }));
    } catch (e) { /* private mode */ }
  }
  function load() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORE) || 'null');
      // An earlier visit kept a list of texts; this one keeps the dials beside them.
      const list = Array.isArray(saved) ? saved : saved?.questions;
      if (Array.isArray(list)) {
        if (!Array.isArray(saved)) {
          S.signalDials = { user: pickDials(saved.signals?.user), other: pickDials(saved.signals?.other) };
        }
        return list.map((e) => (typeof e === 'string' ? { text: e } : e))
          .filter((e) => e && typeof e.text === 'string' && e.text.trim());
      }
    } catch (e) { /* the defaults */ }
    return JSON.parse($('#defaults').textContent).map((text) => ({ text }));
  }
  function newQuestion(entry) {
    const used = new Set(S.typed.map((q) => q.color));
    const color = COLORS.slice(2).find((c) => !used.has(c)) || COLORS[2];
    return { id: `q${S.nextId++}`, text: entry.text.trim(), color, dials: pickDials(entry) };
  }
  function addQuestion(text) {
    text = text.trim();
    if (!text) return false;
    if (S.typed.length >= MAX_TYPED) { toast('Six questions at a time. Remove one to ask another.', true); return false; }
    S.typed.push(newQuestion({ text }));
    changed();
    return true;
  }
  function removeQuestion(id) {
    S.typed = S.typed.filter((q) => q.id !== id);
    changed();
  }
  function changed() {
    save();
    renderRows();
    sendQuestions();
  }
  function sendQuestions() {
    send({
      type: 'questions',
      questions: S.typed.map((q) => ({ id: q.id, text: q.text, ...q.dials })),
      signals: S.signalDials,
    });
  }

  // ------------------------------------------------------------------ the rows

  function lane(id) {
    if (!S.lanes.has(id)) S.lanes.set(id, { ts: [], ps: [], events: [], state: 'no', p: null, lastT: 0 });
    return S.lanes.get(id);
  }
  function renderRows() {
    const all = [...SIGNALS, ...S.typed];
    for (const id of [...S.lanes.keys()]) if (!all.some((q) => q.id === id)) S.lanes.delete(id);
    for (const id of [...S.open]) if (!all.some((q) => q.id === id)) S.open.delete(id);
    const rows = [rowEl({ id: 'audio', text: 'Sound in' }, 'audio')];
    for (const q of all) {
      const l = lane(q.id);
      Object.assign(l, { id: q.id, color: q.color, text: q.text });
      l.row = rowEl(q, q.id === 'user' || q.id === 'other' ? 'signal' : 'typed');
      rows.push(l.row);
      paintRow(l);
      paintKnobs(l);
    }
    $('#rows').replaceChildren(...rows);
    renderEnroll();
    const full = S.typed.length >= MAX_TYPED;
    $('#ask-text').disabled = full;
    $('#ask-text').placeholder = full
      ? 'Six questions at a time. Remove one to ask another.'
      : 'Ask another question, e.g. Is a dog barking?';
    layout();
  }
  const TUNE_ICON = '<svg viewBox="0 0 24 24"><path d="M4 7h10M18 7h2M4 17h4M12 17h8"/>' +
    '<circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/></svg>';
  function rowEl(q, kind) {
    const row = document.createElement('div');
    row.className = `row ${kind}`;
    row.dataset.id = q.id;
    row.style.setProperty('--c', q.color || '#888');
    // Criteria typed as "question || yes means || no means": the row shows the question.
    const shown = q.text.split('||')[0].trim();
    const tune = kind === 'audio' ? '' : `<button class="tune-btn" type="button" title="Tune when this answer turns yes and no"
      aria-label="Tune: ${esc(shown)}">${TUNE_ICON}</button>`;
    row.innerHTML = `<div class="row-main">
        <div class="q"><div class="q-body"><span class="q-text" title="${esc(q.text)}">${esc(shown)}</span>
          ${q.id === 'user' ? '<span class="q-note" id="enroll-note"></span>' : ''}</div>${tune}</div>
        <div></div><div class="ans"></div></div>`;
    if (kind === 'typed') {
      const b = document.createElement('button');
      b.className = 'remove';
      b.type = 'button';
      b.title = 'Remove';
      b.setAttribute('aria-label', `Remove: ${q.text}`);
      b.innerHTML = '<svg viewBox="0 0 24 24"><path d="M6 6l12 12M18 6 6 18"/></svg>';
      b.addEventListener('click', () => removeQuestion(q.id));
      row.querySelector('.row-main').append(b);
    }
    if (kind !== 'audio') {
      row.append(knobsEl(q.id));
      row.querySelector('.tune-btn').addEventListener('click', () => toggleKnobs(q.id));
    }
    return row;
  }

  // The four settings of a question, as two sentences: the event starts when the answer
  // rises above a level for a while, and stops when it falls below a lower level for a
  // while. Unset, the question runs with the defaults the server reports.
  function knobsEl(id) {
    const box = document.createElement('div');
    box.className = 'tune';
    box.hidden = !S.open.has(id);
    box.innerHTML = `
      <div class="knob-line"><b>Starts</b>
        <label class="knob" data-k="threshold" title="threshold: the event starts when the probability rises above this.">
          <span>when above</span><input type="range" min="0.3" max="0.95" step="0.05"><output></output></label>
        <label class="knob" data-k="start_secs" title="start_secs: how long the probability must stay up before the event starts. Longer ignores brief blips.">
          <span>for</span><input type="range" min="0.04" max="1" step="0.04"><output></output></label></div>
      <div class="knob-line"><b>Stops</b>
        <label class="knob" data-k="stop_threshold" title="stop_threshold: the event stops when the probability falls below this. If the answer rests above it, the event never stops.">
          <span>when below</span><input type="range" min="0.05" max="0.9" step="0.05"><output></output></label>
        <label class="knob" data-k="stop_secs" title="stop_secs: how long the probability must stay low before the event stops. Longer holds through short pauses.">
          <span>for</span><input type="range" min="0.1" max="3" step="0.1"><output></output></label></div>
      <button class="reset" type="button">Reset</button>`;
    for (const knob of box.querySelectorAll('.knob')) {
      const k = knob.dataset.k;
      const input = knob.querySelector('input');
      input.addEventListener('input', () => showKnob(knob, Number(input.value)));
      input.addEventListener('change', () => {
        const d = dialsOf(id);
        d[k] = Number(input.value);
        // The stop threshold stays under the threshold, as the server requires: when either
        // move would cross them, the stop threshold is lowered to just under.
        const on = knobValue(id, 'threshold');
        if (knobValue(id, 'stop_threshold') > on - 0.05) {
          d.stop_threshold = Math.max(0.05, Math.round((on - 0.05) * 100) / 100);
        }
        retuned(id);
      });
    }
    box.querySelector('.reset').addEventListener('click', () => {
      for (const k of KNOBS) delete dialsOf(id)[k];
      retuned(id);
    });
    return box;
  }
  function dialsOf(id) {
    if (S.signalDials[id]) return S.signalDials[id];
    return S.typed.find((q) => q.id === id).dials;
  }
  // What a question runs with: its own dial, else what the server says it runs with.
  function knobValue(id, k) {
    const own = dialsOf(id)[k];
    if (Number.isFinite(own)) return own;
    const q = S.server?.questions.find((x) => x.id === id);
    return Number.isFinite(q?.[k]) ? q[k] : KNOB_DEFAULT[k];
  }
  function showKnob(knob, v) {
    const input = knob.querySelector('input');
    const k = knob.dataset.k;
    knob.querySelector('output').textContent = k === 'threshold' || k === 'stop_threshold' ? v.toFixed(2)
      : k === 'start_secs' ? `${v.toFixed(2)} s` : `${v.toFixed(1)} s`;
    input.style.setProperty('--pct', `${((v - input.min) / (input.max - input.min)) * 100}%`);
  }
  function paintKnobs(l) {
    if (!l.row) return;
    const tuned = KNOBS.some((k) => Number.isFinite(dialsOf(l.id)[k]));
    l.row.classList.toggle('tuned', tuned);
    l.row.classList.toggle('open', S.open.has(l.id));
    const box = l.row.querySelector('.tune');
    box.hidden = !S.open.has(l.id);
    box.querySelector('.reset').hidden = !tuned;
    for (const knob of box.querySelectorAll('.knob')) {
      const input = knob.querySelector('input');
      if (document.activeElement === input) continue;
      const v = knobValue(l.id, knob.dataset.k);
      input.value = String(v);
      showKnob(knob, Number(input.value));
    }
    l.threshold = knobValue(l.id, 'threshold');
    l.off = knobValue(l.id, 'stop_threshold');
  }
  function toggleKnobs(id) {
    if (S.open.has(id)) S.open.delete(id); else S.open.add(id);
    paintKnobs(S.lanes.get(id));
  }
  function retuned(id) {
    save();
    sendQuestions();
    paintKnobs(S.lanes.get(id));
  }
  function paintRow(l) {
    if (!l.row) return;
    l.row.classList.toggle('on', l.state === 'yes');
    l.row.classList.toggle('held', l.state === 'cant_tell');
    l.row.querySelector('.ans').textContent =
      l.p == null ? '–' : l.state === 'yes' ? 'Yes' : l.state === 'cant_tell' ? "Can't tell" : 'No';
  }

  // The "You" row needs to have heard your voice once.
  function renderEnroll() {
    const note = $('#enroll-note');
    if (!note) return;
    if (S.enroll.recording) {
      note.textContent = `Keep talking… ${S.enroll.left} s`;
    } else if (S.server?.enrolled) {
      note.innerHTML = 'Knows your voice · <button type="button" data-act="clear">forget it</button>';
    } else {
      note.innerHTML = '<button type="button" data-act="rec" title="Records 5 s from the microphone. ' +
        'Or drop a recording of your voice on this row.">Teach it your voice</button> (5 s)';
    }
    const b = note.querySelector('button');
    if (b) {
      b.disabled = S.conn !== 'up' || (S.live && S.source === 'file');
      b.addEventListener('click', () => {
        if (b.dataset.act === 'rec') recordEnrollment();
        else { S.enroll.pcm = null; send({ type: 'enroll', pcm: null }); }
      });
    }
  }

  // ------------------------------------------------------------------ the connection

  function send(msg) {
    if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(msg));
  }
  async function connect() {
    let port = 8781;
    try { port = (await (await fetch('config.json', { cache: 'no-store' })).json()).ws_port || port; } catch (e) { /* default */ }
    const ws = new WebSocket(`ws://${location.hostname}:${port}`);
    ws.binaryType = 'arraybuffer';
    S.ws = ws;
    S.firstReady = true;
    ws.onopen = () => sendQuestions();
    ws.onmessage = (e) => {
      if (S.ws !== ws || typeof e.data !== 'string') return;
      const m = JSON.parse(e.data);
      (ON[m.type] || (() => {}))(m);
    };
    ws.onclose = (e) => {
      if (S.ws !== ws) return;
      S.ws = null;
      if (S.live) endCapture();
      if (e.code === 4000) { setConn('taken'); return; }
      setConn('down');
      setTimeout(connect, 1500);
    };
  }
  function setConn(c) {
    S.conn = c;
    $('#mic').disabled = c !== 'up';
    renderCaption();
    renderEnroll();
    renderFoot();
  }

  const ON = {
    ready(m) {
      const first = S.firstReady;
      S.firstReady = false;
      S.server = m;
      if (S.conn !== 'up') setConn('up');
      if (m.state === 'idle' && S.live && !S.stopping) endCapture();
      // A restarted server lost the voice it had learned: teach it again.
      if (first && !m.enrolled && S.enroll.pcm) send({ type: 'enroll', pcm: b64(S.enroll.pcm) });
      for (const l of S.lanes.values()) paintKnobs(l);
      renderEnroll();
      renderFoot();
    },
    decisions(m) {
      for (const step of m.steps) {
        for (const a of step.answers) {
          const l = S.lanes.get(a.id);
          if (!l) continue;
          l.ts.push(step.t_s);
          l.ps.push(a.p ?? 0);
          l.lastT = step.t_s;
          if (l.state !== a.state || l.p == null) { l.state = a.state; l.p = a.p; paintRow(l); }
          l.p = a.p;
        }
      }
      for (const l of S.lanes.values()) {
        if (l.ts.length > KEEP_S * 25 + 500) { l.ts.splice(0, 500); l.ps.splice(0, 500); }
      }
    },
    event(m) {
      const l = S.lanes.get(m.question);
      if (!l) return;
      if (m.event === 'started') l.events.push({ start: m.start_s, end: null, born: performance.now() });
      else {
        const ev = [...l.events].reverse().find((e) => e.end === null);
        if (ev) ev.end = m.end_s;
      }
    },
    error(m) {
      toast(m.message, true, 7000);
      if (m.where === 'questions' && S.server) {
        const known = new Set(S.server.questions.map((q) => q.id));
        S.typed = S.typed.filter((q) => known.has(q.id));
        save();
        renderRows();
      }
    },
  };

  // ------------------------------------------------------------------ audio

  let actx = null;
  async function audioContext() {
    if (!actx) {
      try { actx = new AudioContext({ sampleRate: SR, latencyHint: 'interactive' }); }
      catch (e) { actx = new AudioContext({ latencyHint: 'interactive' }); }
      const url = URL.createObjectURL(new Blob([WORKLET], { type: 'text/javascript' }));
      await actx.audioWorklet.addModule(url);
      URL.revokeObjectURL(url);
    }
    if (actx.state !== 'running') await Promise.race([actx.resume(), new Promise((r) => setTimeout(r, 250))]);
    // No click on the page yet (a file dropped on a fresh page): ask for one.
    if (actx.state !== 'running') {
      await new Promise((resolve) => {
        $('#gesture').hidden = false;
        $('#gesture-go').onclick = async () => { await actx.resume(); $('#gesture').hidden = true; resolve(); };
      });
    }
    return actx;
  }
  function makeTap(onPcm) {
    const node = new AudioWorkletNode(actx, 'crele-tap', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const mute = actx.createGain();
    mute.gain.value = 0;
    node.connect(mute).connect(actx.destination);   // a silent path keeps the node running
    let flushed = null;
    node.port.onmessage = (e) => {
      if (e.data.flushed) { if (flushed) flushed(e.data.pcm); flushed = null; return; }
      onPcm(e.data.pcm);
    };
    return {
      node,
      flush: () => new Promise((resolve) => {
        flushed = resolve;
        node.port.postMessage('flush');
        setTimeout(() => { if (flushed === resolve) { flushed = null; resolve(new Float32Array(0)); } }, 400);
      }),
      close: () => { node.port.onmessage = null; node.disconnect(); mute.disconnect(); },
    };
  }
  function getMic() {
    // Raw input: echo cancellation, noise suppression and gain control
    // would remove the sounds the questions ask about.
    return navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false },
    });
  }
  function micError(e) {
    toast(e && e.name === 'NotAllowedError'
      ? 'The browser blocked the microphone. Allow it in the address bar and press the button again.'
      : `The microphone did not start: ${e?.message || e}`, true, 8000);
  }

  function begin(source, name, dur) {
    for (const l of S.lanes.values()) { l.ts = []; l.ps = []; l.events = []; l.state = 'no'; l.p = null; l.lastT = 0; paintRow(l); }
    S.input = [];
    Object.assign(S, { live: true, stopping: false, source, fileName: name || null, fileDur: dur || 0,
      samplesSent: 0, lastChunkWall: performance.now() });
    document.body.classList.add('live');
    renderCaption();
    renderEnroll();
  }
  function sendPcm(pcm) {
    if (!pcm || !pcm.length || !S.live || !S.ws || S.ws.readyState !== WebSocket.OPEN) return;
    S.ws.send(pcm.buffer.byteLength === pcm.byteLength ? pcm.buffer : pcm.slice().buffer);
    let sum = 0;
    for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
    const level = clamp((20 * Math.log10(Math.sqrt(sum / pcm.length) + 1e-9) + 66) / 60, 0, 1);
    S.samplesSent += pcm.length;
    S.lastChunkWall = performance.now();
    S.input.push({ t: S.samplesSent / SR, d: pcm.length / SR, level });
    if (S.input.length > WINDOW_S * 13 + 200) S.input.splice(0, 200);
    $('#mic').style.setProperty('--level', level.toFixed(3));
  }
  async function startMic() {
    if (S.conn !== 'up' || S.live) return;
    let stream;
    try { await audioContext(); stream = await getMic(); } catch (e) { micError(e); return; }
    begin('mic');
    send({ type: 'start', source: 'mic' });
    const src = actx.createMediaStreamSource(stream);
    const tap = makeTap(sendPcm);
    src.connect(tap.node);
    S.cap = { stream, src, tap };
  }
  async function playFile(file) {
    if (S.conn !== 'up') return;
    let buf;
    try { await audioContext(); buf = await actx.decodeAudioData(await file.arrayBuffer()); }
    catch (e) { toast(`Can't read ${file.name} as audio.`, true); return; }
    if (S.live) await stop();
    begin('file', file.name, buf.duration);
    send({ type: 'wav', name: file.name, duration_s: buf.duration });
    const node = actx.createBufferSource();
    node.buffer = buf;
    const tap = makeTap(sendPcm);
    node.connect(actx.destination);
    node.connect(tap.node);
    node.onended = () => { if (S.cap && S.cap.node === node) stop('end of file'); };
    S.cap = { node, tap };
    node.start();
  }
  function release(cap) {
    if (cap.stream) { cap.src.disconnect(); cap.stream.getTracks().forEach((t) => t.stop()); }
    if (cap.node) { cap.node.onended = null; try { cap.node.stop(); } catch (e) { /* ended */ } cap.node.disconnect(); }
  }
  // Stop the source, send what the tap still holds, then `stop`.
  async function stop(reason = 'stop') {
    if (!S.live || S.stopping) return;
    S.stopping = true;
    const cap = S.cap;
    S.cap = null;
    if (cap) {
      release(cap);
      sendPcm(await cap.tap.flush());
      cap.tap.close();
    }
    send({ type: 'stop', reason });
    finish();
  }
  // The server ended the stream, or went away: stop capturing, send nothing.
  function endCapture() {
    const cap = S.cap;
    S.cap = null;
    if (cap) { release(cap); cap.tap.close(); }
    finish();
  }
  function finish() {
    S.live = false;
    S.stopping = false;
    S.frozenEnd = S.samplesSent / SR;
    document.body.classList.remove('live');
    $('#mic').style.setProperty('--level', '0');
    renderCaption();
    renderEnroll();
  }

  // Five seconds of the user's voice, for the "You" row.
  function b64(f32) {
    const u8 = new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength);
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
    return btoa(s);
  }
  async function recordEnrollment() {
    if (S.enroll.recording) return;
    let stream;
    try { await audioContext(); stream = await getMic(); } catch (e) { micError(e); return; }
    const need = ENROLL_S * SR;
    const pcm = new Float32Array(need);
    let n = 0;
    S.enroll.recording = true;
    S.enroll.left = ENROLL_S;
    renderEnroll();
    const src = actx.createMediaStreamSource(stream);
    const tap = makeTap((chunk) => {
      if (!S.enroll.recording) return;
      pcm.set(chunk.subarray(0, Math.min(chunk.length, need - n)), n);
      n += chunk.length;
      const left = Math.max(0, Math.ceil((need - n) / SR));
      if (left !== S.enroll.left) { S.enroll.left = left; renderEnroll(); }
      if (n < need) return;
      S.enroll.recording = false;
      src.disconnect();
      tap.close();
      stream.getTracks().forEach((t) => t.stop());
      let peak = 0;
      for (let i = 0; i < need; i++) peak = Math.max(peak, Math.abs(pcm[i]));
      if (peak < 1e-3) { toast('That was silent. Check the microphone and try again.', true); renderEnroll(); return; }
      S.enroll.pcm = pcm;
      send({ type: 'enroll', pcm: b64(pcm) });
      renderEnroll();
    });
    src.connect(tap.node);
  }

  // The same from a recording dropped on the "You" row: up to 30 s, at 16 kHz.
  async function enrollFromFile(file) {
    try {
      const off = new OfflineAudioContext(1, SR, SR);
      const buf = await off.decodeAudioData(await file.arrayBuffer());
      const n = Math.min(buf.length, 30 * SR);
      const pcm = new Float32Array(n);
      for (let c = 0; c < buf.numberOfChannels; c++) {
        const d = buf.getChannelData(c);
        for (let i = 0; i < n; i++) pcm[i] += d[i] / buf.numberOfChannels;
      }
      S.enroll.pcm = pcm;
      send({ type: 'enroll', pcm: b64(pcm) });
      toast(`Learning the voice in ${file.name}.`, false, 3000);
    } catch (e) {
      toast(`Can't read ${file.name} as audio.`, true);
    }
  }

  // ------------------------------------------------------------------ the words around it

  function renderCaption() {
    const c = $('#caption');
    if (S.live) c.textContent = S.source === 'file' ? `Playing ${S.fileName}` : 'Listening';
    else if (S.conn === 'up') c.textContent = 'Start listening';
    else if (S.conn === 'wait') c.textContent = 'Connecting to the model…';
    else if (S.conn === 'taken') c.textContent = 'Another tab has the model. Reload this page to take it back.';
    else c.textContent = 'The model is not running: python -m crele_nimbus.playground';
    $('#mic').setAttribute('aria-label', S.live ? 'Stop' : 'Start listening');
    $('#file-link').hidden = S.live || S.conn !== 'up';
  }
  function tickCaption() {
    if (!S.live) return;
    const t = S.samplesSent / SR;
    const what = S.source === 'file' ? `Playing ${S.fileName}` : 'Listening';
    const time = S.source === 'file' && S.fileDur ? `${fmt(t)} / ${fmt(S.fileDur)}` : fmt(t);
    const text = `${what} · ${time}`;
    if ($('#caption').textContent !== text) $('#caption').textContent = text;
  }
  function renderFoot() {
    const m = S.server?.model;
    const f = $('#foot');
    if (!m || S.conn !== 'up') { f.textContent = ''; return; }
    f.textContent = m.name;
  }

  // ------------------------------------------------------------------ the timeline

  const canvas = $('#tl');
  const g = canvas.getContext('2d');
  let W = 0, H = 0, DPR = 1, ROW = 64, AUDIO = 44;
  const tops = new Map();

  function layout() {
    const rows = [...document.querySelectorAll('#rows .row')];
    const css = getComputedStyle(document.documentElement);
    W = canvas.parentElement.clientWidth - parseFloat(css.getPropertyValue('--q-w')) - parseFloat(css.getPropertyValue('--a-w'));
    H = rows.reduce((a, r) => a + r.offsetHeight, 0);
    tops.clear();
    for (const r of rows) tops.set(r.dataset.id, r.offsetTop);
    AUDIO = rows[0] ? rows[0].querySelector('.row-main').offsetHeight : 44;
    ROW = rows[1] ? rows[1].querySelector('.row-main').offsetHeight : 64;
    DPR = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(W * DPR));
    canvas.height = Math.max(1, Math.round(H * DPR));
    canvas.style.height = `${H}px`;
  }
  function viewEnd() {
    if (!S.live) return S.frozenEnd;
    return S.samplesSent / SR + clamp((performance.now() - S.lastChunkWall) / 1000, 0, 0.1);
  }
  function alpha(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
  }

  function draw() {
    requestAnimationFrame(draw);
    tickCaption();
    if (W <= 0 || H <= 0) return;
    const end = viewEnd();
    const t0 = end - WINDOW_S;
    const pps = W / WINDOW_S;
    const X = (t) => W - (end - t) * pps;
    g.setTransform(DPR, 0, 0, DPR, 0, 0);
    g.clearRect(0, 0, W, H);

    // The sound coming in: the level of each 80 ms, mirrored.
    g.fillStyle = 'rgba(195,194,183,0.45)';
    for (const a of S.input) {
      if (a.t < t0) continue;
      const h = Math.max(0.75, a.level * (AUDIO / 2 - 8));
      g.fillRect(X(a.t - a.d) + 0.3, AUDIO / 2 - h, Math.max(0.8, a.d * pps - 0.6), h * 2);
    }

    // Each question: a bar while the answer is yes, a faint line for the probability.
    const ids = [...SIGNALS.map((q) => q.id), ...S.typed.map((q) => q.id)];
    ids.forEach((id) => {
      const l = S.lanes.get(id);
      const top = tops.get(id);
      if (!l || top === undefined) return;
      const barY = top + 9;
      for (const ev of l.events) {
        const e = ev.end ?? l.lastT;
        if (e < t0) continue;
        const xa = X(ev.start);
        const xb = Math.max(X(e), xa + 3);
        const age = performance.now() - ev.born;
        g.save();
        if (age < 800) { g.shadowColor = alpha(l.color, 0.8 * (1 - age / 800)); g.shadowBlur = 16; }
        g.fillStyle = l.color;
        g.beginPath();
        g.roundRect(xa, barY, xb - xa, 26, Math.min(7, (xb - xa) / 2));
        g.fill();
        g.restore();
      }
      // A thin strip under the bars: 0 at its bottom, 1 at its top. While the row's
      // settings are open, the yes level and the fainter no level are drawn across it.
      const y = (p) => top + ROW - 6 - clamp(p, 0, 1) * (ROW - 44);
      if (S.open.has(id)) {
        g.setLineDash([3, 3]);
        g.lineWidth = 1;
        for (const [level, a] of [[l.threshold, 0.5], [l.off, 0.22]]) {
          g.strokeStyle = `rgba(240,239,236,${a})`;
          g.beginPath();
          g.moveTo(0, Math.round(y(level)) + 0.5);
          g.lineTo(W, Math.round(y(level)) + 0.5);
          g.stroke();
        }
        g.setLineDash([]);
      }
      const n = l.ts.length;
      if (n > 1) {
        let k = n - 1;
        while (k > 0 && l.ts[k - 1] >= t0 - 0.1) k--;
        g.beginPath();
        for (let j = k; j < n; j++) {
          if (j === k) g.moveTo(X(l.ts[j]), y(l.ps[j])); else g.lineTo(X(l.ts[j]), y(l.ps[j]));
        }
        g.strokeStyle = alpha(l.color, 0.55);
        g.lineWidth = 1.25;
        g.lineJoin = 'round';
        g.stroke();
      }
    });

    // Now, at the right edge of each line (not across the knobs).
    g.fillStyle = S.live ? 'rgba(229,83,75,0.85)' : 'rgba(240,239,236,0.18)';
    g.fillRect(W - 1, 0, 1, AUDIO);
    for (const id of ids) if (tops.has(id)) g.fillRect(W - 1, tops.get(id), 1, ROW);
  }

  // ------------------------------------------------------------------ wiring

  function wire() {
    $('#mic').addEventListener('click', () => (S.live ? stop() : startMic()));
    $('#file-link').addEventListener('click', () => $('#file-input').click());
    $('#file-input').addEventListener('change', (e) => {
      const f = e.target.files[0];
      e.target.value = '';
      if (f) playFile(f);
    });
    $('#ask').addEventListener('submit', (e) => {
      e.preventDefault();
      if (addQuestion($('#ask-text').value)) $('#ask-text').value = '';
    });
    // Space starts and stops, unless the question box has the keys.
    document.addEventListener('keydown', (e) => {
      if (e.code !== 'Space' || e.repeat || S.conn !== 'up') return;
      const a = document.activeElement;
      if (a && (a.tagName === 'INPUT' || a.tagName === 'BUTTON')) return;
      e.preventDefault();
      if (S.live) stop(); else startMic();
    });
    // Drop an audio file anywhere to play it to the model.
    let depth = 0;
    window.addEventListener('dragenter', (e) => {
      if (![...(e.dataTransfer?.types || [])].includes('Files')) return;
      depth += 1;
      $('#drop').hidden = false;
    });
    const onYou = (e) => Boolean(e.target.closest && e.target.closest('.row[data-id="user"]'));
    window.addEventListener('dragover', (e) => {
      e.preventDefault();
      $('#drop p').textContent = onYou(e) ? 'Drop it to teach the model this voice as yours' : 'Drop it to play it to the model';
    });
    window.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) $('#drop').hidden = true; });
    window.addEventListener('drop', (e) => {
      e.preventDefault();
      depth = 0;
      $('#drop').hidden = true;
      const f = e.dataTransfer?.files?.[0];
      if (!f || S.conn !== 'up') return;
      if (onYou(e)) enrollFromFile(f); else playFile(f);
    });
    new ResizeObserver(layout).observe($('#lanes'));
  }

  for (const entry of load().slice(0, MAX_TYPED)) S.typed.push(newQuestion(entry));
  wire();
  renderRows();
  renderCaption();
  requestAnimationFrame(draw);
  connect();

  window.crelePlayground = { state: S, addQuestion, stop, playFile, enrollFromFile };
})();
