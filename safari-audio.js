/* iPadOS/iOS mixes every stem into a single Web Audio output. */
class SafariStemAudio extends EventTarget {
  constructor(context) {
    super();
    this.context = context;
    this.gain = context.createGain();
    this.gain.connect(context.destination);
    this.buffer = null;
    this.source = null;
    this.error = null;
    this.offset = 0;
    this.startedAt = 0;
    this.rate = 1;
    this.level = 1;
    this.silent = false;
    this.url = '';
    this.pending = null;
    this.controller = null;
  }

  get src() { return this.url; }
  set src(value) {
    this.load();
    this.url = value;
    this.prepare().catch(() => {});
  }
  get duration() { return this.buffer ? this.buffer.duration : NaN; }
  get readyState() { return this.buffer ? 4 : 0; }
  get paused() { return !this.source; }
  get currentTime() {
    const elapsed = this.source ? (this.context.currentTime - this.startedAt) * this.rate : 0;
    return Math.min(this.offset + elapsed, this.buffer?.duration ?? Infinity);
  }
  set currentTime(value) {
    const playing = !this.paused;
    this.pause();
    this.offset = Math.max(0, Math.min(Number(value) || 0, this.buffer?.duration ?? Infinity));
    if (playing) this.start();
  }
  get playbackRate() { return this.rate; }
  set playbackRate(value) {
    const playing = !this.paused;
    this.pause();
    this.rate = Number(value) || 1;
    if (playing) this.start();
  }
  get volume() { return this.level; }
  set volume(value) { this.level = value; this.updateGain(); }
  get muted() { return this.silent; }
  set muted(value) { this.silent = value; this.updateGain(); }

  updateGain() {
    this.gain.gain.setValueAtTime(this.silent ? 0 : this.level, this.context.currentTime);
  }

  removeAttribute(name) {
    if (name === 'src') { this.url = ''; this.load(); }
  }

  load() {
    this.pause();
    this.controller?.abort();
    this.controller = null;
    this.pending = null;
    this.buffer = null;
    this.error = null;
    this.offset = 0;
  }

  prepare() {
    if (this.buffer) return Promise.resolve();
    if (this.pending) return this.pending;
    const controller = new AbortController();
    this.controller = controller;
    this.pending = (async () => {
      try {
        const response = await fetch(this.url, { signal: controller.signal });
        if (!response.ok) throw new Error('Não foi possível baixar a faixa de áudio.');
        const bytes = await response.arrayBuffer();
        if (controller.signal.aborted) throw new DOMException('Cancelado', 'AbortError');
        const buffer = await this.context.decodeAudioData(bytes);
        if (controller.signal.aborted) throw new DOMException('Cancelado', 'AbortError');
        this.buffer = buffer;
        this.dispatchEvent(new Event('durationchange'));
      } catch (error) {
        if (!controller.signal.aborted) {
          this.error = error;
          this.dispatchEvent(new Event('error'));
        }
        throw error;
      }
    })();
    return this.pending;
  }

  start() {
    if (!this.buffer || this.source) return;
    if (this.offset >= this.duration) this.offset = 0;
    const source = this.context.createBufferSource();
    source.buffer = this.buffer;
    source.playbackRate.value = this.rate;
    source.connect(this.gain);
    source.onended = () => {
      if (this.source !== source) return;
      this.source = null;
      this.offset = this.duration;
      source.disconnect();
      this.dispatchEvent(new Event('ended'));
    };
    this.startedAt = this.context.currentTime;
    this.source = source;
    source.start(0, this.offset);
  }

  async play() {
    await this.prepare();
    if (this.context.state !== 'running') throw new Error('Toque em Tocar para ativar o áudio.');
    this.start();
  }

  pause() {
    if (!this.source) return;
    this.offset = this.currentTime;
    const source = this.source;
    this.source = null;
    source.onended = null;
    source.stop();
    source.disconnect();
  }
}

function createStemPlayback() {
  const appleTouch = /iPad|iPhone|iPod/.test(navigator.userAgent)
    || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!appleTouch || !AudioContextClass) return null;
  const context = new AudioContextClass();
  try {
    if (navigator.audioSession) navigator.audioSession.type = 'playback';
  } catch {
    // Older iPadOS versions manage the audio session automatically.
  }
  return {
    createAudio: () => new SafariStemAudio(context),
    async prepare(audios) {
      // Resume inside the original tap, before waiting for any network or decoding work.
      const resumed = context.resume();
      await Promise.allSettled(audios.map(audio => audio.prepare()));
      await resumed;
    }
  };
}
