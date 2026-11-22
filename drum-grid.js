/* Optional drum grid drawn from the same MusicXML notes as the staff. */
const DRUM_GRID_LANES = [
  { label: 'Cr', name: 'Crash / pratos', notes: [49, 52, 55, 57], symbol: 'x' },
  { label: 'Rd', name: 'Ride', notes: [51, 53, 59], symbol: 'x' },
  { label: 'Ch', name: 'Chimbal', notes: [42, 44, 46], symbol: 'x' },
  { label: 'Cx', name: 'Caixa', notes: [37, 38, 39, 40], symbol: 'o' },
  { label: 'Ta', name: 'Tom agudo', notes: [48, 50], symbol: 'o' },
  { label: 'Tm', name: 'Tom médio', notes: [45, 47], symbol: 'o' },
  { label: 'Su', name: 'Surdo', notes: [41, 43], symbol: 'o' },
  { label: 'Bu', name: 'Bumbo', notes: [35, 36], symbol: 'o' }
];

function parseDrumGrid(buffer) {
  const source = new TextDecoder().decode(buffer);
  if (!source.includes('<score-partwise')) return null;
  const xml = new DOMParser().parseFromString(source, 'application/xml');
  if (xml.querySelector('parsererror')) return null;

  // MusicXML uses one-based midi-unpitched values; our rows use GM note numbers.
  const instruments = new Map();
  for (const el of xml.querySelectorAll('midi-instrument')) {
    const number = Number(el.querySelector('midi-unpitched')?.textContent);
    if (el.id && Number.isInteger(number)) instruments.set(el.id, number - 1);
  }
  const parts = [...xml.querySelectorAll('part')];
  const part = parts.find(item => [...item.querySelectorAll('note > instrument')]
    .some(inst => instruments.has(inst.getAttribute('id')))) || parts[0];
  if (!part) return null;

  let divisions = 0;
  let beats = 4;
  let beatType = 4;
  const bars = [];
  for (const measure of [...part.children].filter(el => el.localName === 'measure')) {
    const attrs = [...measure.children].find(el => el.localName === 'attributes');
    if (attrs) {
      divisions = Number(attrs.querySelector('divisions')?.textContent) || divisions;
      beats = Number(attrs.querySelector('time > beats')?.textContent) || beats;
      beatType = Number(attrs.querySelector('time > beat-type')?.textContent) || beatType;
    }
    // The app currently aligns playback to 4/4. Reject other meters instead of showing a wrong cursor.
    if (!divisions || beats !== 4 || beatType !== 4) return null;
    const slots = Array.from({ length: 16 }, () => new Map());
    let cursor = 0;
    let onset = 0;
    for (const el of measure.children) {
      const duration = Number(el.querySelector('duration')?.textContent) || 0;
      if (el.localName === 'backup') { cursor -= duration; continue; }
      if (el.localName === 'forward') { cursor += duration; continue; }
      if (el.localName !== 'note') continue;
      const chord = !!el.querySelector('chord');
      if (!chord) onset = cursor;
      if (!el.querySelector('rest')) {
        const id = el.querySelector('instrument')?.getAttribute('id');
        const gm = instruments.get(id) ?? Number(id?.match(/-I(\d+)$/)?.[1]) - 1;
        const lane = DRUM_GRID_LANES.find(item => item.notes.includes(gm));
        const index = Math.round(onset * 4 / divisions);
        if (lane && index >= 0 && index < 16) {
          const ghost = lane.label === 'Cx' && el.querySelector('notehead')?.getAttribute('parentheses') === 'yes';
          const previous = slots[index].get(lane.label);
          if (!previous || !ghost) slots[index].set(lane.label, ghost ? '(o)' : lane.symbol);
        }
      }
      if (!chord) cursor += duration;
    }
    const lyric = [...measure.querySelectorAll('direction-type > words')]
      .map(el => el.textContent.trim()).filter(Boolean).join(' ');
    bars.push({ number: measure.getAttribute('number') || String(bars.length + 1), lyric, slots });
  }
  return bars.length ? bars : null;
}

function renderDrumGrid(container, bars) {
  container.replaceChildren();
  const ns = 'http://www.w3.org/2000/svg';
  const cards = [];
  function svgNode(tag, attrs = {}, content) {
    const node = document.createElementNS(ns, tag);
    for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
    if (content != null) node.textContent = content;
    return node;
  }
  const width = 410, firstX = 58, step = 21, top = 36, rowHeight = 27;
  const height = top + DRUM_GRID_LANES.length * rowHeight + 10;
  bars.forEach((bar, barIndex) => {
    const card = document.createElement('section');
    card.className = 'drum-grid-card';
    card.setAttribute('aria-label', `Compasso ${bar.number}`);
    const heading = document.createElement('div');
    heading.className = 'drum-grid-heading';
    const number = document.createElement('strong');
    number.textContent = `Compasso ${bar.number}`;
    heading.append(number);
    if (bar.lyric) {
      const lyric = document.createElement('span');
      lyric.textContent = bar.lyric;
      heading.append(lyric);
    }
    card.append(heading);
    const svg = svgNode('svg', { viewBox: `0 0 ${width} ${height}`, class: 'drum-grid-svg', role: 'img', 'aria-label': `Grade de bateria, compasso ${bar.number}` });
    for (let beat = 0; beat < 4; beat++) {
      svg.append(svgNode('rect', { x: firstX - step / 2 + beat * step * 4, y: top, width: step * 4, height: height - top, class: beat % 2 ? 'grid-beat-alt' : 'grid-beat' }));
      svg.append(svgNode('line', { x1: firstX - step / 2 + beat * step * 4, x2: firstX - step / 2 + beat * step * 4, y1: top, y2: height, class: 'grid-beat-line' }));
    }
    for (let slot = 0; slot < 16; slot++) {
      const count = slot % 4 === 0 ? String(slot / 4 + 1) : ['','e','+','a'][slot % 4];
      svg.append(svgNode('text', { x: firstX + slot * step, y: 21, class: slot % 4 === 0 ? 'grid-count grid-main-count' : 'grid-count' }, count));
    }
    DRUM_GRID_LANES.forEach((lane, row) => {
      const y = top + 17 + row * rowHeight;
      const present = bar.slots.some(slot => slot.has(lane.label));
      svg.append(svgNode('line', { x1: 49, x2: 401, y1: y, y2: y, class: 'grid-lane-line' }));
      const label = svgNode('g', { class: present ? 'grid-label grid-has-hits' : 'grid-label' });
      label.append(svgNode('title', {}, lane.name));
      label.append(svgNode('rect', { x: 4, y: y - 11, width: 42, height: 21, rx: 5 }));
      label.append(svgNode('text', { x: 25, y: y + 1 }, lane.label));
      svg.append(label);
      bar.slots.forEach((slot, slotIndex) => {
        const hit = slot.get(lane.label);
        if (hit) svg.append(svgNode('text', { x: firstX + slotIndex * step, y: y + 2, class: hit === '(o)' ? 'grid-hit grid-ghost' : 'grid-hit' }, hit));
      });
    });
    const cursor = svgNode('line', { x1: firstX, x2: firstX, y1: top, y2: height, class: 'grid-playhead' });
    svg.append(cursor);
    card.append(svg);
    container.append(card);
    cards.push({ card, cursor, index: barIndex });
  });
  return { cards, firstX, step };
}
