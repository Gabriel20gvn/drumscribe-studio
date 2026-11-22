/* Present DrumScribe's generated single-voice drum scores as two percussion voices. */
function formatGeneratedDrumScore(buffer) {
  const source = new TextDecoder().decode(buffer);
  if (!source.includes('DrumScribe AI')) return buffer;

  const doc = new DOMParser().parseFromString(source, 'application/xml');
  if (doc.getElementsByTagName('parsererror').length) return buffer;
  const creators = [...doc.getElementsByTagName('creator')];
  if (!creators.some(el => (el.textContent || '').startsWith('DrumScribe AI'))) return buffer;

  const measures = [...doc.getElementsByTagName('measure')];
  const durations = [16, 12, 8, 6, 4, 3, 2, 1];
  const durationTypes = {
    16: ['whole', false], 12: ['half', true], 8: ['half', false],
    6: ['quarter', true], 4: ['quarter', false], 3: ['eighth', true],
    2: ['eighth', false], 1: ['16th', false]
  };
  const cymbalIds = new Set(['P1-I43', 'P1-I50']);
  let changed = false;

  function element(name, value) {
    const node = doc.createElement(name);
    if (value !== undefined) node.textContent = String(value);
    return node;
  }

  function appendVoice(measure, hits, voiceNumber, stem) {
    const positions = [...hits.keys()].sort((a, b) => a - b);
    let cursor = 0;
    while (cursor < 16) {
      const present = hits.get(cursor) || [];
      const next = positions.find(position => position > cursor) ?? 16;
      const span = next - cursor;
      const length = durations.find(value => value <= span) || 1;
      const [type, dotted] = durationTypes[length];

      if (!present.length) {
        const rest = element('note');
        rest.append(element('rest'), element('duration', length),
          element('voice', voiceNumber), element('type', type));
        if (dotted) rest.append(element('dot'));
        measure.append(rest);
      } else {
        present.forEach((original, index) => {
          const note = element('note');
          if (index) note.append(element('chord'));
          note.append(original.getElementsByTagName('unpitched')[0].cloneNode(true));
          note.append(element('duration', length));
          note.append(original.getElementsByTagName('instrument')[0].cloneNode(true));
          note.append(element('voice', voiceNumber), element('type', type));
          if (dotted) note.append(element('dot'));
          note.append(element('stem', stem));
          const head = original.getElementsByTagName('notehead')[0];
          if (head) note.append(head.cloneNode(true));
          measure.append(note);
        });
      }
      cursor += length;
    }
  }

  for (const measure of measures) {
    const notes = [...measure.children].filter(child => child.tagName === 'note');
    if (!notes.length || notes.some(note => {
      const voice = note.getElementsByTagName('voice')[0];
      return voice && voice.textContent !== '1';
    })) continue;

    const cymbals = new Map();
    const kit = new Map();
    let cursor = 0;
    let onset = 0;
    let valid = true;
    for (const note of notes) {
      const chord = note.getElementsByTagName('chord').length > 0;
      if (!chord) onset = cursor;
      const length = Number(note.getElementsByTagName('duration')[0]?.textContent);
      if (!Number.isInteger(length) || length < 1) { valid = false; break; }
      if (!chord) cursor += length;
      if (note.getElementsByTagName('rest').length) continue;
      const instrument = note.getElementsByTagName('instrument')[0];
      if (!instrument || !note.getElementsByTagName('unpitched').length) { valid = false; break; }
      const target = cymbalIds.has(instrument.getAttribute('id')) ? cymbals : kit;
      if (!target.has(onset)) target.set(onset, []);
      target.get(onset).push(note);
    }
    if (!valid || cursor !== 16) continue;

    notes.forEach(note => note.remove());
    if (cymbals.size) {
      appendVoice(measure, cymbals, 1, 'up');
      if (kit.size) {
        const backup = element('backup');
        backup.append(element('duration', 16));
        measure.append(backup);
        appendVoice(measure, kit, 2, 'down');
      }
    } else {
      appendVoice(measure, kit, 1, 'down');
    }
    changed = true;
  }

  return changed ? new TextEncoder().encode(new XMLSerializer().serializeToString(doc)) : buffer;
}
