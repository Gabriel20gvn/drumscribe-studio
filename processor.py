import os
import re
import glob
import json
import math
import time
import uuid
import base64
import shutil
import zipfile
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import Callable, Optional, Dict, Any, List

import requests
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
import yt_dlp
import whisper
from music21 import stream, note, chord, meter, tempo, metadata
import torch
from adtof_pytorch import (
    calculate_n_bins, create_frame_rnn_model, get_default_weights_path,
    load_pytorch_weights, PeakPicker, FRAME_RNN_THRESHOLDS, LABELS_5,
)
from adtof_pytorch.audio import create_adtof_processor
from drum_timing import first_drum_beat_origin


def sanitize_filename(name: str) -> str:
    clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return clean[:80] if clean else "musica_drumscribe"


def get_library_root(base_dir: str) -> str:
    drive_root = "/content/drive/MyDrive"
    if os.path.isdir(drive_root):
        lib_dir = os.path.join(drive_root, "DrumScribe_Biblioteca")
    else:
        lib_dir = os.path.join(base_dir, "library")
    os.makedirs(lib_dir, exist_ok=True)
    return lib_dir


def run_ffmpeg_convert(input_path: str, output_path: str, bitrate: str = "160k"):
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_path,
        "-b:a", bitrate,
        output_path
    ]
    subprocess.run(cmd, check=True)


def convert_uploaded_to_wav(input_path: str, job_dir: str, original_name: str) -> tuple[str, str]:
    wav_path = os.path.join(job_dir, "raw_audio.wav")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", input_path,
        "-ar", "44100",
        "-ac", "2",
        "-acodec", "pcm_s16le",
        wav_path
    ]
    subprocess.run(cmd, check=True)
    base_title = os.path.splitext(original_name)[0]
    return wav_path, sanitize_filename(base_title)


def _download_with_pytubefix(url: str, job_dir: str) -> tuple[str, str]:
    from pytubefix import YouTube

    wav_path = os.path.join(job_dir, "raw_audio.wav")
    last_err = None

    for client_name in ["ANDROID_VR", "IOS", "TV", "MWEB", "WEB"]:
        try:
            yt = YouTube(url, client=client_name)
            title = sanitize_filename(yt.title or "musica_youtube")
            audio_stream = (
                yt.streams.filter(only_audio=True).order_by("abr").desc().first()
                or yt.streams.order_by("resolution").asc().first()
            )
            if not audio_stream:
                continue

            temp_media = audio_stream.download(output_path=job_dir, filename="raw_ptf_media")
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", temp_media,
                "-ar", "44100",
                "-ac", "2",
                "-acodec", "pcm_s16le",
                wav_path
            ]
            subprocess.run(cmd, check=True)
            if os.path.exists(temp_media):
                os.remove(temp_media)
            if os.path.exists(wav_path):
                return wav_path, title
        except Exception as exc:
            last_err = exc
            continue

    raise RuntimeError(f"Fallback pytubefix falhou: {last_err}")


def _is_region_restricted_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(phrase in message for phrase in (
        "not available in your country",
        "not made this video available in your country",
        "not available in your region",
        "geo-restricted",
        "geo restricted",
    ))


def download_youtube_wav(url: str, job_dir: str, cookies_path: Optional[str] = None) -> tuple[str, str]:
    raw_template = os.path.join(job_dir, "raw_audio.%(ext)s")

    drive_cookies = "/content/drive/MyDrive/cookies.txt"
    effective_cookies = cookies_path if (cookies_path and os.path.exists(cookies_path)) else (
        drive_cookies if os.path.exists(drive_cookies) else None
    )

    attempts = [
        {"clients": None, "format": "bestaudio/best"},
        {"clients": ["android_vr", "tv"], "format": "bestaudio/best"},
        {"clients": ["ios", "mweb"], "format": "bestaudio/best"},
        {"clients": ["web_safari", "web"], "format": "best/bestaudio"},
    ]

    region_restricted = False
    title = "musica_youtube"

    for attempt in attempts:
        for old_file in glob.glob(os.path.join(job_dir, "raw_audio.*")):
            try:
                os.remove(old_file)
            except OSError:
                pass

        ydl_opts: Dict[str, Any] = {
            "outtmpl": raw_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "geo_bypass": True,
            "remote_components": ["ejs:github"],
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
            }],
            "postprocessor_args": [
                "-ar", "44100",
                "-ac", "2",
                "-acodec", "pcm_s16le"
            ],
        }

        if attempt["format"]:
            ydl_opts["format"] = attempt["format"]

        if attempt["clients"]:
            ydl_opts["extractor_args"] = {
                "youtube": {
                    "player_client": attempt["clients"]
                }
            }

        if effective_cookies:
            ydl_opts["cookiefile"] = effective_cookies

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = sanitize_filename(info.get("title", "musica_youtube"))
            wav_path = os.path.join(job_dir, "raw_audio.wav")
            if os.path.exists(wav_path):
                return wav_path, title
        except Exception as exc:
            region_restricted = region_restricted or _is_region_restricted_error(exc)
            continue

    if region_restricted:
        raise RuntimeError(
            "Este vídeo não está disponível na região do servidor do Colab. "
            "Para gerar a partitura, envie um arquivo de áudio/vídeo pelo campo 'Ou envie um arquivo' "
            "ou escolha outro link disponível nessa região. Cookies não removem essa restrição."
        )

    try:
        return _download_with_pytubefix(url, job_dir)
    except Exception:
        raise RuntimeError(
            "Não foi possível obter o áudio deste link no servidor. "
            "Tente outro link ou envie um arquivo de áudio/vídeo pelo campo 'Ou envie um arquivo'."
        )


def separate_with_demucs(wav_path: str, job_dir: str, model: str = "htdemucs") -> Dict[str, str]:
    stems_root = os.path.join(job_dir, "stems_out")
    os.makedirs(stems_root, exist_ok=True)

    cmd = [
        "demucs",
        "-n", model,
        "-o", stems_root,
        wav_path
    ]
    subprocess.run(cmd, check=True)

    track_dir = os.path.join(stems_root, model, "raw_audio")
    stems = {}
    for part in ["drums", "vocals", "bass", "other"]:
        p = os.path.join(track_dir, f"{part}.wav")
        if os.path.exists(p):
            dest = os.path.join(job_dir, f"{part}.wav")
            shutil.copy2(p, dest)
            stems[part] = dest

    playback_arrays = []
    sr = 44100
    for part in ["vocals", "bass", "other"]:
        if part in stems:
            data, sr = sf.read(stems[part], always_2d=True)
            playback_arrays.append(data)

    if playback_arrays:
        min_len = min(arr.shape[0] for arr in playback_arrays)
        mix = sum(arr[:min_len] for arr in playback_arrays)
        peak = np.max(np.abs(mix))
        if peak > 0.99:
            mix = mix / peak * 0.95
        no_drums_wav = os.path.join(job_dir, "playback_sem_bateria.wav")
        sf.write(no_drums_wav, mix, sr, subtype="PCM_16")
        stems["no_drums"] = no_drums_wav

    mp3_stems = {}
    for key, wpath in stems.items():
        mp3_path = os.path.join(job_dir, f"{key}.mp3")
        run_ffmpeg_convert(wpath, mp3_path, bitrate="160k")
        mp3_stems[key] = os.path.basename(mp3_path)

    return {"wav": stems, "mp3": mp3_stems}


def _transcribe_drum_events_adtof(drums_wav: str, job_dir: str) -> Dict[str, List[float]]:
    """Run the trained five-class model in bounded chunks of the Demucs drum stem."""
    weights_path = get_default_weights_path()
    if not weights_path or not os.path.isfile(weights_path):
        raise RuntimeError("Pesos do ADTOF ausentes. Instale adtof-pytorch no ambiente do Colab.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_frame_rnn_model(calculate_n_bins())
    model = load_pytorch_weights(model, weights_path, strict=True)
    model.eval().to(device)
    picker = PeakPicker(thresholds=FRAME_RNN_THRESHOLDS, fps=100)
    processor = create_adtof_processor()
    pitch_to_piece = {35: "kick", 38: "snare", 47: "tom", 42: "hihat", 49: "crash"}
    events: Dict[str, List[float]] = {piece: [] for piece in pitch_to_piece.values()}
    chunk_file = os.path.join(job_dir, "adtof_chunk.wav")

    try:
        with sf.SoundFile(drums_wav) as audio:
            sr = audio.samplerate
            chunk_frames = 20 * sr
            margin_frames = sr
            for start in range(0, len(audio), chunk_frames):
                read_start = max(0, start - margin_frames)
                read_end = min(len(audio), start + chunk_frames + margin_frames)
                audio.seek(read_start)
                data = audio.read(read_end - read_start, dtype="float32", always_2d=True)
                sf.write(chunk_file, data.mean(axis=1), sr, subtype="PCM_16")
                features = processor.process_audio(chunk_file)
                with torch.inference_mode():
                    activations = model(torch.from_numpy(features).unsqueeze(0).to(device)).cpu().numpy()
                picked = picker.pick(activations, labels=LABELS_5)[0]
                valid_start = start / sr
                valid_end = min(len(audio), start + chunk_frames) / sr
                chunk_offset = read_start / sr
                for pitch, times in picked.items():
                    for relative_time in times:
                        absolute_time = chunk_offset + relative_time
                        if valid_start <= absolute_time < valid_end:
                            events[pitch_to_piece[pitch]].append(absolute_time)
    finally:
        if os.path.exists(chunk_file):
            os.remove(chunk_file)
    return events


def _refine_bpm_from_beats(tempo_estimate: float, beat_times: np.ndarray) -> float:
    """Average tracked beat intervals to remove librosa's discrete tempo-bin error."""
    if not np.isfinite(tempo_estimate) or tempo_estimate <= 0 or len(beat_times) < 12:
        return tempo_estimate
    expected_interval = 60.0 / tempo_estimate
    intervals = np.diff(beat_times)
    regular = intervals[
        (intervals >= expected_interval * 0.8)
        & (intervals <= expected_interval * 1.2)
    ]
    if len(regular) < 10 or len(regular) < len(intervals) * 0.5:
        return tempo_estimate
    refined = 60.0 / float(np.mean(regular))
    if abs(refined - tempo_estimate) > tempo_estimate * 0.08:
        return tempo_estimate
    nearest_integer = round(refined)
    return float(nearest_integer if abs(refined - nearest_integer) < 0.05 else round(refined, 2))


def _detect_ghost_snare_times(
    y: np.ndarray, sr: int, snare_times: List[float], crash_times: List[float]
) -> set[float]:
    """Flag exceptionally quiet detected snares, relative to nearby snare hits."""
    if len(snare_times) < 8:
        return set()
    snare_band = sosfilt(
        butter(4, [180, 3000], btype="bandpass", fs=sr, output="sos"), y
    )
    times = np.asarray(snare_times, dtype=float)
    energy = np.zeros(len(times), dtype=float)
    for index, hit_time in enumerate(times):
        start = max(0, int((hit_time - 0.015) * sr))
        end = min(len(snare_band), int((hit_time + 0.065) * sr))
        if end > start:
            segment = snare_band[start:end]
            energy[index] = float(np.sqrt(np.mean(segment * segment)))

    ghosts: set[float] = set()
    global_reference = float(np.percentile(energy, 85))
    for index, hit_time in enumerate(times):
        nearby = energy[np.abs(times - hit_time) <= 12.0]
        reference = float(np.percentile(nearby, 85)) if len(nearby) >= 6 else global_reference
        if reference <= 0 or energy[index] >= reference * 0.35:
            continue
        if any(abs(hit_time - crash_time) <= 0.04 for crash_time in crash_times):
            continue
        ghosts.add(float(hit_time))
    return ghosts


def _build_drum_musicxml(
    title: str,
    bpm: float,
    measures_events: List[List[tuple[int, set]]],
    lyrics_by_measure: Dict[int, List[str]],
    audio_offset_seconds: float = 0.0,
) -> str:
    """
    MusicXML 3.1 com numeração MIDI 1-based (GM MIDI + 1):
    37 = Bumbo (Bass Drum 1), 39 = Caixa (Acoustic Snare),
    43 = Chimbal (Closed Hi-Hat), 48 = Tom (Low-Mid Tom), 50 = Crash (Crash Cymbal 1).
    """
    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title

    identification = ET.SubElement(score, "identification")
    creator = ET.SubElement(identification, "creator", type="composer")
    creator.text = "DrumScribe AI (Demucs + ADTOF + Whisper)"
    miscellaneous = ET.SubElement(identification, "miscellaneous")
    ET.SubElement(miscellaneous, "miscellaneous-field", name="drumscribe-audio-offset-seconds").text = f"{audio_offset_seconds:.3f}"

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Bateria"

    instruments = [
        ("P1-I37", "Bass Drum 1", 37),
        ("P1-I39", "Acoustic Snare", 39),
        ("P1-I43", "Closed Hi-Hat", 43),
        ("P1-I48", "Low-Mid Tom", 48),
        ("P1-I50", "Crash Cymbal 1", 50),
    ]
    for inst_id, inst_name, midi_num in instruments:
        si = ET.SubElement(score_part, "score-instrument", id=inst_id)
        ET.SubElement(si, "instrument-name").text = inst_name
        mi = ET.SubElement(score_part, "midi-instrument", id=inst_id)
        ET.SubElement(mi, "midi-channel").text = "10"
        ET.SubElement(mi, "midi-program").text = "1"
        ET.SubElement(mi, "midi-unpitched").text = str(midi_num)

    part = ET.SubElement(score, "part", id="P1")

    piece_spec = {
        "kick":  ("F", "4", "normal", "P1-I37"),
        "snare": ("C", "5", "normal", "P1-I39"),
        "snare_ghost": ("C", "5", "normal", "P1-I39"),
        "tom":   ("E", "5", "normal", "P1-I48"),
        "hihat": ("G", "5", "x",      "P1-I43"),
        "crash": ("A", "5", "x",      "P1-I50"),
    }

    dur_type_map = {
        16: ("whole", None),
        12: ("half", "dot"),
        8:  ("half", None),
        6:  ("quarter", "dot"),
        4:  ("quarter", None),
        3:  ("eighth", "dot"),
        2:  ("eighth", None),
        1:  ("16th", None),
    }

    def append_voice(measure, hits_by_step: Dict[int, set], voice_number: int, stem: str):
        positions = sorted(hits_by_step)
        position = 0
        while position < 16:
            following = next((step for step in positions if step > position), 16)
            span = following - position
            dur_steps = next(steps for steps in dur_type_map if steps <= span)
            note_type, dot_flag = dur_type_map[dur_steps]
            hits = hits_by_step.get(position, set())
            if not hits:
                n_el = ET.SubElement(measure, "note")
                ET.SubElement(n_el, "rest")
                ET.SubElement(n_el, "duration").text = str(dur_steps)
                ET.SubElement(n_el, "voice").text = str(voice_number)
                ET.SubElement(n_el, "type").text = note_type
                if dot_flag:
                    ET.SubElement(n_el, "dot")
            else:
                ordered = [p for p in ["kick", "snare", "snare_ghost", "tom", "hihat", "crash"] if p in hits]
                for idx_h, piece in enumerate(ordered):
                    step, octave, nhead, inst_id = piece_spec[piece]
                    n_el = ET.SubElement(measure, "note")
                    if idx_h:
                        ET.SubElement(n_el, "chord")
                    unp = ET.SubElement(n_el, "unpitched")
                    ET.SubElement(unp, "display-step").text = step
                    ET.SubElement(unp, "display-octave").text = octave
                    ET.SubElement(n_el, "duration").text = str(dur_steps)
                    ET.SubElement(n_el, "instrument", id=inst_id)
                    ET.SubElement(n_el, "voice").text = str(voice_number)
                    ET.SubElement(n_el, "type").text = note_type
                    if dot_flag:
                        ET.SubElement(n_el, "dot")
                    ET.SubElement(n_el, "stem").text = stem
                    head = ET.SubElement(n_el, "notehead")
                    head.text = nhead
                    if piece == "snare_ghost":
                        head.set("parentheses", "yes")
            position += dur_steps

    for m_idx, events in enumerate(measures_events):
        measure = ET.SubElement(part, "measure", number=str(m_idx + 1))
        if m_idx == 0:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = "4"
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = "4"
            ET.SubElement(time_el, "beat-type").text = "4"
            clef_el = ET.SubElement(attrs, "clef")
            ET.SubElement(clef_el, "sign").text = "percussion"
            ET.SubElement(clef_el, "line").text = "2"

            direction = ET.SubElement(measure, "direction", placement="above")
            dir_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(dir_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = str(bpm)
            ET.SubElement(direction, "sound", tempo=str(bpm))

        if m_idx in lyrics_by_measure and lyrics_by_measure[m_idx]:
            frase = " ".join(lyrics_by_measure[m_idx])
            direction = ET.SubElement(measure, "direction", placement="above")
            dir_type = ET.SubElement(direction, "direction-type")
            words = ET.SubElement(dir_type, "words", **{"font-style": "italic", "font-size": "9"})
            words.text = frase

        cymbals: Dict[int, set] = {}
        kit: Dict[int, set] = {}
        position = 0
        for dur_steps, hits in events:
            upper = hits.intersection({"hihat", "crash"})
            lower = hits.intersection({"kick", "snare", "snare_ghost", "tom"})
            if upper:
                cymbals[position] = upper
            if lower:
                kit[position] = lower
            position += dur_steps

        if cymbals:
            append_voice(measure, cymbals, 1, "up")
            if kit:
                backup = ET.SubElement(measure, "backup")
                ET.SubElement(backup, "duration").text = "16"
                append_voice(measure, kit, 2, "down")
        else:
            append_voice(measure, kit, 1, "down")

    rough = ET.tostring(score, encoding="utf-8")
    parsed = minidom.parseString(rough)
    return parsed.toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def transcribe_drums_and_lyrics(
    drums_wav: str,
    vocals_wav: Optional[str],
    title: str,
    job_dir: str,
    whisper_model_name: str = "small",
    language: str = "pt",
    progress_cb: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    if progress_cb:
        progress_cb(60, "Analisando andamento (BPM) e golpes de bateria com ADTOF...")

    y, sr = librosa.load(drums_wav, sr=22050, mono=True)
    duration_sec = librosa.get_duration(y=y, sr=sr)

    tempo_est, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=128.0)
    bpm = float(np.atleast_1d(tempo_est)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = _refine_bpm_from_beats(bpm, beat_times)
    if bpm < 60 or bpm > 210 or math.isnan(bpm):
        bpm = 120.0

    sec_per_beat = 60.0 / bpm
    sec_per_16th = sec_per_beat / 4.0
    sec_per_measure = sec_per_beat * 4.0

    drum_events = _transcribe_drum_events_adtof(drums_wav, job_dir)
    ghost_snare_times = _detect_ghost_snare_times(
        y, sr, drum_events["snare"], drum_events["crash"]
    )
    grid_origin = first_drum_beat_origin(y, sr, bpm, beat_times, drum_events)

    effective_duration = max(sec_per_measure, duration_sec - grid_origin)
    total_measures = max(1, int(math.ceil(effective_duration / sec_per_measure)))
    total_steps = total_measures * 16

    grid: Dict[int, set] = {}

    for piece, times in drum_events.items():
        for t in times:
            if t < grid_origin - 0.05:
                continue
            step = int(round((t - grid_origin) / sec_per_16th))
            if 0 <= step < total_steps:
                score_piece = "snare_ghost" if piece == "snare" and t in ghost_snare_times else piece
                grid.setdefault(step, set()).add(score_piece)

    for hits in grid.values():
        if "snare" in hits:
            hits.discard("snare_ghost")

    lyrics_by_measure: Dict[int, list[str]] = {}
    if vocals_wav and os.path.exists(vocals_wav):
        if progress_cb:
            progress_cb(75, f"Transcrevendo letra com Whisper ({whisper_model_name}) e alinhando compassos...")
        w_model = whisper.load_model(whisper_model_name)
        lang_arg = None if language == "auto" else language
        result = w_model.transcribe(
            vocals_wav,
            language=lang_arg,
            word_timestamps=True
        )
        for seg in result.get("segments", []):
            words = seg.get("words", [])
            if words:
                for w in words:
                    w_start = float(w.get("start", 0.0))
                    w_text = w.get("word", "").strip()
                    if w_text:
                        m_idx = max(0, int((w_start - grid_origin) // sec_per_measure))
                        lyrics_by_measure.setdefault(m_idx, []).append(w_text)
            else:
                s_start = float(seg.get("start", 0.0))
                s_text = seg.get("text", "").strip()
                if s_text:
                    m_idx = max(0, int((s_start - grid_origin) // sec_per_measure))
                    lyrics_by_measure.setdefault(m_idx, []).append(s_text)

    if progress_cb:
        progress_cb(86, "Montando partitura MusicXML 3.1 de bateria com letras...")

    measures_events: List[List[tuple[int, set]]] = []
    midi_score = stream.Score()
    midi_score.insert(0, metadata.Metadata())
    midi_score.metadata.title = title
    midi_part = stream.Part()
    midi_pitch_map = {"kick": "B1", "snare": "D2", "snare_ghost": "D2", "tom": "B2", "hihat": "F#2", "crash": "C#3"}

    for m_idx in range(total_measures):
        m_midi = stream.Measure(number=m_idx + 1)
        if m_idx == 0:
            m_midi.insert(0.0, meter.TimeSignature("4/4"))
            m_midi.insert(0.0, tempo.MetronomeMark(number=bpm))

        base_step = m_idx * 16
        step_offset = 0
        m_events: List[tuple[int, set]] = []

        while step_offset < 16:
            abs_step = base_step + step_offset
            hits = grid.get(abs_step, set())

            span = 1
            while (step_offset + span) < 16 and not grid.get(base_step + step_offset + span):
                span += 1

            if span >= 8 and (step_offset % 8 == 0):
                dur_steps = 8
            elif span >= 4 and (step_offset % 4 == 0):
                dur_steps = 4
            elif span >= 2 and (step_offset % 2 == 0):
                dur_steps = 2
            else:
                dur_steps = 1

            m_events.append((dur_steps, hits))

            quarter_len = dur_steps * 0.25
            if not hits:
                m_midi.append(note.Rest(quarterLength=quarter_len))
            elif len(hits) == 1:
                piece = next(iter(hits))
                m_midi.append(note.Note(midi_pitch_map[piece], quarterLength=quarter_len))
            else:
                m_midi.append(chord.Chord([midi_pitch_map[p] for p in hits], quarterLength=quarter_len))

            step_offset += dur_steps

        measures_events.append(m_events)
        midi_part.append(m_midi)

    midi_score.append(midi_part)

    xml_content = _build_drum_musicxml(title, bpm, measures_events, lyrics_by_measure, grid_origin)
    xml_path = os.path.join(job_dir, "partitura_bateria.musicxml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    mid_path = os.path.join(job_dir, "partitura_bateria.mid")
    midi_score.write("midi", fp=mid_path)

    return {
        "bpm": bpm,
        "measures": total_measures,
        "audio_offset_seconds": grid_origin,
        "musicxml": "partitura_bateria.musicxml",
        "midi": "partitura_bateria.mid",
    }


def _github_put_file(repo: str, path: str, content_bytes: bytes, token: str, message: str):
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    sha = None
    get_resp = requests.get(api_url, headers=headers, timeout=20)
    if get_resp.status_code == 200:
        sha = get_resp.json().get("sha")

    payload: Dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content_bytes).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(api_url, headers=headers, json=payload, timeout=60)
    if put_resp.status_code not in (200, 201):
        raise RuntimeError(f"GitHub API ({path}): {put_resp.status_code} - {put_resp.text[:200]}")


def push_item_to_github_library(
    job_dir: str,
    meta: Dict[str, Any],
    github_token: str,
    github_repo: str = "Gabriel20gvn/drumscribe-studio",
) -> bool:
    item_id = meta["id"]
    files_to_upload = [
        ("partitura_bateria.musicxml", meta.get("musicxml")),
        ("drums.mp3", meta.get("audio_mp3", {}).get("drums")),
        ("no_drums.mp3", meta.get("audio_mp3", {}).get("no_drums")),
        ("vocals.mp3", meta.get("audio_mp3", {}).get("vocals")),
    ]

    for target_name, local_name in files_to_upload:
        if not local_name:
            continue
        local_path = os.path.join(job_dir, local_name)
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                _github_put_file(
                    repo=github_repo,
                    path=f"library/{item_id}/{target_name}",
                    content_bytes=f.read(),
                    token=github_token,
                    message=f"Add {target_name} for {meta['title']}"
                )

    index_url = f"https://api.github.com/repos/{github_repo}/contents/library/index.json"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }
    catalog: List[Dict[str, Any]] = []
    get_idx = requests.get(index_url, headers=headers, timeout=20)
    if get_idx.status_code == 200:
        try:
            raw_b64 = get_idx.json().get("content", "")
            catalog = json.loads(base64.b64decode(raw_b64).decode("utf-8"))
        except Exception:
            catalog = []

    catalog = [entry for entry in catalog if entry.get("id") != item_id]
    catalog.insert(0, {
        "id": item_id,
        "title": meta["title"],
        "bpm": meta["bpm"],
        "measures": meta["measures"],
        "audio_offset_seconds": meta.get("audio_offset_seconds", 0),
        "created_at": meta["created_at"],
        "musicxml": f"library/{item_id}/partitura_bateria.musicxml",
        "audio_mp3": {
            "drums": f"library/{item_id}/drums.mp3" if meta["audio_mp3"].get("drums") else None,
            "no_drums": f"library/{item_id}/no_drums.mp3" if meta["audio_mp3"].get("no_drums") else None,
            "vocals": f"library/{item_id}/vocals.mp3" if meta["audio_mp3"].get("vocals") else None,
        }
    })

    _github_put_file(
        repo=github_repo,
        path="library/index.json",
        content_bytes=json.dumps(catalog, ensure_ascii=False, indent=2).encode("utf-8"),
        token=github_token,
        message=f"Update library catalog: {meta['title']}"
    )
    return True


def delete_item_from_github_library(
    item_id: str,
    github_token: str,
    github_repo: str = "Gabriel20gvn/drumscribe-studio",
):
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
    }
    index_url = f"https://api.github.com/repos/{github_repo}/contents/library/index.json"
    get_idx = requests.get(index_url, headers=headers, timeout=20)
    if get_idx.status_code == 200:
        raw_b64 = get_idx.json().get("content", "")
        catalog = json.loads(base64.b64decode(raw_b64).decode("utf-8"))
        catalog = [entry for entry in catalog if entry.get("id") != item_id]
        _github_put_file(
            repo=github_repo,
            path="library/index.json",
            content_bytes=json.dumps(catalog, ensure_ascii=False, indent=2).encode("utf-8"),
            token=github_token,
            message=f"Remove {item_id} from library catalog"
        )

    for fname in ["partitura_bateria.musicxml", "drums.mp3", "no_drums.mp3", "vocals.mp3"]:
        f_url = f"https://api.github.com/repos/{github_repo}/contents/library/{item_id}/{fname}"
        r = requests.get(f_url, headers=headers, timeout=15)
        if r.status_code == 200:
            sha = r.json().get("sha")
            requests.delete(
                f_url,
                headers=headers,
                json={"message": f"Delete {fname}", "sha": sha},
                timeout=20,
            )


def save_to_library(job_dir: str, result_data: Dict[str, Any], base_dir: str) -> Dict[str, Any]:
    lib_root = get_library_root(base_dir)
    item_id = f"{int(time.time())}_{uuid.uuid4().hex[:6]}"
    item_dir = os.path.join(lib_root, item_id)
    os.makedirs(item_dir, exist_ok=True)

    files_to_copy = [
        result_data.get("musicxml"),
        result_data.get("midi"),
        result_data.get("zip"),
    ]
    for mp3_name in result_data.get("audio_mp3", {}).values():
        files_to_copy.append(mp3_name)

    for fname in files_to_copy:
        if fname:
            src = os.path.join(job_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(item_dir, fname))

    meta = {
        "id": item_id,
        "title": result_data["title"],
        "bpm": result_data["bpm"],
        "measures": result_data["measures"],
        "audio_offset_seconds": result_data.get("audio_offset_seconds", 0),
        "created_at": time.time(),
        "musicxml": result_data["musicxml"],
        "midi": result_data["midi"],
        "zip": result_data["zip"],
        "audio_mp3": result_data["audio_mp3"],
    }
    with open(os.path.join(item_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def process_full_pipeline(
    url: str,
    job_dir: str,
    demucs_model: str = "htdemucs",
    whisper_model: str = "small",
    language: str = "pt",
    cookies_path: Optional[str] = None,
    uploaded_media_path: Optional[str] = None,
    uploaded_media_name: Optional[str] = None,
    base_dir: Optional[str] = None,
    github_token: Optional[str] = None,
    github_repo: str = "Gabriel20gvn/drumscribe-studio",
    progress_cb: Optional[Callable[[int, str], None]] = None
) -> Dict[str, Any]:
    if uploaded_media_path and os.path.exists(uploaded_media_path):
        if progress_cb:
            progress_cb(12, "Convertendo arquivo enviado para WAV...")
        wav_path, title = convert_uploaded_to_wav(
            uploaded_media_path, job_dir, uploaded_media_name or "musica_enviada.mp3"
        )
    else:
        if progress_cb:
            progress_cb(10, "Baixando áudio do link em WAV...")
        wav_path, title = download_youtube_wav(url, job_dir, cookies_path=cookies_path)

    if progress_cb:
        progress_cb(30, f"Separando instrumentos com Demucs ({demucs_model})...")
    stems = separate_with_demucs(wav_path, job_dir, model=demucs_model)

    drums_wav = stems["wav"].get("drums")
    vocals_wav = stems["wav"].get("vocals")
    if not drums_wav:
        raise RuntimeError("Não foi possível isolar a faixa de bateria.")

    score_info = transcribe_drums_and_lyrics(
        drums_wav=drums_wav,
        vocals_wav=vocals_wav,
        title=title,
        job_dir=job_dir,
        whisper_model_name=whisper_model,
        language=language,
        progress_cb=progress_cb
    )

    if progress_cb:
        progress_cb(91, "Empacotando arquivos WAV, MIDI e MusicXML...")

    zip_filename = f"{title}_kit_bateria.zip"
    zip_path = os.path.join(job_dir, zip_filename)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in [
            "partitura_bateria.musicxml",
            "partitura_bateria.mid",
            "drums.wav",
            "playback_sem_bateria.wav",
            "vocals.wav",
            "bass.wav",
            "other.wav",
        ]:
            fpath = os.path.join(job_dir, fname)
            if os.path.exists(fpath):
                zf.write(fpath, arcname=fname)

    result_payload: Dict[str, Any] = {
        "title": title,
        "bpm": score_info["bpm"],
        "measures": score_info["measures"],
        "audio_offset_seconds": score_info["audio_offset_seconds"],
        "musicxml": score_info["musicxml"],
        "midi": score_info["midi"],
        "zip": zip_filename,
        "audio_mp3": stems["mp3"],
        "cloud_synced": False,
    }

    if base_dir:
        meta = save_to_library(job_dir, result_payload, base_dir)
        result_payload["library_id"] = meta["id"]

        effective_token = (github_token or os.environ.get("GITHUB_TOKEN") or "").strip()
        effective_repo = (github_repo or os.environ.get("GITHUB_REPO") or "Gabriel20gvn/drumscribe-studio").strip()
        if effective_token:
            if progress_cb:
                progress_cb(96, "Publicando partitura e stems na Biblioteca Nuvem (acesso no celular)...")
            try:
                push_item_to_github_library(job_dir, meta, effective_token, effective_repo)
                result_payload["cloud_synced"] = True
            except Exception as sync_exc:
                result_payload["cloud_sync_error"] = str(sync_exc)

    return result_payload
