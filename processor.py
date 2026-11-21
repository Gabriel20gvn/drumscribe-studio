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

    last_error = None
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
            last_error = exc
            continue

    try:
        return _download_with_pytubefix(url, job_dir)
    except Exception as ptf_exc:
        raise RuntimeError(
            f"Não foi possível extrair o áudio deste link. "
            f"(yt-dlp: {last_error} | pytubefix: {ptf_exc})"
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


def _bandpass_filter(y: np.ndarray, sr: int, low: float, high: float) -> np.ndarray:
    nyq = 0.5 * sr
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    sos = butter(4, [low_n, high_n], btype="band", output="sos")
    return sosfilt(sos, y)


def _highpass_filter(y: np.ndarray, sr: int, cutoff: float) -> np.ndarray:
    nyq = 0.5 * sr
    cut_n = min(cutoff / nyq, 0.999)
    sos = butter(4, cut_n, btype="high", output="sos")
    return sosfilt(sos, y)


def _detect_band_onsets(y_band: np.ndarray, sr: int, delta: float = 0.12) -> tuple[np.ndarray, np.ndarray]:
    hop_length = 256
    onset_env = librosa.onset.onset_strength(y=y_band, sr=sr, hop_length=hop_length)
    max_val = float(np.max(onset_env)) if len(onset_env) > 0 else 0.0
    if max_val > 0:
        onset_env = onset_env / max_val
    peaks = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,
        delta=delta,
        wait=max(1, int(0.055 * sr / hop_length))
    )
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    strengths = onset_env[peaks] if len(peaks) > 0 else np.array([])
    return times, strengths


def _build_drum_musicxml(
    title: str,
    bpm: int,
    measures_events: List[List[tuple[int, set]]],
    lyrics_by_measure: Dict[int, List[str]]
) -> str:
    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = title

    identification = ET.SubElement(score, "identification")
    creator = ET.SubElement(identification, "creator", type="composer")
    creator.text = "DrumScribe AI (Demucs + Whisper)"

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Bateria"

    instruments = [
        ("P1-I36", "Bass Drum 1", 36),
        ("P1-I39", "Acoustic Snare", 38),
        ("P1-I43", "Closed Hi-Hat", 42),
        ("P1-I48", "Low-Mid Tom", 47),
        ("P1-I50", "Crash Cymbal 1", 49),
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
        "kick":  ("F", "4", "normal", "P1-I36"),
        "snare": ("C", "5", "normal", "P1-I39"),
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

        for dur_steps, hits in events:
            note_type, dot_flag = dur_type_map.get(dur_steps, ("16th", None))
            if not hits:
                n_el = ET.SubElement(measure, "note")
                ET.SubElement(n_el, "rest")
                ET.SubElement(n_el, "duration").text = str(dur_steps)
                ET.SubElement(n_el, "voice").text = "1"
                ET.SubElement(n_el, "type").text = note_type
                if dot_flag:
                    ET.SubElement(n_el, "dot")
            else:
                ordered = [p for p in ["kick", "snare", "tom", "hihat", "crash"] if p in hits]
                for idx_h, p in enumerate(ordered):
                    step, octave, nhead, inst_id = piece_spec[p]
                    n_el = ET.SubElement(measure, "note")
                    if idx_h > 0:
                        ET.SubElement(n_el, "chord")
                    unp = ET.SubElement(n_el, "unpitched")
                    ET.SubElement(unp, "display-step").text = step
                    ET.SubElement(unp, "display-octave").text = octave
                    ET.SubElement(n_el, "duration").text = str(dur_steps)
                    ET.SubElement(n_el, "instrument", id=inst_id)
                    ET.SubElement(n_el, "voice").text = "1"
                    ET.SubElement(n_el, "type").text = note_type
                    if dot_flag:
                        ET.SubElement(n_el, "dot")
                    ET.SubElement(n_el, "stem").text = "up"
                    if nhead != "normal":
                        ET.SubElement(n_el, "notehead").text = nhead

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
        progress_cb(60, "Analisando andamento (BPM) e transientes da bateria...")

    y, sr = librosa.load(drums_wav, sr=22050, mono=True)
    duration_sec = librosa.get_duration(y=y, sr=sr)

    tempo_est, beat_frames = librosa.beat.beat_track(y=y, sr=sr, start_bpm=128.0)
    bpm = float(np.atleast_1d(tempo_est)[0])
    if bpm < 60 or bpm > 210 or math.isnan(bpm):
        bpm = 120.0
    bpm = round(bpm)

    sec_per_beat = 60.0 / bpm
    sec_per_16th = sec_per_beat / 4.0
    sec_per_measure = sec_per_beat * 4.0

    y_kick = _bandpass_filter(y, sr, 40.0, 130.0)
    y_snare = _bandpass_filter(y, sr, 180.0, 2400.0)
    y_cymbal = _highpass_filter(y, sr, 5500.0)

    kick_times, _ = _detect_band_onsets(y_kick, sr, delta=0.13)
    snare_times, _ = _detect_band_onsets(y_snare, sr, delta=0.14)
    cymbal_times, cymbal_strengths = _detect_band_onsets(y_cymbal, sr, delta=0.09)

    all_onsets = np.concatenate([kick_times, snare_times, cymbal_times]) if (
        len(kick_times) + len(snare_times) + len(cymbal_times) > 0
    ) else np.array([0.0])

    first_hit = float(np.min(all_onsets))
    if len(beat_frames) > 0:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        valid_beats = beat_times[beat_times >= max(0.0, first_hit - 0.15)]
        grid_origin = float(valid_beats[0]) if len(valid_beats) > 0 else first_hit
    else:
        grid_origin = first_hit

    effective_duration = max(sec_per_measure, duration_sec - grid_origin)
    total_measures = max(1, int(math.ceil(effective_duration / sec_per_measure)))
    total_steps = total_measures * 16

    grid: Dict[int, set] = {}

    for t in kick_times:
        if t < grid_origin - 0.05:
            continue
        step = int(round((t - grid_origin) / sec_per_16th))
        if 0 <= step < total_steps:
            grid.setdefault(step, set()).add("kick")

    for t in snare_times:
        if t < grid_origin - 0.05:
            continue
        step = int(round((t - grid_origin) / sec_per_16th))
        if 0 <= step < total_steps:
            grid.setdefault(step, set()).add("snare")

    for t, strength in zip(cymbal_times, cymbal_strengths):
        if t < grid_origin - 0.05:
            continue
        step = int(round((t - grid_origin) / sec_per_16th))
        if 0 <= step < total_steps:
            if strength > 0.72 and (step % 16 in (0, 8)):
                grid.setdefault(step, set()).add("crash")
            else:
                grid.setdefault(step, set()).add("hihat")

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
    midi_pitch_map = {"kick": "C2", "snare": "D2", "hihat": "F#2", "crash": "C#3"}

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

    xml_content = _build_drum_musicxml(title, bpm, measures_events, lyrics_by_measure)
    xml_path = os.path.join(job_dir, "partitura_bateria.musicxml")
    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    mid_path = os.path.join(job_dir, "partitura_bateria.mid")
    midi_score.write("midi", fp=mid_path)

    return {
        "bpm": bpm,
        "measures": total_measures,
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
