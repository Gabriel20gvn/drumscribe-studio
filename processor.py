import os
import re
import math
import shutil
import zipfile
import subprocess
from typing import Callable, Optional, Dict, Any

import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, sosfilt
import yt_dlp
import whisper
from music21 import stream, note, chord, meter, tempo, clef, expressions, metadata


def sanitize_filename(name: str) -> str:
    clean = re.sub(r'[\\/*?:"<>|]', "", name).strip()
    return clean[:80] if clean else "musica_drumscribe"


def run_ffmpeg_convert(input_path: str, output_path: str, bitrate: str = "192k"):
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


def download_youtube_wav(url: str, job_dir: str, cookies_path: Optional[str] = None) -> tuple[str, str]:
    raw_template = os.path.join(job_dir, "raw_audio.%(ext)s")

    # Tenta clientes que reduzem bloqueio de IP em servidores na nuvem sem exigir cookies
    client_profiles = [
        ["tv", "mweb"],
        ["android_vr", "web"],
        ["ios", "android"],
    ]

    last_error = None
    title = "musica_youtube"

    for clients in client_profiles:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": raw_template,
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "extractor_args": {
                "youtube": {
                    "player_client": clients
                }
            },
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
        if cookies_path and os.path.exists(cookies_path):
            ydl_opts["cookiefile"] = cookies_path

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

    raise RuntimeError(
        f"O YouTube bloqueou o IP do servidor para este vídeo ({last_error}). "
        "Solução rápida sem cookies: baixe o MP3/M4A no seu celular e envie no campo 'Ou envie um arquivo de áudio/vídeo'."
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
        run_ffmpeg_convert(wpath, mp3_path, bitrate="192k")
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
    if np.max(onset_env) > 0:
        onset_env = onset_env / np.max(onset_env)
    peaks = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        hop_length=hop_length,
        backtrack=False,
        delta=delta,
        wait=int(0.06 * sr / hop_length)
    )
    times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
    strengths = onset_env[peaks] if len(peaks) > 0 else np.array([])
    return times, strengths


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
        progress_cb(62, "Analisando andamento (BPM) e transientes da bateria...")

    y, sr = librosa.load(drums_wav, sr=22050, mono=True)
    duration_sec = librosa.get_duration(y=y, sr=sr)

    tempo_est, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(np.atleast_1d(tempo_est)[0])
    if bpm < 55 or bpm > 210 or math.isnan(bpm):
        bpm = 120.0
    bpm = round(bpm)

    sec_per_beat = 60.0 / bpm
    sec_per_16th = sec_per_beat / 4.0
    sec_per_measure = sec_per_beat * 4.0
    total_measures = max(1, int(math.ceil(duration_sec / sec_per_measure)))
    total_steps = total_measures * 16

    y_kick = _bandpass_filter(y, sr, 40.0, 130.0)
    y_snare = _bandpass_filter(y, sr, 180.0, 2400.0)
    y_cymbal = _highpass_filter(y, sr, 6000.0)

    kick_times, _ = _detect_band_onsets(y_kick, sr, delta=0.14)
    snare_times, _ = _detect_band_onsets(y_snare, sr, delta=0.15)
    cymbal_times, cymbal_strengths = _detect_band_onsets(y_cymbal, sr, delta=0.10)

    grid: Dict[int, set] = {}

    for t in kick_times:
        step = int(round(t / sec_per_16th))
        if 0 <= step < total_steps:
            grid.setdefault(step, set()).add("kick")

    for t in snare_times:
        step = int(round(t / sec_per_16th))
        if 0 <= step < total_steps:
            grid.setdefault(step, set()).add("snare")

    for t, strength in zip(cymbal_times, cymbal_strengths):
        step = int(round(t / sec_per_16th))
        if 0 <= step < total_steps:
            if strength > 0.72 and (step % 16 in (0, 8)):
                grid.setdefault(step, set()).add("crash")
            else:
                grid.setdefault(step, set()).add("hihat")

    lyrics_by_measure: Dict[int, list[str]] = {}
    if vocals_wav and os.path.exists(vocals_wav):
        if progress_cb:
            progress_cb(78, f"Transcrevendo letra com Whisper ({whisper_model_name}) e alinhando compassos...")
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
                        m_idx = int(w_start // sec_per_measure)
                        lyrics_by_measure.setdefault(m_idx, []).append(w_text)
            else:
                s_start = float(seg.get("start", 0.0))
                s_text = seg.get("text", "").strip()
                if s_text:
                    m_idx = int(s_start // sec_per_measure)
                    lyrics_by_measure.setdefault(m_idx, []).append(s_text)

    if progress_cb:
        progress_cb(90, "Montando partitura MusicXML com clave de percussão e letras...")

    score = stream.Score()
    score.insert(0, metadata.Metadata())
    score.metadata.title = title
    score.metadata.composer = "DrumScribe AI (Demucs + Whisper)"

    part = stream.Part()
    part.id = "Drums"
    part.partName = "Bateria"

    pitch_map = {
        "kick": ("F4", "normal"),
        "snare": ("C5", "normal"),
        "hihat": ("G5", "x"),
        "crash": ("A5", "x"),
    }

    for m_idx in range(total_measures):
        m = stream.Measure(number=m_idx + 1)
        if m_idx == 0:
            m.insert(0.0, clef.PercussionClef())
            m.insert(0.0, meter.TimeSignature("4/4"))
            m.insert(0.0, tempo.MetronomeMark(number=bpm))

        if m_idx in lyrics_by_measure:
            frase = " ".join(lyrics_by_measure[m_idx])
            txt_exp = expressions.TextExpression(frase)
            txt_exp.placement = "above"
            txt_exp.style.fontSize = 9.5
            txt_exp.style.fontStyle = "italic"
            m.insert(0.0, txt_exp)

        base_step = m_idx * 16
        step_offset = 0
        while step_offset < 16:
            abs_step = base_step + step_offset
            hits = grid.get(abs_step, set())

            span = 1
            while (step_offset + span) < 16 and not grid.get(base_step + step_offset + span):
                span += 1

            if span >= 4 and (step_offset % 4 == 0):
                dur_steps = 4
            elif span >= 2 and (step_offset % 2 == 0):
                dur_steps = 2
            else:
                dur_steps = 1

            quarter_len = dur_steps * 0.25

            if not hits:
                m.append(note.Rest(quarterLength=quarter_len))
            elif len(hits) == 1:
                piece = next(iter(hits))
                p_name, n_head = pitch_map[piece]
                n = note.Note(p_name, quarterLength=quarter_len)
                if n_head != "normal":
                    n.notehead = n_head
                m.append(n)
            else:
                ordered_pieces = [p for p in ["kick", "snare", "hihat", "crash"] if p in hits]
                c = chord.Chord([pitch_map[p][0] for p in ordered_pieces], quarterLength=quarter_len)
                for idx_n, p in enumerate(ordered_pieces):
                    if pitch_map[p][1] != "normal":
                        c.noteheads[idx_n] = pitch_map[p][1]
                m.append(c)

            step_offset += dur_steps

        part.append(m)

    score.append(part)

    xml_path = os.path.join(job_dir, "partitura_bateria.musicxml")
    mid_path = os.path.join(job_dir, "partitura_bateria.mid")
    score.write("musicxml", fp=xml_path)
    score.write("midi", fp=mid_path)

    return {
        "bpm": bpm,
        "measures": total_measures,
        "musicxml": "partitura_bateria.musicxml",
        "midi": "partitura_bateria.mid",
    }


def process_full_pipeline(
    url: str,
    job_dir: str,
    demucs_model: str = "htdemucs",
    whisper_model: str = "small",
    language: str = "pt",
    cookies_path: Optional[str] = None,
    uploaded_media_path: Optional[str] = None,
    uploaded_media_name: Optional[str] = None,
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
        progress_cb(96, "Empacotando arquivos WAV, MIDI e MusicXML...")

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

    return {
        "title": title,
        "bpm": score_info["bpm"],
        "measures": score_info["measures"],
        "musicxml": score_info["musicxml"],
        "midi": score_info["midi"],
        "zip": zip_filename,
        "audio_mp3": stems["mp3"],
    }
