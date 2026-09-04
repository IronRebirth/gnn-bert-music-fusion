"""
Audio feature extraction: load, resample, segment, extract log-mel & chroma.
Saves per-track feature dictionaries as .pt files.
"""

import os
import warnings

import numpy as np
import torch
import librosa
from tqdm import tqdm


def extract_audio_features(
    file_path: str,
    sr: int = 22050,
    segment_length_s: float = 5.0,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_chroma: int = 12,
    max_segments: int = 6,
) -> list[dict]:
    """
    Load audio, resample, segment into fixed windows, extract features.

    Returns list of dicts, each with:
      'log_mel': np.ndarray [n_mels, T]
      'chroma': np.ndarray [n_chroma, T]
      'mel_mean': np.ndarray [n_mels]   (time-averaged, for graph nodes)
      'chroma_mean': np.ndarray [n_chroma]
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(file_path, sr=sr, mono=True)

    if len(y) == 0:
        raise ValueError(f"Empty audio: {file_path}")

    segment_samples = int(segment_length_s * sr)
    segments = []

    for i, start in enumerate(range(0, len(y), segment_samples)):
        if i >= max_segments:
            break

        seg = y[start : start + segment_samples]
        if len(seg) < segment_samples // 2:
            # Skip very short trailing segments
            continue
        if len(seg) < segment_samples:
            seg = np.pad(seg, (0, segment_samples - len(seg)))

        # Log-mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=seg, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)

        # Per-segment normalisation
        mean, std = log_mel.mean(), log_mel.std()
        log_mel = (log_mel - mean) / (std + 1e-8)

        # Chroma
        chroma = librosa.feature.chroma_stft(
            y=seg, sr=sr, n_chroma=n_chroma, n_fft=n_fft, hop_length=hop_length
        )

        segments.append({
            "log_mel": log_mel.astype(np.float32),
            "chroma": chroma.astype(np.float32),
            "mel_mean": log_mel.mean(axis=1).astype(np.float32),
            "chroma_mean": chroma.mean(axis=1).astype(np.float32),
        })

    if not segments:
        raise ValueError(f"No segments extracted: {file_path}")

    return segments


def process_track(
    track_id: int,
    raw_audio_dir: str,
    output_dir: str,
    config: dict,
) -> bool:
    """
    Process a single track: extract features and save as .pt file.
    Returns True on success.
    """
    # FMA directory layout: raw_audio_dir / <3-digit prefix> / <6-digit>.mp3
    tid_str = f"{track_id:06d}"
    prefix = tid_str[:3]
    mp3_path = os.path.join(raw_audio_dir, prefix, f"{tid_str}.mp3")

    if not os.path.exists(mp3_path):
        return False

    out_path = os.path.join(output_dir, f"{tid_str}_features.pt")
    if os.path.exists(out_path):
        return True  # already processed

    ds = config["dataset"]
    try:
        segments = extract_audio_features(
            mp3_path,
            sr=ds["sample_rate"],
            segment_length_s=ds["segment_length_s"],
            n_mels=ds["n_mels"],
            n_fft=ds["n_fft"],
            hop_length=ds["hop_length"],
            n_chroma=ds["n_chroma"],
            max_segments=ds["max_segments_per_track"],
        )
        torch.save(segments, out_path)
        return True
    except Exception as e:
        print(f"  [WARN] Failed {tid_str}: {e}")
        return False


def batch_process_tracks(
    track_ids: list[int],
    raw_audio_dir: str,
    output_dir: str,
    config: dict,
    desc: str = "Extracting audio features",
):
    """Process a list of track IDs with a progress bar."""
    os.makedirs(output_dir, exist_ok=True)
    ok, fail = 0, 0
    for tid in tqdm(track_ids, desc=desc):
        if process_track(tid, raw_audio_dir, output_dir, config):
            ok += 1
        else:
            fail += 1
    print(f"Audio features: {ok} succeeded, {fail} failed out of {len(track_ids)}")
