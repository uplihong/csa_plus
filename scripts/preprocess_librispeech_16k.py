#!/usr/bin/env python3
"""
Preprocess LibriSpeech by resampling all .flac audio to a target sample rate.
This creates a new dataset root with the same structure and transcript files.
"""

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, List, Tuple

import librosa
import soundfile as sf


def _resample_one(args: Tuple[str, str, str, int, bool, bool, float]) -> Tuple[str, str, int, float]:
    src_path, dst_path, rel_path, target_sr, overwrite, trim, trim_top_db = args
    if os.path.exists(dst_path) and not overwrite:
        info = sf.info(dst_path)
        num_samples = int(info.frames)
        duration_sec = float(num_samples / info.samplerate) if info.samplerate else 0.0
        return "skip", rel_path, num_samples, duration_sec

    audio, sr = sf.read(src_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)

    if trim:
        audio, _ = librosa.effects.trim(audio, top_db=trim_top_db)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    sf.write(dst_path, audio, target_sr)
    num_samples = int(audio.shape[0])
    duration_sec = float(num_samples / target_sr) if target_sr else 0.0
    return "ok", rel_path, num_samples, duration_sec


def _iter_files(root: str) -> Tuple[Iterable[str], Iterable[str]]:
    flac_files = []
    txt_files = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.endswith(".flac"):
                flac_files.append(os.path.join(dirpath, name))
            elif name.endswith(".txt"):
                txt_files.append(os.path.join(dirpath, name))
    return flac_files, txt_files


def _copy_txt_files(txt_files: Iterable[str], input_root: str, output_root: str) -> None:
    for src_path in txt_files:
        rel_path = os.path.relpath(src_path, input_root)
        dst_path = os.path.join(output_root, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample LibriSpeech FLAC files to target sample rate.")
    parser.add_argument("--input-root", required=True, help="Original LibriSpeech root directory")
    parser.add_argument("--output-root", required=True, help="Output directory for resampled dataset")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate (default: 16000)")
    parser.add_argument("--trim", action="store_true", help="Apply silence trim after resampling")
    parser.add_argument("--trim-top-db", type=float, default=60.0, help="Top-dB threshold for librosa trim")
    parser.add_argument("--manifest-path", default=None, help="Optional TSV manifest output path")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--log-every", type=int, default=500, help="Log progress every N files")
    args = parser.parse_args()

    input_root = os.path.abspath(args.input_root)
    output_root = os.path.abspath(args.output_root)

    if not os.path.isdir(input_root):
        raise SystemExit(f"Input root does not exist: {input_root}")

    flac_files, txt_files = _iter_files(input_root)
    if not flac_files:
        raise SystemExit("No .flac files found under input root.")

    _copy_txt_files(txt_files, input_root, output_root)

    tasks = []
    for src_path in flac_files:
        rel_path = os.path.relpath(src_path, input_root)
        dst_path = os.path.join(output_root, rel_path)
        tasks.append((src_path, dst_path, rel_path, args.target_sr, args.overwrite, args.trim, args.trim_top_db))

    processed = 0
    manifest_rows: List[Tuple[str, int, float]] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for status, rel_path, num_samples, duration_sec in executor.map(_resample_one, tasks, chunksize=16):
                processed += 1
                manifest_rows.append((rel_path, num_samples, duration_sec))
                if processed % args.log_every == 0:
                    print(f"Processed {processed}/{len(tasks)} files (last={status})")
    else:
        for task in tasks:
            status, rel_path, num_samples, duration_sec = _resample_one(task)
            processed += 1
            manifest_rows.append((rel_path, num_samples, duration_sec))
            if processed % args.log_every == 0:
                print(f"Processed {processed}/{len(tasks)} files (last={status})")

    manifest_path = args.manifest_path
    if manifest_path:
        manifest_path = os.path.abspath(manifest_path)
        manifest_dir = os.path.dirname(manifest_path)
        if manifest_dir:
            os.makedirs(manifest_dir, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("rel_path\tnum_samples\tduration_sec\n")
            for rel_path, num_samples, duration_sec in sorted(manifest_rows, key=lambda x: x[0]):
                f.write(f"{rel_path}\t{num_samples}\t{duration_sec:.6f}\n")
        print(f"Manifest written to: {manifest_path}")

    print(f"Done. Output dataset at: {output_root}")


if __name__ == "__main__":
    main()
