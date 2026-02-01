#!/usr/bin/env python3
import argparse
import wave
from pathlib import Path
import numpy as np
from PIL import Image


def load_and_prepare_image(path: Path, target_width: int) -> np.ndarray:
    """
    Returns grayscale image as float32 array in [0, 1] with shape (H, target_width).
    """
    img = Image.open(path).convert("L")  # grayscale
    w0, h0 = img.size
    if w0 != target_width:
        target_height = max(1, round(h0 * (target_width / float(w0))))
        img = img.resize((target_width, target_height), resample=Image.LANCZOS)

    arr_u8 = np.array(img, dtype=np.uint8)          # (H, W)
    arr = (arr_u8.astype(np.float32) / 255.0)       # (H, W) in [0,1]
    return arr


def synthesize_am_wav(
    gray: np.ndarray,
    fs: int,
    carrier_hz: float,
    seconds_per_line: float,
    out_wav_path: Path,
) -> None:
    """
    gray: (H, W) float in [0,1]
    For each line: produce fs*seconds_per_line samples of AM carrier,
    where amplitude follows pixel intensity across the line.
    """
    height, width = gray.shape
    samples_per_line = int(round(fs * seconds_per_line))

    audio_sample_count = height * samples_per_line
    audio = np.zeros(audio_sample_count, dtype=np.float32)

    # For mapping pixels -> samples, use interpolation across the line:
    # sample positions across pixel indices [0, width-1]
    x_pixels = np.arange(width, dtype=np.float32)
    x_samples = np.linspace(0.0, float(width - 1), num=samples_per_line, dtype=np.float32)

    # work with arrays of length equal to samples_per_line to keep things fast
    # with numpy. This will hold [0...samples_per_line-1] at the beginning and 
    # each element incremented the end of the loop body
    sample_indices_for_row = np.arange(samples_per_line, dtype=np.int64)

    for row in range(height):
        t_row = sample_indices_for_row / fs

        # since t_row is continouse, we get a phase continous carrier here:
        carrier = np.sin(2.0 * np.pi * carrier_hz * t_row).astype(np.float32)

        amp = np.interp(x_samples, x_pixels, gray[row]).astype(np.float32)

        am_modulated_row = amp * carrier
        
        start = sample_indices_for_row[0]
        audio[start:start+samples_per_line] = am_modulated_row
        
        sample_indices_for_row += samples_per_line


    peak = float(np.max(np.abs(audio))) if audio_sample_count > 0 else 0.0
    if peak < 1e-12:
        pcm = np.zeros(audio_sample_count, dtype=np.int16)
    else:
        audio = audio / peak
        pcm = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)

    # 1 seconds of silence before and after
    silence_samples = np.zeros(fs, dtype=np.int16)
    pcm = np.concatenate([silence_samples, pcm, silence_samples])

    # Write WAV (mono, 16-bit)
    with wave.open(str(out_wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(pcm.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transmit an image by AM-modulating pixel intensities onto a 2400 Hz carrier."
    )
    parser.add_argument("input_png", type=Path, help="Input PNG path")
    parser.add_argument("output_wav", type=Path, help="Output WAV path")
    parser.add_argument("--width", type=int, default=2080, help="Target width in pixels (default: 2080)")
    parser.add_argument("--fs", type=int, default=48000, help="Sample rate (default: 48000)")
    parser.add_argument("--carrier", type=float, default=2400.0, help="Carrier frequency in Hz (default: 2400)")
    parser.add_argument("--line-seconds", type=float, default=0.5, help="Seconds per line (default: 0.5)")
    args = parser.parse_args()

    if args.fs <= 0:
        raise SystemExit("fs must be > 0")
    if args.width <= 0:
        raise SystemExit("width must be > 0")
    if args.line_seconds <= 0:
        raise SystemExit("line-seconds must be > 0")

    gray = load_and_prepare_image(args.input_png, args.width)
    synthesize_am_wav(
        gray=gray,
        fs=args.fs,
        carrier_hz=args.carrier,
        seconds_per_line=args.line_seconds,
        out_wav_path=args.output_wav,
    )


if __name__ == "__main__":
    main()
