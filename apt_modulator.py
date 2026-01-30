#!/usr/bin/env python3
import argparse
import math
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
    amplitude_floor: float = 0.0,
) -> None:
    """
    gray: (H, W) float in [0,1]
    For each line: produce fs*seconds_per_line samples of AM carrier,
    where amplitude follows pixel intensity across the line.
    amplitude_floor: optional small bias so dark pixels still carry some signal.
                     0.0 means pure intensity amplitude.
    """
    if gray.ndim != 2:
        raise ValueError("Expected a 2D grayscale array (H, W).")

    height, width = gray.shape
    samples_per_line = int(round(fs * seconds_per_line))
    if samples_per_line <= 0:
        raise ValueError("seconds_per_line too small for given sample rate.")

    silence_seconds = 1.0
    silence_samples = int(round(fs * silence_seconds))

    total_samples = silence_samples + height * samples_per_line
    audio = np.zeros(total_samples, dtype=np.float32)

    # Precompute time vector for one line
    t = np.arange(samples_per_line, dtype=np.float32) / float(fs)
    carrier = np.sin(2.0 * math.pi * carrier_hz * t).astype(np.float32)

    # For mapping pixels -> samples, use interpolation across the line:
    # sample positions across pixel indices [0, width-1]
    x_pixels = np.arange(width, dtype=np.float32)
    x_samples = np.linspace(0.0, float(width - 1), num=samples_per_line, dtype=np.float32)

    for row in range(height):
        intens = gray[row]  # (W,) in [0,1]
        amp = np.interp(x_samples, x_pixels, intens).astype(np.float32)  # (samples_per_line,)
        if amplitude_floor != 0.0:
            amp = amplitude_floor + (1.0 - amplitude_floor) * amp

        line_audio = amp * carrier  # AM: amplitude follows pixel intensity
        start = silence_samples + row * samples_per_line
        audio[start : start + samples_per_line] = line_audio

    # Normalize to int16 safely (avoid clipping, preserve relative levels)
    peak = float(np.max(np.abs(audio))) if total_samples > 0 else 0.0
    if peak < 1e-12:
        pcm = np.zeros(total_samples, dtype=np.int16)
    else:
        audio = audio / peak
        pcm = np.clip(audio * 32767.0, -32768.0, 32767.0).astype(np.int16)

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
    parser.add_argument(
        "--amplitude-floor",
        type=float,
        default=0.0,
        help="Optional minimum amplitude (0..1). Example 0.1 keeps some carrier in dark pixels.",
    )
    args = parser.parse_args()

    if args.fs <= 0:
        raise SystemExit("fs must be > 0")
    if args.width <= 0:
        raise SystemExit("width must be > 0")
    if args.line_seconds <= 0:
        raise SystemExit("line-seconds must be > 0")
    if not (0.0 <= args.amplitude_floor <= 1.0):
        raise SystemExit("amplitude-floor must be between 0 and 1")

    gray = load_and_prepare_image(args.input_png, args.width)
    synthesize_am_wav(
        gray=gray,
        fs=args.fs,
        carrier_hz=args.carrier,
        seconds_per_line=args.line_seconds,
        out_wav_path=args.output_wav,
        amplitude_floor=args.amplitude_floor,
    )


if __name__ == "__main__":
    main()
