#!/usr/bin/env python3
import argparse
import math
import wave
from pathlib import Path

import numpy as np


def read_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """
    Read a WAV file and return (audio_float32, sample_rate).

    audio_float32 is mono in [-1, 1] (approximately). Stereo inputs are downmixed.
    Supports PCM with sample widths 8/16/24/32-bit.
    """
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        fs = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 1:
        # 8-bit PCM is unsigned
        x = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        x = (x - 128.0) / 128.0
    elif sampwidth == 2:
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        x = x / 32768.0
    elif sampwidth == 3:
        # 24-bit PCM little-endian -> int32 with sign extension
        b = np.frombuffer(raw, dtype=np.uint8)
        if (b.size % 3) != 0:
            raise ValueError("24-bit PCM byte count is not divisible by 3")
        b = b.reshape(-1, 3)
        x = (b[:, 0].astype(np.int32) |
             (b[:, 1].astype(np.int32) << 8) |
             (b[:, 2].astype(np.int32) << 16))
        # sign extend 24->32
        sign = x & 0x800000
        x = x - (sign << 1)
        x = x.astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        # Assume 32-bit PCM signed (not float WAV). If yours is float WAV, wave module
        # won’t tell you; you’d typically use soundfile/scipy. This covers common PCM.
        x = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        x = x / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    if n_channels > 1:
        if (x.size % n_channels) != 0:
            raise ValueError("WAV data size is not divisible by channel count")
        x = x.reshape(-1, n_channels).mean(axis=1)

    # Avoid NaNs if file is silent; keep in reasonable range
    x = np.clip(x, -1.0, 1.0).astype(np.float32)
    return x, fs


def fm_modulate_iq_cs16(
    audio: np.ndarray,
    fs: int,
    deviation_hz: float,
    out_path: Path,
    carrier_offset_hz: float = 0.0,
    iq_gain: float = 0.9,
) -> None:
    """
    FM modulate audio into complex baseband IQ and write CS16 (interleaved int16).

    Instantaneous frequency:
        f_inst[n] = carrier_offset_hz + deviation_hz * audio[n]

    Phase accumulation:
        phase[n] = phase[n-1] + 2π * f_inst[n] / fs

    Output:
        int16 little-endian interleaved IQ: I0,Q0,I1,Q1,...

    iq_gain scales the final IQ amplitude before int16 conversion (<= 1.0 recommended).
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if deviation_hz < 0:
        raise ValueError("deviation_hz must be >= 0")
    if not (0.0 < iq_gain <= 1.0):
        raise ValueError("iq_gain must be in (0, 1]")

    audio = audio.astype(np.float32)
    # Optional: ensure audio is not crazy hot; FM expects roughly [-1, 1]
    audio = np.clip(audio, -1.0, 1.0)

    f_inst = carrier_offset_hz + deviation_hz * audio  # Hz
    phase_inc = (2.0 * math.pi * f_inst) / float(fs)   # rad/sample
    phase = np.cumsum(phase_inc, dtype=np.float64)

    i = np.cos(phase).astype(np.float32) * iq_gain
    q = np.sin(phase).astype(np.float32) * iq_gain

    # Interleave to CS16
    scale = 32767.0
    iq = np.empty(i.size * 2, dtype=np.int16)
    iq[0::2] = np.clip(i * scale, -32768.0, 32767.0).astype(np.int16)
    iq[1::2] = np.clip(q * scale, -32768.0, 32767.0).astype(np.int16)

    out_path.write_bytes(iq.tobytes())


def fm_modulate_iq_cf32(
    audio: np.ndarray,
    fs: int,
    deviation_hz: float,
    out_path: Path,
    carrier_offset_hz: float = 0.0,
    iq_gain: float = 0.9,
) -> None:
    """
    FM modulate audio into complex baseband IQ and write CF32 (interleaved float32).

    Output format:
        float32 little-endian interleaved IQ: I0,Q0,I1,Q1,...

    iq_gain scales the IQ magnitude (<= 1.0 recommended).
    """
    if fs <= 0:
        raise ValueError("fs must be > 0")
    if deviation_hz < 0:
        raise ValueError("deviation_hz must be >= 0")
    if not (0.0 < iq_gain <= 1.0):
        raise ValueError("iq_gain must be in (0, 1]")

    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)

    f_inst = carrier_offset_hz + deviation_hz * audio  # Hz
    phase_inc = (2.0 * math.pi * f_inst) / float(fs)   # rad/sample
    phase = np.cumsum(phase_inc, dtype=np.float64)

    i = (np.cos(phase).astype(np.float32) * iq_gain)
    q = (np.sin(phase).astype(np.float32) * iq_gain)

    iq = np.empty(i.size * 2, dtype=np.float32)
    iq[0::2] = i
    iq[1::2] = q

    out_path.write_bytes(iq.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate FM-modulated complex baseband IQ (CS16) from a WAV audio file."
    )
    parser.add_argument("input_wav", type=Path, help="Input WAV (PCM) file")
    parser.add_argument("output_cs16", type=Path, help="Output IQ file (CS16 interleaved int16)")

    parser.add_argument(
        "--deviation-hz",
        type=float,
        default=5000.0,
        help="FM deviation in Hz for audio=±1.0 (default: 5000)",
    )
    parser.add_argument(
        "--carrier-offset-hz",
        type=float,
        default=0.0,
        help="Optional frequency offset (Hz) to shift the FM spectrum (default: 0)",
    )
    parser.add_argument(
        "--iq-gain",
        type=float,
        default=0.9,
        help="Output IQ amplitude scaling before int16 conversion (0..1, default: 0.9)",
    )

    parser.add_argument(
        "--format",
        choices=["cs16", "cf32"],
        default="cs16",
        help="Output IQ sample format (default: cs16)",
    )


    args = parser.parse_args()
    audio, fs = read_wav_mono(args.input_wav)

    if args.format == "cs16":
        fm_modulate_iq_cs16(
            audio=audio,
            fs=fs,
            deviation_hz=args.deviation_hz,
            carrier_offset_hz=args.carrier_offset_hz,
            iq_gain=args.iq_gain,
            out_path=args.output_cs16,
        )
    else:
        fm_modulate_iq_cf32(
            audio=audio,
            fs=fs,
            deviation_hz=args.deviation_hz,
            carrier_offset_hz=args.carrier_offset_hz,
            iq_gain=args.iq_gain,
            out_path=args.output_cs16,
        )
        
    print(f"Output format: {args.format}")
    print(f"Wrote IQ: {args.output_cs16} ({audio.size} samples, {audio.size / fs:.3f} s)")

if __name__ == "__main__":
    main()
