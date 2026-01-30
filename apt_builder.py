#!/usr/bin/env python3

# https://web.archive.org/web/20070316190349/http://www2.ncdc.noaa.gov/docs/klm/html/c4/sec4-2.htm#f423-3

import argparse
import numpy as np
from PIL import Image

# Fs_video = 4160 samples/sec, so 0.5 sec/line => 2080 samples/line.
# We are building a *raster image* that corresponds to those 2080 samples per line.
FS_VIDEO = 4160
LOW = 11
HIGH = 244

# Keep your original layout widths (classic-ish APT sizing)
SYNC_WIDTH = 39
SPACE_WIDTH = 47
IMG_WIDTH = 909
TLM_WIDTH = 45

# Blocks 10..15: choose from wedge levels 1..8 (random-ish but constant)
CAL_BLOCK_10 = 2
CAL_BLOCK_11 = 5
CAL_BLOCK_12 = 7
CAL_BLOCK_13 = 3
CAL_BLOCK_14 = 6
CAL_BLOCK_15 = 1

# Block 16 sensor identifiers (must be 1..6 per your note)
SENSOR_ID_A = 3   # typically wedge 2 or 3
SENSOR_ID_B = 4   # typically wedge 4

TELEM_BLOCK_LINES = 8     # each wedge/cal block is 8 lines tall
TELEM_BLOCKS = 16         # total blocks
TELEM_PERIOD = TELEM_BLOCK_LINES * TELEM_BLOCKS  # 128 lines


def wedge_level(k: int) -> int:
    """
    k in [0..8], where:
      0 => LOW (zero intensity)
      1..8 => LOW + (k/8)*(HIGH-LOW)
    """
    if k < 0 or k > 8:
        raise ValueError("wedge level k must be in 0..8")
    return int(round(LOW + (k / 8.0) * (HIGH - LOW)))


def minute_marker_level(line):
    """
    Returns 'black', 'white', or 'normal'
    """
    m = line % 120

    if m < 2:
        return "black"

    if m < 4:
        return "white"

    return "normal"



def make_sync_a(height):
    one_line = np.full(SYNC_WIDTH, LOW, dtype=np.uint8)

    start = 4
    cycles = 7
    samples_per_cycle = FS_VIDEO // 1040  # 4

    tone_len = cycles * samples_per_cycle  # 28
    end = start + tone_len  # 32

    one_cycle = np.array([LOW, LOW, HIGH, HIGH], dtype=np.uint8)
    one_line[start:end] = np.tile(one_cycle, cycles)

    # remaining samples (32..38) stay LOW

    return np.repeat(one_line[np.newaxis, :], height, axis=0)



def make_sync_b(height):
    one_line = np.full(SYNC_WIDTH, LOW, dtype=np.uint8)

    start = 4
    samples_per_period = FS_VIDEO // 832  # 5
    periods = 7  # exactly fills after lead-in

    one_period = np.array([HIGH, HIGH, HIGH, LOW, LOW], dtype=np.uint8)
    one_line[start:start + periods * samples_per_period] = np.tile(one_period, periods)

    return np.repeat(one_line[np.newaxis, :], height, axis=0)



def make_telemetry(height: int, sensor_id_wedge: int) -> np.ndarray:
    """
    Telemetry column (wedge blocks), repeating every 128 lines.
    Each block spans 8 lines and has constant intensity across the telemetry width.

    Blocks:
      1..8  : wedge levels 1/8..8/8
      9     : 0 intensity
      10..15: calibration values (constants chosen from wedge levels 1..8)
      16    : sensor id (matches wedge level 1..6)
    """
    if sensor_id_wedge < 1 or sensor_id_wedge > 6:
        raise ValueError("sensor_id_wedge must be in 1..6")

    # Precompute block intensities (1-indexed in the spec, but stored 0-indexed here)
    block_levels = []

    # 1..8
    for k in range(1, 9):
        block_levels.append(wedge_level(k))

    # 9 (zero)
    block_levels.append(wedge_level(0))

    # 10..15 (calibration constants, each is a wedge index 1..8)
    cal_indices = [
        CAL_BLOCK_10, CAL_BLOCK_11, CAL_BLOCK_12,
        CAL_BLOCK_13, CAL_BLOCK_14, CAL_BLOCK_15,
    ]
    for idx in cal_indices:
        if idx < 1 or idx > 8:
            raise ValueError("CAL_BLOCK_* must be in 1..8")
        block_levels.append(wedge_level(idx))

    # 16 (sensor id)
    block_levels.append(wedge_level(sensor_id_wedge))

    if len(block_levels) != 16:
        raise RuntimeError("Telemetry block_levels must be 16 entries")

    tlm = np.zeros((height, TLM_WIDTH), dtype=np.uint8)

    for line in range(height):
        within = line % TELEM_PERIOD               # 0..127
        block = within // TELEM_BLOCK_LINES        # 0..15
        tlm[line, :] = block_levels[block]

    return tlm



def prepare_image(path):
    img = Image.open(path).convert("L")

    aspect = img.width / img.height
    new_width = IMG_WIDTH
    new_height = int(round(new_width / aspect))

    img = img.resize((new_width, new_height), Image.LANCZOS)

    return np.array(img, dtype=np.uint8)



def make_space_columns(height: int):
    """
    SPACE A and MARKER / SPACE B and MARKER

    Normal:
      - A is black (LOW)
      - B is white (HIGH)

    Each minute (120 lines @ 2 lines/sec):
      - 2 black marker lines: visible in B (force B to LOW), A remains normal
      - 2 white marker lines: visible in A (force A to HIGH), B remains normal
    """
    # Normal backgrounds
    space_a = np.full((height, SPACE_WIDTH), LOW, dtype=np.uint8)    # black
    space_b = np.full((height, SPACE_WIDTH), HIGH, dtype=np.uint8)   # white

    for line in range(height):
        m = line % 120

        if m < 2:
            # "2 black lines" -> visible in B only
            space_b[line, :] = LOW

        elif m < 4:
            # "2 white lines" -> visible in A only
            space_a[line, :] = HIGH

        # else: normal (already set)

    return space_a, space_b



def build_channel(
    image: np.ndarray,
    which: str,
    space_a: np.ndarray,
    space_b: np.ndarray,
) -> np.ndarray:
    h = image.shape[0]

    if which == "a":
        sync = make_sync_a(h)
        space = space_a
        tlm = make_telemetry(h, sensor_id_wedge=SENSOR_ID_A)
    elif which == "b":
        sync = make_sync_b(h)
        space = space_b
        tlm = make_telemetry(h, sensor_id_wedge=SENSOR_ID_B)
    else:
        raise ValueError("which must be 'a' or 'b'")

    return np.hstack([sync, space, image, tlm])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_png")
    parser.add_argument("output_png")
    parser.add_argument("--height", type=int, default=2000)
    args = parser.parse_args()

    image = prepare_image(args.input_png, args.height)

    # NEW: build SPACE/MARKER columns once, shared timing across A/B
    space_a, space_b = make_space_columns(args.height)

    # Build channels with correct sync + correct space column
    chA = build_channel(image, "a", space_a=space_a, space_b=space_b)
    chB = build_channel(image, "b", space_a=space_a, space_b=space_b)

    apt = np.hstack([chA, chB])

    print("APT width:", apt.shape[1])  # should be 2080 with your widths
    Image.fromarray(apt).save(args.output_png)


if __name__ == "__main__":
    main()
