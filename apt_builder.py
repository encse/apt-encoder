#!/usr/bin/env python3
"""
NOAA APT (Automatic Picture Transmission) frame compositor.

This script builds a *video-domain* APT raster (grayscale PNG) that follows the
APT line format described in the NOAA KLM User’s Guide, Section 4.2
(“Real-Time Data Systems for Local Users / APT”), including:

  - Channel A and Channel B line structure:
        SYNC  | SPACE+MARKER | IMAGE | TELEMETRY
  - Sync signals:
        Sync A: 1040 Hz reference (implemented as a 2/2 square wave at Fs=4160)
        Sync B: 832 Hz reference / pulse train (implemented as a 3/2 pattern at Fs=4160)
  - Minute markers in the SPACE/MARKER field:
        Every 60 seconds (120 lines at 2 lines/s), transmit 2 “black” lines then
        2 “white” lines, arranged so each is visible in the intended channel.
  - Telemetry “wedges” and calibration blocks:
        16 blocks per frame, each block spans 8 successive video lines (128-line frame).

The output is a convenient *intermediate representation* of the APT video levels
(0–255) suitable for later conversion into an audio or IQ waveform.
"""

# Reference (archived):
# https://web.archive.org/web/20070316190349/http://www2.ncdc.noaa.gov/docs/klm/html/c4/sec4-2.htm#f423-3

import argparse
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Video sampling geometry
# ---------------------------------------------------------------------------
# In APT, each transmitted line corresponds to 0.5 seconds of video.
# Using the canonical APT “video sample rate”:
#     Fs_video = 4160 samples/second
# gives exactly:
#     0.5 s/line * 4160 samples/s = 2080 samples per APT line
#
# The line is split into two 1040-sample channels (A then B), each composed of:
#     SYNC (39) + SPACE/MARKER (47) + IMAGE (909) + TELEMETRY (45) = 1040 samples
FS_VIDEO = 4160

# ---------------------------------------------------------------------------
# Video levels (8-bit grayscale)
# ---------------------------------------------------------------------------
# The KLM documentation describes APT video levels using a nominal 256-level scale.
# We reserve “near-black” and “near-white” for synchronization/markers to keep them
# unambiguous (common practice in APT tooling).
LOW = 11
HIGH = 244

# ---------------------------------------------------------------------------
# Per-channel field widths (samples per line)
# ---------------------------------------------------------------------------
# These widths match the standard APT line layout (per channel).
SYNC_WIDTH = 39
SPACE_WIDTH = 47
IMG_WIDTH = 909
TLM_WIDTH = 45

# ---------------------------------------------------------------------------
# Telemetry frame structure
# ---------------------------------------------------------------------------
# The telemetry region is defined as 16 blocks (“wedges” + calibration + ID),
# where each block spans 8 successive video lines. Total telemetry frame height:
#     16 blocks * 8 lines/block = 128 lines (one APT telemetry frame)
TELEM_BLOCK_LINES = 8
TELEM_BLOCKS = 16
TELEM_PERIOD = TELEM_BLOCK_LINES * TELEM_BLOCKS  # 128 lines

# Blocks 10..15 encode calibration values for the selected sensor channel.
# The spec defines them as matching one of the wedge intensities (1..8).
# For this generator we pick stable (constant) wedge indices; real hardware
# would use sensor-derived calibration values.
CAL_BLOCK_10 = 2
CAL_BLOCK_11 = 5
CAL_BLOCK_12 = 7
CAL_BLOCK_13 = 3
CAL_BLOCK_14 = 6
CAL_BLOCK_15 = 1

# Block 16 identifies which AVHRR channel produced the preceding IMAGE field by
# matching the intensity of one of wedges 1..6. Typical conventions:
#   - Video channel A matches wedge 2 or 3
#   - Video channel B matches wedge 4
SENSOR_ID_A = 3
SENSOR_ID_B = 4


def wedge_level(k: int) -> int:
    """
    Convert a wedge index into an 8-bit video level.

    The APT telemetry wedges are defined as:
      - Wedges 1..8: 1/8 to 8/8 of full-scale intensity
      - Wedge 9: zero intensity
    Here we map the fractional level into the [LOW, HIGH] range used by this generator.
    """
    if k < 0 or k > 8:
        raise ValueError("wedge level k must be in 0..8")
    return int(round(LOW + (k / 8.0) * (HIGH - LOW)))


def minute_marker_level(line: int) -> str:
    """
    Minute marker timing state for the SPACE/MARKER field.

    APT transmits 2 video lines per second. Therefore:
      60 seconds => 120 lines.

    The specification describes a four-line minute marker pattern every minute:
      - 2 lines “black”
      - followed by 2 lines “white”

    The SPACE/MARKER columns for channels A and B use opposite polarities (A normally
    black, B normally white), so the generator makes the “black” marker visible in
    channel B and the “white” marker visible in channel A.
    """
    m = line % 120
    if m < 2:
        return "black"
    if m < 4:
        return "white"
    return "normal"


def make_sync_a(height: int) -> np.ndarray:
    """
    Build the Channel A SYNC field (39 samples wide) for all lines.

    Per APT spec, Channel A begins with a 1040 Hz reference for synchronization.
    At Fs_video=4160 Hz, 1040 Hz corresponds to exactly 4 samples per cycle.
    We implement this as a square wave with two LOW samples and two HIGH samples.

    The sync field includes a short lead-in interval; the 1040 Hz pattern begins at
    sample index 4 (T=4/Fs_video) and runs for 7 cycles (28 samples).
    Any remaining samples in the 39-sample SYNC field remain at LOW.
    """
    one_line = np.full(SYNC_WIDTH, LOW, dtype=np.uint8)

    start = 4
    cycles = 7
    samples_per_cycle = FS_VIDEO // 1040  # expected 4

    tone_len = cycles * samples_per_cycle  # 28
    end = start + tone_len  # 32

    one_cycle = np.array([LOW, LOW, HIGH, HIGH], dtype=np.uint8)
    one_line[start:end] = np.tile(one_cycle, cycles)

    # Remaining SYNC samples stay LOW.
    return np.repeat(one_line[np.newaxis, :], height, axis=0)


def make_sync_b(height: int) -> np.ndarray:
    """
    Build the Channel B SYNC field (39 samples wide) for all lines.

    The Channel B sync reference is specified as an 832 Hz / 832 pps timing pattern.
    At Fs_video=4160 Hz, 832 pps corresponds to 5 samples per period.

    The specification’s pulse structure is represented here as a repeating 5-sample
    pattern [HIGH, HIGH, HIGH, LOW, LOW] (3/2 duty) starting after a 4-sample lead-in.
    With SYNC_WIDTH=39, the 4-sample lead-in plus 7 periods (7*5=35) fills the field.
    """
    one_line = np.full(SYNC_WIDTH, LOW, dtype=np.uint8)

    start = 4
    samples_per_period = FS_VIDEO // 832  # expected 5
    periods = 7  # 4 + 7*5 = 39

    one_period = np.array([HIGH, HIGH, HIGH, LOW, LOW], dtype=np.uint8)
    one_line[start:start + periods * samples_per_period] = np.tile(one_period, periods)

    return np.repeat(one_line[np.newaxis, :], height, axis=0)


def make_telemetry(height: int, sensor_id_wedge: int) -> np.ndarray:
    """
    Build the TELEMETRY field for one channel.

    The telemetry region is a repeating 128-line “frame” consisting of 16 blocks,
    each block spanning 8 successive video lines:

      Blocks  1..8 : Wedges at 1/8 .. 8/8 intensity
      Block     9  : Zero intensity
      Blocks 10..15: Sensor calibration values (encoded as one of wedge levels 1..8)
      Block    16  : Sensor/channel identifier (encoded as one of wedge levels 1..6)

    The KLM APT documentation indicates that channel A’s ID typically matches wedge 2 or 3,
    and channel B’s ID typically matches wedge 4.
    """
    if sensor_id_wedge < 1 or sensor_id_wedge > 6:
        raise ValueError("sensor_id_wedge must be in 1..6")

    block_levels = []

    # Blocks 1..8 (1/8 .. 8/8)
    for k in range(1, 9):
        block_levels.append(wedge_level(k))

    # Block 9 (zero)
    block_levels.append(wedge_level(0))

    # Blocks 10..15 (calibration values; each must map to a wedge level 1..8)
    cal_indices = [
        CAL_BLOCK_10, CAL_BLOCK_11, CAL_BLOCK_12,
        CAL_BLOCK_13, CAL_BLOCK_14, CAL_BLOCK_15,
    ]
    for idx in cal_indices:
        if idx < 1 or idx > 8:
            raise ValueError("CAL_BLOCK_* must be in 1..8")
        block_levels.append(wedge_level(idx))

    # Block 16 (sensor/channel identifier)
    block_levels.append(wedge_level(sensor_id_wedge))

    if len(block_levels) != 16:
        raise RuntimeError("Telemetry block_levels must be 16 entries")

    tlm = np.zeros((height, TLM_WIDTH), dtype=np.uint8)

    # Repeat the 128-line telemetry frame down the output.
    for line in range(height):
        within = line % TELEM_PERIOD            # 0..127
        block = within // TELEM_BLOCK_LINES     # 0..15
        tlm[line, :] = block_levels[block]

    return tlm


def prepare_image(path):
    """
    Prepare the source image for insertion into the APT IMAGE field.

    The APT IMAGE field is 909 samples wide per channel. We rescale the input PNG
    to IMG_WIDTH and convert it to 8-bit grayscale. The resulting raster height is
    the encoded “duration” of this synthetic transmission (APT is continuous; there
    is no fixed frame height in the format itself).
    """
    img = Image.open(path).convert("L")

    aspect = img.width / img.height
    new_width = IMG_WIDTH
    new_height = int(round(new_width / aspect))

    img = img.resize((new_width, new_height), Image.LANCZOS)

    arr = np.array(img, dtype=np.uint8)

    assert arr.shape[1] == IMG_WIDTH
    return arr

def pad_image_to_height(image: np.ndarray, height: int) -> np.ndarray:
    """
    Pad a (H, IMG_WIDTH) grayscale image to the requested height by adding black lines
    at the bottom (top-aligned).
    """
    if image.shape[0] == height:
        return image
    if image.shape[0] > height:
        raise ValueError("pad_image_to_height called with smaller target height")

    out = np.zeros((height, image.shape[1]), dtype=np.uint8)
    out[: image.shape[0], :] = image
    return out


def make_space_columns(height: int):
    """
    Build the SPACE/MARKER fields for channels A and B.

    The SPACE/MARKER fields provide minute markers and a consistent background
    reference. The channel polarities are opposite:

      - Channel A SPACE background is normally black (LOW)
      - Channel B SPACE background is normally white (HIGH)

    Every minute (120 lines at 2 lines/sec), the specification defines a four-line
    marker sequence “2 black lines then 2 white lines”. With opposite channel
    polarities, this sequence is arranged so each marker is visible where intended:
      - First 2 lines: “black” marker visible in channel B (force B to LOW)
      - Next 2 lines : “white” marker visible in channel A (force A to HIGH)
    """
    space_a = np.full((height, SPACE_WIDTH), LOW, dtype=np.uint8)   # Channel A normal background
    space_b = np.full((height, SPACE_WIDTH), HIGH, dtype=np.uint8)  # Channel B normal background

    for line in range(height):
        state = minute_marker_level(line)

        if state == "black":
            # Make the “black” marker visible in Channel B.
            space_b[line, :] = LOW

        elif state == "white":
            # Make the “white” marker visible in Channel A.
            space_a[line, :] = HIGH

        # "normal" leaves both channels at their background level.

    return space_a, space_b


def build_channel(
    image: np.ndarray,
    which: str,
    space_a: np.ndarray,
    space_b: np.ndarray,
) -> np.ndarray:
    """
    Assemble one channel (A or B) for all lines:
        SYNC | SPACE/MARKER | IMAGE | TELEMETRY

    The IMAGE field is duplicated into both channels by the caller; only the SYNC,
    SPACE/MARKER polarity, and telemetry channel-ID differ.
    """
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
    parser.add_argument("input_png_a")
    parser.add_argument("output_png")
    parser.add_argument(
        "--input-png-b",
        default=None,
        help="Optional second input image; if omitted, channel B image is black.",
    )
    args = parser.parse_args()

    image_a = prepare_image(args.input_png_a)

    if args.input_png_b is None:
        image_b = np.zeros((1, IMG_WIDTH), dtype=np.uint8)  # placeholder; will be padded
        image_b_height = 0
    else:
        image_b = prepare_image(args.input_png_b)
        image_b_height = image_b.shape[0]

    out_height = max(image_a.shape[0], image_b_height)

    image_a = pad_image_to_height(image_a, out_height)

    if args.input_png_b is None:
        image_b = np.zeros((out_height, IMG_WIDTH), dtype=np.uint8)
    else:
        image_b = pad_image_to_height(image_b, out_height)

    space_a, space_b = make_space_columns(out_height)

    chA = build_channel(image_a, "a", space_a=space_a, space_b=space_b)
    chB = build_channel(image_b, "b", space_a=space_a, space_b=space_b)

    apt = np.hstack([chA, chB])

    if apt.shape[1] != 2080:
        raise RuntimeError(f"APT line width must be 2080, got {apt.shape[1]}")

    Image.fromarray(apt).save(args.output_png)


if __name__ == "__main__":
    main()