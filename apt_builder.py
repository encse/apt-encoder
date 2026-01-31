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

The output is an intermediate representation of the APT video levels suitable for
conversion into an audio or IQ waveform.

References:
- https://web.archive.org/web/20070316190349/http://www2.ncdc.noaa.gov/docs/klm/html/c4/sec4-2.htm#f423-3
- https://noaa-apt.mbernardi.com.ar/how-it-works.html

"""
import argparse
import numpy as np
from PIL import Image

FS_VIDEO = 4160

LOW = 11
HIGH = 244

SYNC_WIDTH = 39
SPACE_WIDTH = 47
IMG_WIDTH = 909
TELEM_WIDTH = 45

#1: Visible. Wavelength: 0.58µm - 0.68µm. Daytime cloud and surface mapping.
#2: Near-infrared. Wavelength: 0.725 - 1.00µm. Land-water boundaries.
#3A: Infrared. Wavelength: 1.58µm - 1.64µm. Snow and ice detection.
#3B: Infrared. Wavelength: 3.55µm - 3.93µm. Night cloud mapping, sea surface temperature.
#4: Infrared. Wavelength: 10.30µm - 11.30µm. Night cloud mapping, sea surface temperature.
#5: Infrared. Wavelength: 11.50µm - 12.50µm. Sea surface temperature.
CHANNEL_ID_A = "3A"
CHANNEL_ID_B = "4"

def make_sync(height: int, pattern) -> np.ndarray:
    line = np.full(SYNC_WIDTH, LOW, dtype=np.uint8)

    repeated_pattern = np.tile(pattern, 7)
    line[4:4 + repeated_pattern.size] = repeated_pattern

    return np.repeat(line[None, :], height, axis=0)

def make_telemetry(height: int, channel_id: int) -> np.ndarray:
    TELEM_BLOCK_LINES = 8
    TELEM_PERIOD = TELEM_BLOCK_LINES * 16
    BASE_WEDGES = [31, 63, 95, 127, 159, 191, 224, 255, 0]
    channel_wedge = {
        "1":  BASE_WEDGES[0],
        "2":  BASE_WEDGES[1],
        "3A": BASE_WEDGES[2],
        "3B": BASE_WEDGES[5],
        "4":  BASE_WEDGES[3],
        "5":  BASE_WEDGES[4],
    }[channel_id]

    wedges = np.array([
        *BASE_WEDGES,
        110,  # Black body radiator termometer #1.
        120,  # Black body radiator termometer #2.
        130,  # Black body radiator termometer #3.
        140,  # Black body radiator termometer #4.
        80,   # Patch temperature. 
        160,  # Temperature of the black body radiator observed by the imaging sensor
        channel_wedge,
    ], dtype=np.uint8)

    tlm = np.zeros((height, TELEM_WIDTH), dtype=np.uint8)
    for line in range(height):
        block = (line % TELEM_PERIOD) // TELEM_BLOCK_LINES
        tlm[line, :] = wedges[block]

    return tlm

def make_space(height: int, reference_value: int):
    space = np.zeros((height, SPACE_WIDTH), dtype=np.uint8) 

    for line in range(height):
        m = line % 120
        if m < 2:
            space[line, :] = LOW
        elif m < 4:
            space[line, :] = HIGH
        else:
            space[line, :] = reference_value

    return space

def build_channel(
    image: np.ndarray,
    channel: str,
) -> np.ndarray:
    h = image.shape[0]

    if channel == "a":
        sync = make_sync(h, [HIGH, HIGH, LOW, LOW])
        space = make_space(h, LOW)
        tlm = make_telemetry(h, CHANNEL_ID_A)
    elif channel == "b":
        sync = make_sync(h, [HIGH, HIGH, HIGH, LOW, LOW])
        space = make_space(h, HIGH)
        tlm = make_telemetry(h, CHANNEL_ID_B)
    else:
        raise ValueError("channel must be 'a' or 'b'")

    return np.hstack([sync, space, image, tlm])

def prepare_image(path):
    img = Image.open(path).convert("L")

    aspect = img.width / img.height
    new_width = IMG_WIDTH
    new_height = int(round(new_width / aspect))

    img = img.resize((new_width, new_height), Image.LANCZOS)

    arr = np.array(img, dtype=np.uint8)

    assert arr.shape[1] == IMG_WIDTH
    return arr

def pad_image_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    if image.shape[0] > height:
        raise ValueError("pad_image_to_height called with smaller target height")

    out = np.zeros((height, image.shape[1]), dtype=np.uint8)
    out[: image.shape[0], :] = image
    return out


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
        image_b = np.zeros((1, IMG_WIDTH), dtype=np.uint8)
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

    chA = build_channel(image_a, "a")
    chB = build_channel(image_b, "b")

    apt = np.hstack([chA, chB])

    if apt.shape[1] != 2080:
        raise RuntimeError(f"APT line width must be 2080, got {apt.shape[1]}")

    Image.fromarray(apt).save(args.output_png)


if __name__ == "__main__":
    main()