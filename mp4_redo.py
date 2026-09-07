#!/usr/bin/env python3
"""Extract an MP4 into PNG frames, clear regions or keep detected outlines."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image, ImageChops, ImageDraw, ImageFilter


FRAME_PATTERN = "%08d.png"
FRAME_NAME_RE = re.compile(r"^\d+\.png$")
FOREGROUND_MARGIN = 10
FOREGROUND_COMPONENT_GAP = 20
RESIZE_SIZE = 240
SKIP_DEFAULT = 0


def run_command(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"required command not found: {error.filename}; install ffmpeg"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"command failed with exit code {error.returncode}") from error


def frame_files(frames_dir: Path) -> list[Path]:
    files = [
        path
        for path in frames_dir.iterdir()
        if path.is_file() and FRAME_NAME_RE.fullmatch(path.name)
    ]
    return sorted(files, key=lambda path: int(path.stem))


def extract_frames(input_path: Path, frames_dir: Path) -> list[Path]:
    frames_dir.mkdir(exist_ok=True)
    for path in frames_dir.iterdir():
        if path.is_file() and FRAME_NAME_RE.fullmatch(path.name):
            path.unlink()

    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(input_path),
            "-vsync",
            "0",
            str(frames_dir / FRAME_PATTERN),
        ]
    )
    files = frame_files(frames_dir)
    if not files:
        raise RuntimeError("ffmpeg produced no PNG frames")
    return files


def reset_region(image_path: Path, region: tuple[int, int, int, int]) -> None:
    x, y, width, height = region
    with Image.open(image_path) as source:
        image = source.convert("RGBA")
    image_width, image_height = image.size
    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"region {region} is outside {image_path.name} ({image_width}x{image_height})"
        )

    clear = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    image.paste(clear, (x, y))
    image.save(image_path, format="PNG")


def modify_frame_regions(
    image_path: Path, regions: list[tuple[int, int, int, int]]
) -> None:
    for region in regions:
        reset_region(image_path, region)


def modify_frames(
    files: list[Path], regions: list[tuple[int, int, int, int]], workers: int
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda path: modify_frame_regions(path, regions), files))


def process_foreground_frame(image_path: Path) -> Path:
    mask_path = image_path.with_name(f"{image_path.stem}_mask.png")
    and_path = image_path.with_name(f"{image_path.stem}_and.png")
    mask = foreground_mask(image_path)
    mask.save(mask_path, format="PNG")
    with Image.open(image_path) as source:
        image = source.convert("RGBA")
    transparent = Image.new("RGBA", image.size, (0, 0, 0, 0))
    image = Image.composite(image, transparent, mask)
    image.save(and_path, format="PNG")
    return and_path


def modify_with_foreground_masks(files: list[Path], workers: int) -> list[Path]:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(process_foreground_frame, files))


def resize_frame(source_path: Path, target_path: Path, resize_size: int) -> Path:
    with Image.open(source_path) as source:
        width, height = source.size
        scale = resize_size / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        resized = source.resize(size, Image.Resampling.LANCZOS)
        resized.save(target_path, format="PNG")
    return target_path


def prepare_merge_frames(
    files: list[Path],
    frames_dir: Path,
    resize_size: int,
    skip: int,
    count: int,
    workers: int,
) -> Path:
    with Image.open(files[0]) as first_frame:
        width, height = first_frame.size
    if resize_size >= min(width, height):
        raise ValueError(
            f"resize must be smaller than the input frame dimensions ({width}x{height})"
        )

    selected = files[:: skip + 1][:count]
    merge_dir = frames_dir / "_merge"
    merge_dir.mkdir(exist_ok=True)
    for path in merge_dir.glob("*.png"):
        path.unlink()
    targets = [
        merge_dir / f"{index:08d}.png" for index in range(1, len(selected) + 1)
    ]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(
            executor.map(
                resize_frame,
                selected,
                targets,
                [resize_size] * len(selected),
            )
        )
    return merge_dir


def foreground_mask(image_path: Path) -> Image.Image:
    with Image.open(image_path) as source:
        red, green, blue = source.convert("RGB").split()
    gray = Image.merge("RGB", (red, green, blue)).convert("L")
    brightest = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    darkest = ImageChops.darker(ImageChops.darker(red, green), blue)
    chroma = ImageChops.subtract(brightest, darkest)
    candidate = Image.new("L", gray.size, 0)
    for y in range(gray.height):
        for x in range(gray.width):
            if gray.getpixel((x, y)) >= 190 and chroma.getpixel((x, y)) <= 18:
                candidate.putpixel((x, y), 255)

    # Close small gaps around the outline before removing edge-connected background.
    candidate = candidate.filter(ImageFilter.MaxFilter(9)).filter(
        ImageFilter.MinFilter(9)
    )
    background = ImageChops.invert(candidate)
    for point in (
        (0, 0),
        (background.width - 1, 0),
        (0, background.height - 1),
        (background.width - 1, background.height - 1),
    ):
        ImageDraw.floodfill(background, point, 128, thresh=0)
    foreground = background.point(lambda value: 0 if value == 128 else 255)
    foreground_array = np.asarray(foreground, dtype=np.uint8) > 0
    labels, count = ndimage.label(foreground_array, structure=np.ones((3, 3)))
    if count:
        sizes = np.bincount(labels.ravel())
        main_label = int(np.argmax(sizes[1:]) + 1)
        main_component = labels == main_label
        distance = ndimage.distance_transform_edt(~main_component)
        foreground_array &= distance <= FOREGROUND_COMPONENT_GAP
        foreground = Image.fromarray(
            np.where(foreground_array, 255, 0).astype(np.uint8), mode="L"
        )
    kernel_size = FOREGROUND_MARGIN * 2 + 1
    expanded = foreground.filter(ImageFilter.MaxFilter(kernel_size))
    return expanded.point(lambda value: 1 if value else 0, mode="1")


def input_frame_rate(input_path: Path) -> str:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(input_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams", [])
        rate = streams[0].get("avg_frame_rate", "0/0") if streams else "0/0"
        if rate != "0/0":
            return rate
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    return "30"


def rebuild_video(
    input_pattern: str, output_mp4: Path, output_gif: Path, rate: str
) -> None:
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            rate,
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_mp4),
        ]
    )
    run_command(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            rate,
            "-i",
            input_pattern,
            "-vf",
            "split[s0][s1];[s0]palettegen=max_colors=256[p];[s1][p]paletteuse",
            "-loop",
            "0",
            str(output_gif),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract an MP4, clear regions in every frame, and rebuild MP4/GIF outputs."
    )
    parser.add_argument("input", type=Path, help="input video, for example xxx.mp4")
    parser.add_argument(
        "--region",
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        action="append",
        default=[],
        type=int,
        help="region to clear; without this option, automatically keep outlines",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=RESIZE_SIZE,
        metavar="PIXEL",
        help=f"maximum output frame dimension (default: {RESIZE_SIZE})",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=SKIP_DEFAULT,
        metavar="N",
        help=f"skip N frames between selected frames (default: {SKIP_DEFAULT})",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        metavar="MAX",
        help="maximum number of selected frames (default: all)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        print(f"input file does not exist: {input_path}", file=sys.stderr)
        return 2

    frames_dir = input_path.with_suffix("")
    try:
        if args.resize <= 0:
            raise ValueError("resize must be greater than zero")
        if args.skip < 0:
            raise ValueError("skip must be zero or greater")
        if args.count is not None and args.count <= 0:
            raise ValueError("count must be greater than zero")
        regions = []
        for values in args.region:
            region = tuple(values)
            x, y, width, height = region
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(
                    "region must be x y width height with x/y >= 0 and width/height > 0"
                )
            regions.append(region)
        files = extract_frames(input_path, frames_dir)
        workers = min(os.cpu_count() or 1, len(files))
        rate = input_frame_rate(input_path)
        if regions:
            output_mp4 = input_path.with_name(f"{input_path.stem}_redo.mp4")
            output_gif = input_path.with_name(f"{input_path.stem}_redo.gif")
            modify_frames(files, regions, workers)
            merge_dir = prepare_merge_frames(
                files,
                frames_dir,
                args.resize,
                args.skip,
                args.count or len(files),
                workers,
            )
            rebuild_video(str(merge_dir / FRAME_PATTERN), output_mp4, output_gif, rate)
        else:
            and_files = modify_with_foreground_masks(files, workers)
            output_mp4 = input_path.with_name(f"{input_path.stem}_and.mp4")
            output_gif = input_path.with_name(f"{input_path.stem}_and.gif")
            merge_dir = prepare_merge_frames(
                and_files,
                frames_dir,
                args.resize,
                args.skip,
                args.count or len(and_files),
                workers,
            )
            rebuild_video(str(merge_dir / FRAME_PATTERN), output_mp4, output_gif, rate)
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    merged_count = len(list(merge_dir.glob("*.png")))
    print(f"frames: {frames_dir} ({len(files)} PNG files)")
    print(f"merged frames: {merged_count} ({merge_dir})")
    print(f"created: {output_mp4}")
    print(f"created: {output_gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())