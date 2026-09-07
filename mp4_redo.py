#!/usr/bin/env python3
"""Extract an MP4 into PNG frames, clear selected regions, and rebuild it."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image


FRAME_PATTERN = "%08d.png"
FRAME_NAME_RE = re.compile(r"^\d+\.png$")


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


def modify_frames(
    files: list[Path], regions: list[tuple[int, int, int, int]]
) -> None:
    for image_path in files:
        for region in regions:
            reset_region(image_path, region)


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


def rebuild_video(frames_dir: Path, output_mp4: Path, output_gif: Path, rate: str) -> None:
    input_pattern = str(frames_dir / FRAME_PATTERN)
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
        help="region to clear; may be specified multiple times",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        print(f"input file does not exist: {input_path}", file=sys.stderr)
        return 2

    frames_dir = input_path.with_suffix("")
    output_mp4 = input_path.with_name(f"{input_path.stem}_redo.mp4")
    output_gif = input_path.with_name(f"{input_path.stem}_redo.gif")
    try:
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
        modify_frames(files, regions)
        rebuild_video(
            frames_dir,
            output_mp4,
            output_gif,
            input_frame_rate(input_path),
        )
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(f"frames: {frames_dir} ({len(files)} PNG files)")
    print(f"created: {output_mp4}")
    print(f"created: {output_gif}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())