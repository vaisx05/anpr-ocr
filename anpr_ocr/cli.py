"""
Unified command-line interface for the anpr-ocr package.

Provides subcommands for image inference, video processing, live camera/RTSP
streaming, and model fine-tuning.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

# pylint: disable=too-many-branches, too-many-statements, import-outside-toplevel
# ruff: noqa: PLR0912, PLR0915, PLC0415, E501


def _create_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="anpr-ocr",
        description="High-performance Automatic License Plate Recognition (ALPR) system.",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="anpr-ocr 0.4.0",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        title="Subcommands",
        description="Select a subcommand to run",
    )

    # 1. Image subcommand
    image_parser = subparsers.add_parser(
        "image",
        help="Run ALPR on still image file(s) or a folder.",
        description="Analyze still images with plate detection and OCR.",
    )
    image_parser.add_argument(
        "images",
        nargs="*",
        default=[],
        help="Path(s) to image file(s) or directory.",
    )
    image_parser.add_argument(
        "-o",
        "--output",
        help="Output file path or directory for annotated image(s).",
    )
    image_parser.add_argument(
        "--detector",
        default="yolo-v9-t-416-license-plate-end2end",
        help="Detector model name (default: yolo-v9-t-416-license-plate-end2end).",
    )
    image_parser.add_argument(
        "--ocr",
        default="cct-xs-v2-global-model",
        help="OCR model name (default: cct-xs-v2-global-model).",
    )
    image_parser.add_argument(
        "--expected",
        help="Expected license plate string for verification.",
    )
    image_parser.add_argument(
        "--crop-margin",
        type=float,
        default=0.05,
        help="Fractional margin around detected bounding boxes (default: 0.05).",
    )
    image_parser.add_argument(
        "--enhance-contrast",
        action="store_true",
        help="Apply CLAHE contrast enhancement before OCR.",
    )
    image_parser.add_argument(
        "--syntax",
        help="Syntax mask pattern (e.g. LLDDLLDDDD).",
    )
    image_parser.add_argument(
        "--json",
        action="store_true",
        help="Output structured JSON results.",
    )

    # 2. Video subcommand
    video_parser = subparsers.add_parser(
        "video",
        help="Process recorded video file(s) with multi-frame tracking and logging.",
        description="Process video files, export annotated video, and generate peak-confidence vehicle logs.",
    )
    video_parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to video file or directory.",
    )
    video_parser.add_argument(
        "-o",
        "--output",
        help="Output annotated video path or directory.",
    )
    video_parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1).",
    )
    video_parser.add_argument(
        "--detector",
        default="yolo-v9-t-416-license-plate-end2end",
        help="Detector model name.",
    )
    video_parser.add_argument(
        "--ocr",
        default="cct-xs-v2-global-model",
        help="OCR model name.",
    )
    video_parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.35,
        help="Detector confidence threshold (default: 0.35).",
    )
    video_parser.add_argument(
        "--enhance-contrast",
        action="store_true",
        help="Apply CLAHE contrast enhancement before OCR (helps read small/low-contrast text).",
    )
    video_parser.add_argument(
        "--crop-margin-x",
        type=float,
        default=0.05,
        help=(
            "Fractional horizontal margin around detected bounding boxes (default: 0.05). "
            "Widen this for plates with a side region/emirate code panel (e.g. UAE plates)."
        ),
    )
    video_parser.add_argument(
        "--crop-margin-y",
        type=float,
        default=0.05,
        help="Fractional vertical margin around detected bounding boxes (default: 0.05).",
    )
    video_parser.add_argument(
        "--min-plate-width",
        type=int,
        default=0,
        help="Upscale plate crops narrower than this width (in px) before OCR (0 to disable).",
    )
    video_parser.add_argument(
        "--region-hint",
        help=(
            "Known region/country of the footage (e.g. 'UAE'). Overrides the OCR model's "
            "unreliable per-plate region guess, which has no class for some regions."
        ),
    )
    video_parser.add_argument(
        "--directml",
        action="store_true",
        default=False,
        help="Enable DirectML GPU acceleration.",
    )
    video_parser.add_argument(
        "--play",
        "--live",
        dest="play",
        action="store_true",
        help="Display live preview window during processing.",
    )
    video_parser.add_argument(
        "--csv",
        help="Custom CSV file path for peak vehicle logs.",
    )
    video_parser.add_argument(
        "--no-csv",
        action="store_true",
        default=False,
        help="Disable CSV plate logging.",
    )
    video_parser.add_argument(
        "--snapshots",
        action="store_true",
        default=False,
        help="Save crop snapshot images for each vehicle.",
    )

    # 3. Stream subcommand (webcam / RTSP)
    stream_parser = subparsers.add_parser(
        "stream",
        help="Live real-time feed from webcam index (0) or RTSP IP camera URL.",
        description="Run live ALPR on a webcam or RTSP/HTTP security camera stream with real-time overlay.",
    )
    stream_parser.add_argument(
        "source",
        nargs="?",
        default="0",
        help="Webcam device index (e.g. 0) or RTSP/HTTP URL (default: 0).",
    )
    stream_parser.add_argument(
        "-o",
        "--output",
        help="Optional recording output MP4 path.",
    )
    stream_parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame for performance (default: 1).",
    )
    stream_parser.add_argument(
        "--detector",
        default="yolo-v9-t-416-license-plate-end2end",
        help="Detector model name.",
    )
    stream_parser.add_argument(
        "--ocr",
        default="cct-xs-v2-global-model",
        help="OCR model name.",
    )
    stream_parser.add_argument(
        "--directml",
        action="store_true",
        default=False,
        help="Enable DirectML GPU acceleration.",
    )
    stream_parser.add_argument(
        "--csv",
        help="CSV log file path.",
    )

    # 4. Fine-tune subcommand
    ft_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tuning helper and dataset generation utilities.",
        description="Generate training commands and data augmentations for custom regional plates.",
    )
    ft_parser.add_argument(
        "--train-csv",
        help="Path to training CSV file.",
    )
    ft_parser.add_argument(
        "--val-csv",
        help="Path to validation CSV file.",
    )
    ft_parser.add_argument(
        "--config",
        default="model_config.yaml",
        help="Path to YAML model configuration file.",
    )
    ft_parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size (default: 64).",
    )
    ft_parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs (default: 100).",
    )
    ft_parser.add_argument(
        "--output-dir",
        default="./trained_ocr_model",
        help="Directory to save model checkpoints.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _create_main_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "image":
        from anpr_ocr.showcase import main as showcase_main

        # Forward arguments to showcase
        forward_args: list[str] = []
        if args.images:
            forward_args.extend([str(p) for p in args.images])
        if args.output:
            forward_args.extend(["--output", str(args.output)])
        if args.expected:
            forward_args.extend(["--expected", str(args.expected)])
        if args.detector:
            forward_args.extend(["--detector", str(args.detector)])
        if args.ocr:
            forward_args.extend(["--ocr", str(args.ocr)])
        if args.crop_margin is not None:
            forward_args.extend(["--crop-margin", str(args.crop_margin)])
        if args.enhance_contrast:
            forward_args.append("--enhance-contrast")
        if args.syntax:
            forward_args.extend(["--syntax", str(args.syntax)])
        if args.json:
            forward_args.append("--json")

        return showcase_main(forward_args)

    if args.command in ("video", "stream"):
        from anpr_ocr.video_demo import main as video_main

        forward_args = []
        if args.command == "video":
            if args.video:
                forward_args.append(str(args.video))
            if args.play:
                forward_args.append("--play")
        else:
            # stream command defaults to live window playback
            forward_args.append(str(args.source))
            forward_args.append("--play")

        if args.output:
            forward_args.extend(["--output", str(args.output)])
        if args.frame_skip:
            forward_args.extend(["--frame-skip", str(args.frame_skip)])
        if args.detector:
            forward_args.extend(["--detector", str(args.detector)])
        if args.ocr:
            forward_args.extend(["--ocr", str(args.ocr)])
        if getattr(args, "conf_thresh", None) is not None:
            forward_args.extend(["--conf-thresh", str(args.conf_thresh)])
        if args.command == "video":
            if getattr(args, "enhance_contrast", False):
                forward_args.append("--enhance-contrast")
            if getattr(args, "crop_margin_x", None) is not None:
                forward_args.extend(["--crop-margin-x", str(args.crop_margin_x)])
            if getattr(args, "crop_margin_y", None) is not None:
                forward_args.extend(["--crop-margin-y", str(args.crop_margin_y)])
            if getattr(args, "min_plate_width", None):
                forward_args.extend(["--min-plate-width", str(args.min_plate_width)])
            if getattr(args, "region_hint", None):
                forward_args.extend(["--region-hint", str(args.region_hint)])
        if args.directml:
            forward_args.append("--directml")
        if args.csv:
            forward_args.extend(["--csv", str(args.csv)])
        if getattr(args, "no_csv", False):
            forward_args.append("--no-csv")
        if getattr(args, "snapshots", False):
            forward_args.append("--snapshots")

        return video_main(forward_args)

    if args.command == "finetune":
        from anpr_ocr.finetune import generate_training_command

        if not args.train_csv or not args.val_csv:
            cmd_example = generate_training_command(
                "train.csv",
                "val.csv",
                args.config,
                batch_size=args.batch_size,
                epochs=args.epochs,
                output_dir=args.output_dir,
            )
            print("=" * 60)
            print("ANPR-OCR FINE-TUNING & DATASET GENERATOR")
            print("=" * 60)
            print("\nTo train a custom OCR model on your regional plates:")
            print("1. Prepare CSV files (train.csv and val.csv) with header: image_path,plate")
            print("2. Ensure fast-plate-ocr[train] is installed:")
            print("   pip install fast-plate-ocr[train] albumentations\n")
            print("3. Example Training Command:")
            print(f"   {cmd_example}\n")
            print("4. Export to ONNX after training completes:")
            print(
                f"   fast_plate_ocr export --model-path {args.output_dir}/best_model.keras "
                f"--output-path ./custom_plate_ocr.onnx\n"
            )
            return 0

        cmd = generate_training_command(
            args.train_csv,
            args.val_csv,
            args.config,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
        )
        print(f"\nGenerated Training Command:\n{cmd}\n")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
