"""CLI demo: runs ANPR-OCR on a video file and writes an annotated output."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any, cast

from fast_plate_ocr.inference.hub import OcrModel
from open_image_models.detection.core.hub import PlateDetectorModel

from anpr_ocr import ALPR, PlateLogger
from anpr_ocr.alpr import SUPPORTED_VIDEO_EXTS
from anpr_ocr.logger import estimate_vehicle_color
from anpr_ocr.utils import get_plate_region

# pylint: disable=too-many-branches, too-many-statements, import-outside-toplevel
# ruff: noqa: PLR0912, PLR0915, PLC0415, E501, ARG001

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

# -- Paths -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEOS_DIR = PROJECT_ROOT / "data" / "videos"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "video_results"

# Common Indian & international plate syntax patterns for disambiguation
PLATE_PATTERNS = [
    "LLDDLLDDDD",  # 10-char: MH12DE1433, HR36AE7971
    "LLDDLDDDD",  # 9-char:  TN45Q3566
    "LLDDDDDD",  # 8-char:  LA020749
    "LLDDDDD",  # 7-char:  IN03044
    "LDDDDDD",  # 7-char:  UAE series + number
    "LDDDDD",  # 6-char:  UAE series + number
    "LLDDDD",  # 6-char:  UAE series + number
    "DDDDDD",  # 6-char:  UAE numeric plate
    "DDDDD",  # 5-char:  UAE numeric plate
    "DLLDDDD",  # 7-char:  5AU5341
    "LLDDLLL",  # 7-char:  LB02APF
    "LLLDDD",  # 6-char:  IZX842
]

# -- ANSI colors (enable on Windows) -----------------------------------------
os.system("")  # enable VT100 on Windows

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
RED = "\033[91m"
RESET = "\033[0m"


def _bar(ratio: float, width: int = 20) -> str:
    """Render a confidence gauge."""
    filled = round(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run ANPR-OCR on video file(s) and produce annotated output video(s).",
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=None,
        help="Path to a video file or directory (default: scans data/videos/).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file or directory (default: artifacts/video_results/<name>_anpr.mp4).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (1 = every frame, 3 = every 3rd). Default: 1.",
    )
    parser.add_argument(
        "--detector",
        default="yolo-v9-t-416-license-plate-end2end",
        help="Detector model name (default: yolo-v9-t-416-license-plate-end2end).",
    )
    parser.add_argument(
        "--ocr",
        default="cct-xs-v2-global-model",
        help="OCR model name (default: cct-xs-v2-global-model).",
    )
    parser.add_argument(
        "--codec",
        default=None,
        help="FourCC codec for the output video (e.g. mp4v, XVID). Auto-detected by default.",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.35,
        help="Detector confidence threshold (default: 0.35).",
    )
    parser.add_argument(
        "--syntax",
        nargs="*",
        default=None,
        help="Syntax mask pattern(s) for disambiguation (e.g. LLDDLLDDDD).",
    )
    parser.add_argument(
        "--enhance-contrast",
        action="store_true",
        help="Apply CLAHE contrast enhancement before OCR.",
    )
    parser.add_argument(
        "--crop-margin-x",
        type=float,
        default=0.05,
        help=(
            "Fractional horizontal margin around detected bounding boxes (default: 0.05). "
            "Widen this (e.g. 0.15-0.2) for plates with a side region/emirate code panel "
            "just outside the detector's box (e.g. UAE plates), so it isn't cropped out."
        ),
    )
    parser.add_argument(
        "--crop-margin-y",
        type=float,
        default=0.05,
        help="Fractional vertical margin around detected bounding boxes (default: 0.05).",
    )
    parser.add_argument(
        "--min-plate-width",
        type=int,
        default=0,
        help=(
            "Upscale plate crops narrower than this width (in px) before OCR (0 to disable). "
            "Helps the model read small text like a side region/emirate code panel."
        ),
    )
    parser.add_argument(
        "--region-hint",
        default=None,
        help=(
            "Known region/country of the footage (e.g. 'UAE'). Overrides the OCR model's own "
            "per-plate region guess, which has no class for some regions and guesses lookalike "
            "countries instead (e.g. Qatar/Norway for UAE plates)."
        ),
    )
    parser.add_argument(
        "--intra-threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op thread count (0 = auto/ORT default).",
    )
    parser.add_argument(
        "--inter-threads",
        type=int,
        default=0,
        help="ONNX Runtime inter-op thread count (0 = auto/ORT default).",
    )
    parser.add_argument(
        "--graph-opt",
        choices=["disable", "basic", "extended", "all"],
        default="all",
        help="ONNX Runtime graph optimization level (default: all).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=4,
        help="Minimum number of recognized characters to display a plate (default: 4).",
    )
    parser.add_argument(
        "--play",
        "--live",
        dest="play",
        action="store_true",
        help="Play video in a live real-time GUI window with detection overlays (Press Q to quit).",
    )
    parser.add_argument(
        "--directml",
        action="store_true",
        default=False,
        help="Enable DirectML GPU acceleration (Intel Iris Xe / AMD / NVIDIA).",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Custom CSV file path for logging peak license plate records (default: auto in output dir).",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        default=False,
        help="Disable automatic CSV plate logging.",
    )
    parser.add_argument(
        "--snapshots",
        action="store_true",
        default=False,
        help="Save high-resolution plate crop snapshot images for each finalized vehicle.",
    )
    parser.add_argument(
        "--min-log-conf",
        type=float,
        default=0.50,
        help="Minimum confidence threshold for logging a unique vehicle (default: 0.50).",
    )
    return parser


def _make_progress_callback(total_frames: int) -> Callable[[int, int], None]:
    """Return a callback that prints a live progress bar."""
    last_pct = [-1]  # mutable to capture in closure
    start = time.perf_counter()

    def _callback(current: int, total: int) -> None:
        pct = int(current / max(total, 1) * 100)
        if pct == last_pct[0]:
            return
        last_pct[0] = pct
        elapsed = time.perf_counter() - start
        fps = current / elapsed if elapsed > 0 else 0
        bar_w = 30
        filled = round(pct / 100 * bar_w)
        bar = "#" * filled + "-" * (bar_w - filled)
        eta = (total - current) / fps if fps > 0 else 0
        sys.stdout.write(
            f"\r  {CYAN}[{bar}]{RESET} {pct:3d}%  "
            f"{DIM}{current}/{total} frames  "
            f"{fps:.1f} fps  "
            f"ETA {eta:.0f}s{RESET}  "
        )
        sys.stdout.flush()

    return _callback


def _play_video_live(
    video_path: Path | str | int,
    alpr: ALPR,
    frame_skip: int = 2,
    min_chars: int = 4,
    logger: PlateLogger | None = None,
) -> None:
    """Play video or live stream in an interactive real-time GUI window with detection overlays."""
    import cv2
    import statistics

    if isinstance(video_path, int) or (isinstance(video_path, str) and video_path.isdigit()):
        cap = cv2.VideoCapture(int(video_path))
        src_name = f"Camera {video_path}"
    else:
        cap = cv2.VideoCapture(str(video_path))
        src_name = video_path.name if isinstance(video_path, Path) else str(video_path)

    if not cap.isOpened():
        print(f"{RED}Error: Failed to open video for playback: {video_path}{RESET}")
        return

    win_name = f"ANPR Real-Time - {src_name} (Press Q or ESC to exit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 720)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if src_fps <= 0.0:
        src_fps = 25.0
    frame_delay = max(1, int(1000 / src_fps))

    from anpr_ocr.utils import PlateTracker

    tracker = PlateTracker(max_unseen_frames=frame_skip * 5, window_size=5)
    active_plates: list[tuple[Any, str, float, str, str]] = []
    frame_idx = 0
    t_start = time.perf_counter()

    print(f"  {GREEN}[LIVE]{RESET} Playing {src_name} in GUI window...")
    print(f"  {DIM}Controls: Press 'Q' or 'ESC' to exit, [SPACE] to pause, [N] to step.{RESET}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        elapsed = time.perf_counter() - t_start
        fps_live = (frame_idx + 1) / elapsed if elapsed > 0 else 0.0

        # Deep inference on every Nth frame
        if frame_idx % frame_skip == 0:
            res = alpr.predict(frame)
            frame_dets = []
            model_regions: dict[str, str | None] = {}
            for r in res:
                if not r.ocr or not r.ocr.text or len(r.ocr.text.strip()) < min_chars:
                    continue
                conf = (
                    statistics.mean(r.ocr.confidence)
                    if isinstance(r.ocr.confidence, list)
                    else (r.ocr.confidence or 0.0)
                )
                if conf < 0.35:
                    continue
                text = r.ocr.text.strip()
                frame_dets.append((r.detection.bounding_box, text, conf))
                model_regions[text] = r.ocr.region

            tracked = tracker.update(frame_dets, frame_idx)
            active_plates = [
                (
                    box,
                    txt,
                    conf,
                    get_plate_region(txt, model_regions.get(txt)),
                    estimate_vehicle_color(frame, box),
                )
                for box, txt, conf, _ in tracked
            ]

            if logger is not None:
                for box, txt, conf, state, _ in active_plates:
                    logger.observe(
                        plate_text=txt,
                        confidence=conf,
                        bounding_box=box,
                        frame_idx=frame_idx,
                        frame_bgr=frame,
                        fps=src_fps,
                        model_region=state,
                    )

        # Render annotations
        display = frame.copy()
        for b, txt, c, state, color in active_plates:
            cv2.rectangle(display, (b.x1, b.y1), (b.x2, b.y2), (36, 255, 12), 2)
            lbl = f"{txt} {c * 100:.0f}% | {color} | {state}"
            cv2.putText(
                display,
                lbl,
                (b.x1, max(b.y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                display,
                lbl,
                (b.x1, max(b.y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # HUD Banner
        veh_count = len(logger.finalize()) if logger is not None else 0
        hud = f"Live FPS: {fps_live:.1f} | Frame: {frame_idx} | Vehicles: {veh_count} | [SPACE] Pause | [Q] Quit"
        cv2.putText(
            display, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            display, hud, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2, cv2.LINE_AA
        )

        cv2.imshow(win_name, display)
        key = cv2.waitKey(frame_delay) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord(" "), ord("p"), ord("P")):
            paused = True
            while paused:
                pause_display = display.copy()
                pause_hud = f"[PAUSED] Frame: {frame_idx} | Vehicles: {veh_count} | [SPACE] Resume | [N] Step Next | [Q] Quit"
                cv2.putText(
                    pause_display,
                    pause_hud,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 0, 0),
                    4,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    pause_display,
                    pause_hud,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow(win_name, pause_display)
                p_key = cv2.waitKey(30) & 0xFF
                if p_key in (ord("q"), ord("Q"), 27):
                    cap.release()
                    cv2.destroyAllWindows()
                    print(f"  {GREEN}[OK] Live playback closed.{RESET}")
                    if logger is not None:
                        print()
                        print(logger.summary_table())
                        if logger.output_csv:
                            csv_p = logger.export_csv()
                            if csv_p:
                                print(
                                    f"  {GREEN}[LOG]{RESET} Finalized peak plate log saved to: {csv_p}"
                                )
                        if logger.snapshots_dir:
                            snaps = logger.save_snapshots()
                            if snaps:
                                print(
                                    f"  {GREEN}[LOG]{RESET} Saved {len(snaps)} vehicle plate snapshot(s) to: {logger.snapshots_dir}"
                                )
                        print()
                    return
                elif p_key in (ord(" "), ord("p"), ord("P")):
                    paused = False
                    break
                elif p_key in (ord("n"), ord("N")):
                    # Step forward one frame
                    break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"  {GREEN}[OK] Live playback finished.{RESET}")

    if logger is not None:
        print()
        print(logger.summary_table())
        if logger.output_csv:
            csv_p = logger.export_csv()
            if csv_p:
                print(f"  {GREEN}[LOG]{RESET} Finalized peak plate log saved to: {csv_p}")
        if logger.snapshots_dir:
            snaps = logger.save_snapshots()
            if snaps:
                print(
                    f"  {GREEN}[LOG]{RESET} Saved {len(snaps)} vehicle plate snapshot(s) to: {logger.snapshots_dir}"
                )
        print()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # -- Resolve video files or live stream source ---------------------------
    video_input_raw = str(args.video) if args.video is not None else ""
    is_stream_input = video_input_raw.isdigit() or video_input_raw.startswith(
        ("rtsp://", "http://", "https://")
    )

    video_files: list[Path | str] = []
    if is_stream_input:
        video_files = [video_input_raw]
        target: Path | str = video_input_raw
    elif args.video is None:
        target = DEFAULT_VIDEOS_DIR
        if target.is_dir():
            video_files = sorted(
                p
                for p in target.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
            )
            if not video_files:
                print(f"{RED}Error: No supported video files found in: {target}{RESET}")
                print(
                    f"  Please place video files ({', '.join(sorted(SUPPORTED_VIDEO_EXTS))}) there,"
                )
                print("  or pass a video path: uv run video-demo path/to/video.mp4")
                return 1
    elif args.video is not None and args.video.is_dir():
        dir_target = args.video.resolve()
        target = dir_target
        video_files = sorted(
            p
            for p in dir_target.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_VIDEO_EXTS
        )
        if not video_files:
            print(f"{RED}Error: No supported video files found in: {target}{RESET}")
            return 1
    elif args.video is not None and args.video.is_file():
        file_target = args.video.resolve()
        target = file_target
        ext = file_target.suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTS:
            print(
                f"{RED}Error: Unsupported video format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTS))}{RESET}"
            )
            return 1
        video_files = [file_target]
    else:
        print(f"{RED}Error: Video path not found: {args.video}{RESET}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    syntax = args.syntax if args.syntax else PLATE_PATTERNS

    # -- Banner --------------------------------------------------------------
    print()
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}{CYAN}  ANPR-OCR Video Demo -- Automatic Number Plate Recognition{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print()
    print(f"  {DIM}Target:{RESET}      {target}")
    print(f"  {DIM}Videos found:{RESET}{len(video_files)} file(s)")
    for v in video_files:
        name_str = v.name if isinstance(v, Path) else v
        print(f"    - {name_str}")
    print(f"  {DIM}Detector:{RESET}   {args.detector}")
    print(f"  {DIM}OCR:{RESET}        {args.ocr}")
    print(f"  {DIM}Frame skip:{RESET} {args.frame_skip}")
    if args.codec:
        print(f"  {DIM}Codec:{RESET}      {args.codec}")
    print()

    # -- Configure Execution Providers (DirectML GPU or CPU) ---------------
    import onnxruntime as ort

    providers = None
    if args.directml:
        if "DmlExecutionProvider" in ort.get_available_providers():
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            print(
                f"  {GREEN}[GPU] DirectML hardware acceleration ENABLED (Intel Iris Xe / GPU){RESET}"
            )
        else:
            print(f"  {YELLOW}[WARN] DirectML provider not available, falling back to CPU{RESET}")

    # -- Configure ONNX Runtime session options (threading / graph opt) -----
    graph_opt_levels = {
        "disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }

    def _build_sess_options() -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = graph_opt_levels[args.graph_opt]
        if args.intra_threads > 0:
            opts.intra_op_num_threads = args.intra_threads
        if args.inter_threads > 0:
            opts.inter_op_num_threads = args.inter_threads
        return opts

    detector_sess_options = _build_sess_options()
    ocr_sess_options = _build_sess_options()

    # -- Load models ---------------------------------------------------------
    print(f"  {YELLOW}Loading models...{RESET}", end="", flush=True)
    t0 = time.perf_counter()
    alpr = ALPR(
        detector_model=cast(PlateDetectorModel, args.detector),
        ocr_model=cast(OcrModel, args.ocr),
        detector_conf_thresh=args.conf_thresh,
        enhance_contrast=args.enhance_contrast,
        crop_margin_x=args.crop_margin_x,
        crop_margin_y=args.crop_margin_y,
        min_plate_width=args.min_plate_width,
        region_hint=args.region_hint,
        detector_providers=providers,
        ocr_providers=providers,
        detector_sess_options=detector_sess_options,
        ocr_sess_options=ocr_sess_options,
        syntax_pattern=syntax,
    )
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"\r  {GREEN}[OK] Models loaded in {load_ms:.0f} ms{RESET}            ")
    print()

    # -- Process videos ------------------------------------------------------
    import cv2

    all_results = []
    total_pipeline_time = 0.0

    for idx, video_path in enumerate(video_files, 1):
        if isinstance(video_path, Path):
            v_stem = video_path.stem
            v_suffix = video_path.suffix
            v_source: int | str | Path = video_path
            display_name = video_path.name
        else:
            v_stem = f"camera_{video_path}" if video_path.isdigit() else "stream"
            v_suffix = ".mp4"
            v_source = int(video_path) if video_path.isdigit() else video_path
            display_name = f"Camera {video_path}" if str(video_path).isdigit() else str(video_path)

        # File Export / Output Paths
        if args.output and len(video_files) == 1 and args.output.suffix:
            output_path = args.output.resolve()
        elif args.output:
            args.output.mkdir(parents=True, exist_ok=True)
            output_path = args.output.resolve() / f"{v_stem}_anpr{v_suffix}"
        else:
            output_path = OUTPUT_DIR / f"{v_stem}_anpr{v_suffix}"

        # Resolve CSV log path and snapshots directory
        if not args.no_csv:
            if args.csv and len(video_files) == 1:
                csv_path = args.csv.resolve()
            else:
                csv_path = output_path.parent / f"{v_stem}_plates.csv"
        else:
            csv_path = None

        snapshots_dir = (output_path.parent / f"{v_stem}_snapshots") if args.snapshots else None

        logger = PlateLogger(
            output_csv=csv_path,
            snapshots_dir=snapshots_dir,
            min_conf=args.min_log_conf,
            min_chars=args.min_chars,
        )

        # Interactive Live Playback Mode
        if args.play:
            _play_video_live(
                video_path=v_source,
                alpr=alpr,
                frame_skip=args.frame_skip,
                min_chars=args.min_chars,
                logger=logger,
            )
            continue

        # Headless Processing Mode
        cap = cv2.VideoCapture(v_source if isinstance(v_source, int) else str(v_source))
        # Some containers (e.g. raw MJPEG streams) report a garbage/negative frame count.
        total_frames = max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"  {BOLD}[{idx}/{len(video_files)}] Processing: {display_name}{RESET}")
        print(
            f"      {DIM}Resolution:{RESET} {width}x{height} @ {src_fps:.1f} fps, "
            f"{total_frames if total_frames > 0 else 'unknown'} frames"
        )
        print(f"      {DIM}Saving to:{RESET}  {output_path.name}")

        progress = _make_progress_callback(total_frames)
        t_start = time.perf_counter()
        result = alpr.draw_predictions_video(
            source=v_source,
            output_path=output_path,
            frame_skip=args.frame_skip,
            codec=args.codec,
            min_chars=args.min_chars,
            progress_callback=progress,
            logger=logger,
        )
        elapsed = time.perf_counter() - t_start
        total_pipeline_time += elapsed

        # Clear progress line
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

        print(
            f"      {GREEN}[OK]{RESET} {result.processed_frames}/{result.total_frames} frames | "
            f"{result.total_plates_detected} plates detected | "
            f"{result.fps_processing:.1f} fps ({result.processing_time_seconds:.1f}s)"
        )
        print()

        # Display Finalized Peak Vehicle Log
        print(logger.summary_table())
        if csv_path:
            saved_csv = logger.export_csv()
            if saved_csv:
                print(
                    f"      {GREEN}[LOG]{RESET} Finalized peak plate log saved to: {saved_csv.name}"
                )
        if snapshots_dir:
            snaps = logger.save_snapshots()
            if snaps:
                print(
                    f"      {GREEN}[LOG]{RESET} Saved {len(snaps)} peak plate snapshot(s) to: {snapshots_dir.name}/"
                )
        print()
        all_results.append(result)

    # -- Summary -------------------------------------------------------------
    total_frames_all = sum(r.total_frames for r in all_results)
    proc_frames_all = sum(r.processed_frames for r in all_results)
    plates_all = sum(r.total_plates_detected for r in all_results)
    unique_vehicles_all = sum(
        len(r.vehicle_records) for r in all_results if r.vehicle_records is not None
    )
    overall_fps = proc_frames_all / total_pipeline_time if total_pipeline_time > 0 else 0

    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"{BOLD}{CYAN}  Overall Summary{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print(f"  {BOLD}Videos processed:{RESET}       {len(all_results)}")
    print(f"  {BOLD}Total frames:{RESET}           {total_frames_all}")
    print(f"  {BOLD}Processed frames:{RESET}       {proc_frames_all}")
    print(f"  {BOLD}Total plates detected:{RESET}  {plates_all}")
    print(
        f"  {BOLD}Unique vehicles logged:{RESET} {unique_vehicles_all} (peak confidence deduplicated)"
    )
    print(f"  {BOLD}Total processing time:{RESET}  {total_pipeline_time:.2f}s")
    print(f"  {BOLD}Effective throughput:{RESET}   {overall_fps:.1f} fps")
    print()
    out_dir_display = args.output if (args.output and args.output.is_dir()) else OUTPUT_DIR
    print(f"  {GREEN}[OK] Annotated videos saved to:{RESET} {out_dir_display}")
    print(f"{BOLD}{CYAN}{'=' * 72}{RESET}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
