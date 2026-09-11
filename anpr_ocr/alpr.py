"""
ALPR module.
"""

import os
import statistics
import time
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
from fast_plate_ocr.inference.hub import OcrModel
from open_image_models.detection.core.hub import PlateDetectorModel

from anpr_ocr.base import BaseDetector, BaseOCR, DetectionResult, OcrResult
from anpr_ocr.default_detector import DefaultDetector
from anpr_ocr.default_ocr import DefaultOCR
from anpr_ocr.logger import PlateLogger, VehicleRecord
from anpr_ocr.utils import (
    PlateTracker,
    disambiguate_plate,
    is_uae_region,
    pad_bounding_box,
)


# pylint: disable=too-many-arguments, too-many-locals
# ruff: noqa: PLR0913, PLR0912, PLR0915

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".flv", ".wmv", ".m4v"}
"""Video file extensions supported for inference."""

# Default fourcc codecs by output extension
_CODEC_MAP: dict[str, str] = {
    ".mp4": "mp4v",
    ".avi": "XVID",
    ".mkv": "mp4v",
    ".mov": "mp4v",
    ".webm": "VP80",
    ".wmv": "WMV2",
}


@dataclass(frozen=True)
class ALPRResult:
    """
    Detection and OCR output for one license plate.

    Attributes:
        detection: Detector output for the plate.
        ocr: OCR output for the plate, or None if OCR does not return a result.
    """

    detection: DetectionResult
    ocr: OcrResult | None


@dataclass(frozen=True, slots=True)
class DrawPredictionsResult:
    """
    Return value from draw_predictions.

    Attributes:
        image: The input image with boxes and text drawn on it.
        results: The ALPR results used to draw the annotations.
    """

    image: np.ndarray
    results: list[ALPRResult]


@dataclass(frozen=True, slots=True)
class VideoResult:
    """
    Summary statistics returned after processing a video.

    Attributes:
        output_path: Path to the annotated output video file.
        total_frames: Total number of frames in the source video.
        processed_frames: Number of frames that were run through the ALPR pipeline.
        total_plates_detected: Cumulative count of plates detected across all processed frames.
        processing_time_seconds: Wall-clock time spent processing the video.
        fps_processing: Effective processing throughput in frames per second.
    """

    output_path: str
    total_frames: int
    processed_frames: int
    total_plates_detected: int
    processing_time_seconds: float
    fps_processing: float
    vehicle_records: list[VehicleRecord] | None = None


def _draw_plate_annotations(
    frame: np.ndarray,
    results: Sequence[ALPRResult],
    show_region: bool = False,
) -> np.ndarray:
    """
    Render bounding boxes and OCR text overlays on an image frame.
    """
    img = frame.copy()
    height, width = img.shape[:2]
    font_scale = min(1.25, max(0.4, width / 1000))
    text_thickness = 1 if font_scale < 0.75 else 2
    outline_thickness = text_thickness + max(3, round(font_scale * 3))

    for result in results:
        ocr_result = result.ocr
        if not ocr_result or not ocr_result.text:
            continue
        bbox = result.detection.bounding_box
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2

        cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)

        conf: float = (
            statistics.mean(ocr_result.confidence)
            if isinstance(ocr_result.confidence, list)
            else (ocr_result.confidence or 0.0)
        )
        display_lines = [f"{ocr_result.text} {conf * 100:.0f}%"]
        if show_region and ocr_result.region:
            reg_text = ocr_result.region
            if ocr_result.region_confidence is not None:
                reg_text = f"{reg_text} {ocr_result.region_confidence * 100:.0f}%"
            display_lines.insert(0, reg_text)

        _, text_height = cv2.getTextSize(
            display_lines[0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )[0]
        line_gap = max(14, round(text_height * 0.6))
        line_height = text_height + line_gap
        text_y = y1 - 10 - ((len(display_lines) - 1) * line_height)
        if text_y - text_height < 0:
            text_y = y2 + text_height + 10

        for idx, line in enumerate(display_lines):
            text_width, current_text_height = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
            )[0]
            text_x = min(max(x1, 5), max(5, width - text_width - 5))
            current_y = min(
                max(text_y + (idx * line_height), current_text_height + 5),
                height - 5,
            )
            cv2.putText(
                img=img,
                text=line,
                org=(text_x, current_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(0, 0, 0),
                thickness=outline_thickness,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                img=img,
                text=line,
                org=(text_x, current_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale,
                color=(255, 255, 255),
                thickness=text_thickness,
                lineType=cv2.LINE_AA,
            )

    return img


class ALPR:
    """
    Automatic License Plate Recognition (ALPR) system class.

    This class combines a detector and an OCR model to recognize license plates in images.
    """

    def __init__(
        self,
        detector: BaseDetector | None = None,
        ocr: BaseOCR | None = None,
        detector_model: PlateDetectorModel = "yolo-v9-t-416-license-plate-end2end",
        detector_conf_thresh: float = 0.4,
        detector_providers: Sequence[str | tuple[str, dict]] | None = None,
        detector_sess_options: ort.SessionOptions | None = None,
        ocr_model: OcrModel | None = "cct-xs-v2-global-model",
        ocr_device: Literal["cuda", "cpu", "auto"] = "auto",
        ocr_providers: Sequence[str | tuple[str, dict]] | None = None,
        ocr_sess_options: ort.SessionOptions | None = None,
        ocr_model_path: str | os.PathLike | None = None,
        ocr_config_path: str | os.PathLike | None = None,
        ocr_force_download: bool = False,
        crop_margin: float = 0.05,
        crop_margin_x: float | None = None,
        crop_margin_y: float | None = None,
        enhance_contrast: bool = False,
        min_plate_width: int = 0,
        syntax_pattern: str | Sequence[str] | None = None,
    ) -> None:
        """
        Initialize the ALPR system.

        Parameters:
            detector: An instance of BaseDetector. If None, the DefaultDetector is used.
            ocr: An instance of BaseOCR. If None, the DefaultOCR is used.
            detector_model: The name of the detector model or a PlateDetectorModel enum instance.
                Defaults to "yolo-v9-t-416-license-plate-end2end".
            detector_conf_thresh: Confidence threshold for the detector.
            detector_providers: Execution providers for the detector.
            detector_sess_options: Session options for the detector.
            ocr_model: The name of the OCR model from the model hub.
                Defaults to "cct-xs-v2-global-model". This can be None if `ocr_model_path` and
                `ocr_config_path` parameters are passed.
            ocr_device: The device to run the OCR model on ("cuda", "cpu", or "auto").
            ocr_providers: Execution providers for the OCR. If None, the default providers are used.
            ocr_sess_options: Session options for the OCR. If None, default session options are
                used.
            ocr_model_path: Custom model path for the OCR. If None, the model is downloaded from the
                hub or cache.
            ocr_config_path: Custom config path for the OCR. If None, the default configuration is
                used.
            ocr_force_download: Whether to force download the OCR model.
            crop_margin: Fractional margin padding around detected bounding boxes (default: 0.05).
                Helps prevent edge characters from being truncated.
            crop_margin_x: Optional horizontal override for crop_margin. Widen this for plate
                layouts where a region/emirate code sits in a side panel just outside the
                detector's box (e.g. UAE plates), so it isn't cropped out before OCR.
            crop_margin_y: Optional vertical override for crop_margin.
            enhance_contrast: Whether to apply CLAHE contrast enhancement before OCR inference.
            min_plate_width: Minimum width to upscale small crops to (0 to disable).
            syntax_pattern: Optional mask to disambiguate characters (e.g. 'LLDDLLDDDD').
        """
        self.crop_margin = crop_margin
        self.crop_margin_x = crop_margin_x if crop_margin_x is not None else crop_margin
        self.crop_margin_y = crop_margin_y if crop_margin_y is not None else crop_margin
        self.enhance_contrast = enhance_contrast
        self.min_plate_width = min_plate_width
        self.syntax_pattern = syntax_pattern

        # Initialize the detector
        self.detector = detector or DefaultDetector(
            model_name=detector_model,
            conf_thresh=detector_conf_thresh,
            providers=detector_providers,
            sess_options=detector_sess_options,
        )

        # Initialize the OCR
        self.ocr = ocr or DefaultOCR(
            hub_ocr_model=ocr_model,
            device=ocr_device,
            providers=ocr_providers,
            sess_options=ocr_sess_options,
            model_path=ocr_model_path,
            config_path=ocr_config_path,
            force_download=ocr_force_download,
            enhance_contrast=enhance_contrast,
            min_plate_width=min_plate_width,
            syntax_pattern=syntax_pattern,
        )

    def predict(self, frame: np.ndarray | str | os.PathLike) -> list[ALPRResult]:
        """
        Run plate detection and OCR on an image.

        Parameters:
            frame: Unprocessed frame (Colors in order: BGR), image path, or PathLike.

        Returns:
            A list of ALPRResult objects, one for each detected plate.
        """
        if isinstance(frame, (str, os.PathLike)):
            img_path = str(frame)
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image from path: {img_path}")
        elif isinstance(frame, np.ndarray):
            if (
                frame.size == 0
                or len(frame.shape) < 2
                or frame.shape[0] == 0
                or frame.shape[1] == 0
            ):
                return []
            img = frame
        else:
            raise TypeError(f"Expected np.ndarray or path-like object, got {type(frame).__name__}")

        plate_detections = self.detector.predict(img)
        alpr_results: list[ALPRResult] = []
        for detection in plate_detections:
            bbox = detection.bounding_box
            if self.crop_margin_x > 0 or self.crop_margin_y > 0:
                x1, y1, x2, y2 = pad_bounding_box(
                    bbox.x1,
                    bbox.y1,
                    bbox.x2,
                    bbox.y2,
                    img.shape[1],
                    img.shape[0],
                    margin_x=self.crop_margin_x,
                    margin_y=self.crop_margin_y,
                )
            else:
                x1, y1 = max(bbox.x1, 0), max(bbox.y1, 0)
                x2, y2 = min(bbox.x2, img.shape[1]), min(bbox.y2, img.shape[0])

            if x2 <= x1 or y2 <= y1:
                alpr_results.append(ALPRResult(detection=detection, ocr=None))
                continue

            cropped_plate = img[y1:y2, x1:x2]
            ocr_result = self.ocr.predict(cropped_plate)

            # Apply syntax disambiguation and auto-healing
            if ocr_result and ocr_result.text:
                disambiguated_text = disambiguate_plate(
                    ocr_result.text,
                    self.syntax_pattern,
                    apply_indian_healing=not is_uae_region(ocr_result.region),
                )
                if disambiguated_text != ocr_result.text:
                    ocr_result = OcrResult(
                        text=disambiguated_text,
                        confidence=ocr_result.confidence,
                        region=ocr_result.region,
                        region_confidence=ocr_result.region_confidence,
                    )

            alpr_result = ALPRResult(detection=detection, ocr=ocr_result)
            alpr_results.append(alpr_result)
        return alpr_results

    def draw_predictions(
        self,
        frame: np.ndarray | str | os.PathLike,
        show_region: bool = False,
        min_chars: int = 3,
        min_conf: float = 0.35,
    ) -> DrawPredictionsResult:
        """
        Draw detections and OCR results on an image.

        Parameters:
            frame: The original frame, image path, or PathLike.
            show_region: Whether to display country/region prediction above the plate.
                Defaults to False (disabled).
            min_chars: Minimum recognized character count to display annotation.
            min_conf: Minimum average OCR confidence to display annotation.

        Returns:
            A DrawPredictionsResult with the annotated image and the ALPR results.
        """
        if isinstance(frame, (str, os.PathLike)):
            img_path = str(frame)
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Failed to load image from path: {img_path}")
        elif isinstance(frame, np.ndarray):
            img = frame.copy()
        else:
            raise TypeError(f"Expected np.ndarray or path-like object, got {type(frame).__name__}")

        # Get ALPR results
        alpr_results = self.predict(img)
        drawn_results = []

        for result in alpr_results:
            ocr_result = result.ocr
            if ocr_result is None or not ocr_result.text:
                continue
            clean_text = ocr_result.text.strip()
            if len(clean_text) < min_chars:
                continue

            confidence: float = (
                statistics.mean(ocr_result.confidence)
                if isinstance(ocr_result.confidence, list)
                else (ocr_result.confidence or 0.0)
            )
            if confidence < min_conf:
                continue

            drawn_results.append(result)

        annotated = _draw_plate_annotations(img, drawn_results, show_region=show_region)
        return DrawPredictionsResult(image=annotated, results=drawn_results)

    def predict_batch(
        self, frames: Sequence[np.ndarray | str | os.PathLike]
    ) -> list[list[ALPRResult]]:
        """
        Run plate detection and OCR on a sequence of images.

        Parameters:
            frames: Sequence of image numpy arrays or file paths.

        Returns:
            A list where each item is the list of ALPRResults for that image.
        """
        return [self.predict(str(f) if isinstance(f, os.PathLike) else f) for f in frames]

    def draw_predictions_batch(
        self, frames: Sequence[np.ndarray | str | os.PathLike]
    ) -> list[DrawPredictionsResult]:
        """
        Draw predictions on a sequence of images.

        Parameters:
            frames: Sequence of image numpy arrays or file paths.

        Returns:
            A list of DrawPredictionsResult for each image.
        """
        return [self.draw_predictions(str(f) if isinstance(f, os.PathLike) else f) for f in frames]

    # -- Video methods -------------------------------------------------------

    @staticmethod
    def _open_video(source: int | str | os.PathLike) -> cv2.VideoCapture:
        """Open a video file, webcam device index, or network stream URL and validate it."""
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            cam_idx = int(source)
            cap = cv2.VideoCapture(cam_idx)
            if not cap.isOpened():
                raise ValueError(f"Failed to open camera device index: {cam_idx}")
            return cap

        path_str = str(source)
        if path_str.startswith(("rtsp://", "http://", "https://")):
            cap = cv2.VideoCapture(path_str)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video stream: {path_str}")
            return cap

        ext = Path(path_str).suffix.lower()
        if ext not in SUPPORTED_VIDEO_EXTS:
            raise ValueError(
                f"Unsupported video format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_VIDEO_EXTS))}"
            )
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {path_str}")
        return cap

    def predict_video(
        self,
        source: int | str | os.PathLike,
        frame_skip: int = 1,
    ) -> Generator[tuple[int, list[ALPRResult]], None, None]:
        """
        Run plate detection and OCR on every *frame_skip*-th frame of a video or live stream.

        This is a **generator** — it yields results lazily so arbitrarily long
        videos can be processed without holding all frames in memory.

        Parameters:
            source: Path to a video file, camera device index (e.g. 0), or stream URL.
            frame_skip: Process every Nth frame (1 = every frame, 2 = every
                other frame, etc.). Must be >= 1.

        Yields:
            Tuples of ``(frame_index, results)`` where *frame_index* is the
            0-based index of the frame in the video and *results* is the list of
            ALPRResult objects for that frame.
        """
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")

        cap = self._open_video(source)
        try:
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_skip == 0:
                    results = self.predict(frame)
                    yield frame_idx, results
                frame_idx += 1
        finally:
            cap.release()

    def draw_predictions_video(
        self,
        source: int | str | os.PathLike,
        output_path: str | os.PathLike | None = None,
        frame_skip: int = 1,
        codec: str | None = None,
        show_region: bool = False,
        min_chars: int = 3,
        min_conf: float = 0.35,
        progress_callback: Callable[[int, int], None] | None = None,
        logger: PlateLogger | None = None,
    ) -> VideoResult:
        """
        Read a video, draw ALPR annotations on each processed frame, and write
        the result to an output video file.

        Frames that are *not* processed (due to ``frame_skip``) carry forward
        active track annotations so playback remains flicker-free at original FPS.

        Parameters:
            source: Path to input video file or camera/stream.
            output_path: Where to write annotated video. If ``None``, saved with ``_anpr`` suffix.
            frame_skip: Run ALPR on every Nth frame (1 = every frame).
            codec: FourCC codec string (e.g. ``'mp4v'``). Auto-detected if ``None``.
            show_region: Whether to display country/region prediction above the plate.
            min_chars: Minimum recognized character count to display annotation.
            min_conf: Minimum average OCR confidence to display annotation.
            progress_callback: Callback receiving ``(current_frame: int, total_frames: int)``.
            logger: Optional PlateLogger instance for deduplication and logging.

        Returns:
            A :class:`VideoResult` with processing statistics.
        """
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")

        source_str = str(source)
        is_stream = (
            isinstance(source, int)
            or source_str.isdigit()
            or source_str.startswith(("rtsp://", "http://", "https://"))
        )
        cap = self._open_video(source)

        try:
            raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_frames = max(0, raw_total) if not is_stream else 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if fps <= 0.0:
                fps = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Resolve output path
            if output_path is None:
                if is_stream:
                    out = Path(f"stream_recording_{int(time.time())}_anpr.mp4")
                else:
                    src_p = Path(source_str)
                    out = src_p.with_stem(src_p.stem + "_anpr")
            else:
                out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            # Resolve codec
            fourcc_str = codec or _CODEC_MAP.get(out.suffix.lower(), "mp4v")
            fourcc = cv2.VideoWriter.fourcc(*fourcc_str)
            writer = cv2.VideoWriter(str(out), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(
                    f"Failed to create video writer for {out} "
                    f"(codec={fourcc_str}, {width}x{height} @ {fps:.1f}fps)"
                )

            processed_frames = 0
            total_plates = 0
            t0 = time.perf_counter()

            # Multi-frame temporal voting tracker with IoU matching and rolling consensus
            tracker = PlateTracker(max_unseen_frames=frame_skip * 5, window_size=5)
            last_smoothed_results: list[ALPRResult] = []

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    alpr_results = self.predict(frame)
                    valid_dets = []
                    det_map = {}

                    for res in alpr_results:
                        if not res.ocr or not res.ocr.text:
                            continue
                        clean_t = res.ocr.text.strip()
                        if len(clean_t) < min_chars:
                            continue
                        c_val: float = (
                            statistics.mean(res.ocr.confidence)
                            if isinstance(res.ocr.confidence, list)
                            else (res.ocr.confidence or 0.0)
                        )
                        if c_val < min_conf:
                            continue
                        b = res.detection.bounding_box
                        valid_dets.append((b, clean_t, c_val))
                        det_map[id(b)] = res

                    tracked = tracker.update(valid_dets, frame_idx)
                    smoothed_results: list[ALPRResult] = []

                    for box, display_text, display_conf, _ in tracked:
                        orig_res = det_map.get(id(box))
                        if logger is not None:
                            logger.observe(
                                plate_text=display_text,
                                confidence=display_conf,
                                bounding_box=box,
                                frame_idx=frame_idx,
                                frame_bgr=frame,
                                fps=fps,
                                model_region=(
                                    orig_res.ocr.region if orig_res and orig_res.ocr else None
                                ),
                            )
                        reg = orig_res.ocr.region if orig_res and orig_res.ocr else None
                        reg_c = (
                            orig_res.ocr.region_confidence if orig_res and orig_res.ocr else None
                        )
                        smoothed_ocr = OcrResult(
                            text=display_text,
                            confidence=display_conf,
                            region=reg,
                            region_confidence=reg_c,
                        )
                        orig_det = (
                            orig_res.detection
                            if orig_res
                            else DetectionResult(
                                label="license-plate", confidence=1.0, bounding_box=box
                            )
                        )
                        smoothed_results.append(ALPRResult(detection=orig_det, ocr=smoothed_ocr))

                    last_smoothed_results = smoothed_results
                    annotated_frame = _draw_plate_annotations(
                        frame, smoothed_results, show_region=show_region
                    )
                    writer.write(annotated_frame)
                    total_plates += len(smoothed_results)
                    processed_frames += 1
                elif last_smoothed_results:
                    # Carry forward active track overlays across skipped frames to avoid flicker
                    inter_annotated = _draw_plate_annotations(
                        frame, last_smoothed_results, show_region=show_region
                    )
                    writer.write(inter_annotated)
                else:
                    writer.write(frame)

                if progress_callback is not None and total_frames > 0:
                    progress_callback(frame_idx + 1, total_frames)

                frame_idx += 1

            elapsed = time.perf_counter() - t0
            writer.release()

            return VideoResult(
                output_path=str(out),
                total_frames=frame_idx,
                processed_frames=processed_frames,
                total_plates_detected=total_plates,
                processing_time_seconds=round(elapsed, 3),
                fps_processing=round(processed_frames / elapsed, 2) if elapsed > 0 else 0.0,
                vehicle_records=logger.finalize() if logger is not None else None,
            )
        finally:
            cap.release()
