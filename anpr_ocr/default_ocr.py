"""
Default OCR module.
"""

import os
from collections.abc import Sequence
from typing import Literal

import cv2
import numpy as np
import onnxruntime as ort
from fast_plate_ocr import LicensePlateRecognizer
from fast_plate_ocr.inference.hub import OcrModel

from anpr_ocr.base import BaseOCR, OcrResult
from anpr_ocr.utils import (
    INDIAN_STATE_CODES,
    disambiguate_plate,
    enhance_plate_image,
    heal_indian_plate,
    is_two_row_plate,
    split_two_row_crop,
    is_uae_region,
)

# pylint: disable=too-many-arguments
# ruff: noqa: PLR0913


class DefaultOCR(BaseOCR):
    """
    Default OCR class for license plate recognition using `fast-plate-ocr` models.
    """

    def __init__(
        self,
        hub_ocr_model: OcrModel | None = None,
        device: Literal["cuda", "cpu", "auto"] = "auto",
        providers: Sequence[str | tuple[str, dict]] | None = None,
        sess_options: ort.SessionOptions | None = None,
        model_path: str | os.PathLike | None = None,
        config_path: str | os.PathLike | None = None,
        force_download: bool = False,
        enhance_contrast: bool = False,
        min_plate_width: int = 0,
        syntax_pattern: str | Sequence[str] | None = None,
        region_hint: str | None = None,
    ) -> None:
        """
        Initialize the DefaultOCR with the specified parameters. Uses `fast-plate-ocr`'s
        `LicensePlateRecognizer`

        Parameters:
            hub_ocr_model: The name of the OCR model from the model hub.
            device: The device to run the model on. Options are "cuda", "cpu", or "auto". Defaults
             to "auto".
            providers: The execution providers to use in ONNX Runtime. If None, the default
             providers are used.
            sess_options: Custom session options for ONNX Runtime. If None, default session options
             are used.
            model_path: Path to a custom OCR model file. If None, the model is downloaded from the
             hub or cache.
            config_path: Path to a custom configuration file. If None, the default configuration is
             used.
            force_download: If True, forces the download of the model and overwrites any existing
             files.
            enhance_contrast: If True, applies CLAHE contrast enhancement before OCR inference.
            min_plate_width: Minimum width to upscale small crops to (0 to disable).
            syntax_pattern: Optional mask to disambiguate characters (e.g. 'LLDDLLDDDD').
            region_hint: Optional known region/country label for the footage's source (e.g. "UAE").
                The bundled OCR model's own region-classification head is unreliable for regions it
                wasn't trained on (it has no UAE class, for example, so it guesses lookalike
                countries), so a hint overrides both the displayed region and the UAE-specific
                character-healing decision instead of trusting that per-plate model guess.
        """
        self.ocr_model = LicensePlateRecognizer(
            hub_ocr_model=hub_ocr_model,
            device=device,
            providers=providers,
            sess_options=sess_options,
            onnx_model_path=model_path,
            plate_config_path=config_path,
            force_download=force_download,
        )
        self.enhance_contrast = enhance_contrast
        self.min_plate_width = min_plate_width
        self.syntax_pattern = syntax_pattern
        self.region_hint = region_hint
        self._region_hint_is_uae = region_hint is not None and is_uae_region(region_hint)

    def _run_single_crop(
        self, crop: np.ndarray
    ) -> tuple[str, list[float], str | None, float | None]:
        """Run OCR inference on a single cropped image and return text, probs, and region."""
        if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return "", [], None, None

        # Upscale low-resolution crops to ensure character strokes are distinct for the OCR network
        h, w = crop.shape[:2]
        if h < 24 or w < 80:
            scale = max(24.0 / max(1, h), 80.0 / max(1, w))
            new_w = max(80, round(w * scale))
            new_h = max(24, round(h * scale))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
            crop = cv2.addWeighted(crop, 1.25, blurred, -0.25, 0)

        if self.ocr_model.config.image_color_mode == "grayscale":
            crop_conv = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        elif self.ocr_model.config.image_color_mode == "rgb":
            crop_conv = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        else:
            crop_conv = crop

        prediction = self.ocr_model.run_one(crop_conv, return_confidence=True)
        char_probs = prediction.char_probs
        confidence: list[float] = (
            [] if char_probs is None else [float(x) for x in char_probs.tolist()]
        )
        text = prediction.plate or ""
        return text, confidence, prediction.region, prediction.region_prob

    def predict(self, cropped_plate: np.ndarray) -> OcrResult | None:
        """
        Perform OCR on a cropped license plate image. Supports both standard single-line
        plates and two-row (stacked) plates (common on auto-rickshaws, bikes, and trucks).

        Parameters:
            cropped_plate: The cropped image of the license plate in BGR format.

        Returns:
            OcrResult: An object containing the recognized text and per-character confidence.
        """
        if cropped_plate is None or cropped_plate.size == 0:
            return None

        # Preprocess plate image if configured
        if self.enhance_contrast or self.min_plate_width > 0:
            cropped_plate = enhance_plate_image(
                cropped_plate,
                enhance_contrast=self.enhance_contrast,
                min_width=self.min_plate_width,
            )

        h, w = cropped_plate.shape[:2]

        # 1. Baseline 1-row OCR prediction
        t1, c1, r1, rp1 = self._run_single_crop(cropped_plate)
        uae_plate = getattr(self, "_region_hint_is_uae", False) or is_uae_region(r1)
        if self.syntax_pattern and t1:
            t1 = disambiguate_plate(t1, self.syntax_pattern, apply_indian_healing=not uae_plate)

        h1 = t1 if uae_plate else heal_indian_plate(t1)
        mean_c1 = (sum(c1) / len(c1)) if c1 else 0.0

        # 2. Check for square / 2-row plate (e.g. auto-rickshaws, motorbikes)
        if is_two_row_plate(w, h):
            top_crop, bot_crop = split_two_row_crop(
                cropped_plate, top_fraction=0.52, bot_fraction=0.48
            )
            tt, ct, rt, rpt = self._run_single_crop(top_crop)
            tb, cb, _rb, _rpb = self._run_single_crop(bot_crop)

            if len(tt) >= 2 and len(tb) >= 2:
                t2 = tt + tb
                c2 = ct + cb
                uae_two_row_plate = getattr(self, "_region_hint_is_uae", False) or is_uae_region(rt)
                if self.syntax_pattern:
                    t2 = disambiguate_plate(
                        t2, self.syntax_pattern, apply_indian_healing=not uae_two_row_plate
                    )
                h2 = t2 if uae_two_row_plate else heal_indian_plate(t2)
                mean_c2 = (sum(c2) / len(c2)) if c2 else 0.0

                # Prefer 2-row if:
                # a) 2-row resolves to valid Indian state plate and 1-row does not
                # b) Both are valid Indian state plates, prefer longer / higher confidence
                # c) 1-row confidence is weak (<0.65) and 2-row has strong confidence (>=0.80)
                is_valid_indian_2row = len(h2) in (8, 9, 10) and h2[:2] in INDIAN_STATE_CODES
                is_valid_indian_1row = len(h1) in (8, 9, 10) and h1[:2] in INDIAN_STATE_CODES

                if is_valid_indian_2row and not is_valid_indian_1row:
                    return OcrResult(
                        text=h2,
                        confidence=c2,
                        region=getattr(self, "region_hint", None) or rt or r1,
                        region_confidence=rpt or rp1,
                    )
                elif is_valid_indian_2row and is_valid_indian_1row:
                    if len(h2) > len(h1) or (len(h2) == len(h1) and mean_c2 > mean_c1):
                        return OcrResult(
                            text=h2,
                            confidence=c2,
                            region=getattr(self, "region_hint", None) or rt or r1,
                            region_confidence=rpt or rp1,
                        )
                elif mean_c1 < 0.65 and mean_c2 >= 0.80 and len(t2) <= 10:
                    return OcrResult(
                        text=h2,
                        confidence=c2,
                        region=getattr(self, "region_hint", None) or rt or r1,
                        region_confidence=rpt or rp1,
                    )

        return OcrResult(
            text=h1,
            confidence=c1,
            region=getattr(self, "region_hint", None) or r1,
            region_confidence=rp1,
        )
