import cv2
import numpy as np
from pathlib import Path
import logging

from src.application.ports.image_preprocessor import (
    ImagePreprocessorPort,
    PreprocessingResult
)

logger = logging.getLogger(__name__)


class OpenCVPreprocessor(ImagePreprocessorPort):
    """
    Preprocesador de imagenes para OCR.

    Estrategia suave (no destructiva):
    1. Upscaling si la imagen es pequena (mejora drastica en EasyOCR).
    2. Escala de grises.
    3. CLAHE (contraste adaptativo) en lugar de binarizacion dura.

    La binarizacion adaptativa queda como plan B (metodo binarize),
    solo se usa si la confianza del OCR es baja.
    """

    MIN_WIDTH = 1500
    MAX_SCALE = 3

    def process(self, image_path: str) -> PreprocessingResult:
        applied_ops = []

        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"No se pudo leer la imagen: {image_path}")

            original_size = img.shape[:2]
            work = img

            # 1. Upscaling si la imagen es pequena
            h, w = work.shape[:2]
            if w < self.MIN_WIDTH:
                scale = min(self.MAX_SCALE, self.MIN_WIDTH / w)
                work = cv2.resize(
                    work, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
                applied_ops.append(f"upscale_x{scale:.1f}")

            # 2. Escala de grises
            gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
            applied_ops.append("grayscale")

            # 3. CLAHE (mejora contraste sin destruir glifos)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            applied_ops.append("clahe")

            logger.info(f"Preprocesamiento completado: {applied_ops}")

            return PreprocessingResult(
                output_path=image_path,
                applied_operations=applied_ops,
                original_size=original_size,
                processed_size=enhanced.shape[:2],
                image_data=enhanced,
            )

        except Exception as e:
            logger.error(f"Error en preprocesamiento: {e}")
            return PreprocessingResult(
                output_path=image_path,
                applied_operations=["none"],
                original_size=(0, 0),
                processed_size=(0, 0),
                image_data=None,
            )

    def binarize(self, image_data: np.ndarray):
        """
        Plan B: binarizacion adaptativa sobre imagen en grises.
        Solo debe usarse cuando el OCR sobre imagen original/mejorada falla.
        """
        if image_data is None:
            return None

        if len(image_data.shape) == 3:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)

        return cv2.adaptiveThreshold(
            image_data, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 15, 10
        )