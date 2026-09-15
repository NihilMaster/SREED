import easyocr
from pathlib import Path
from typing import List, Optional
import logging
import numpy as np

from src.application.ports.ocr_extractor import OCRPort, OCRResult

logger = logging.getLogger(__name__)


class EasyOCRExtractor(OCRPort):
    """Extractor OCR usando EasyOCR"""

    def __init__(self, languages: List[str] = None, use_gpu: bool = True):
        self.languages = languages or ['es', 'en']
        self.use_gpu = use_gpu
        logger.info(f"Inicializando EasyOCR con idiomas: {self.languages}, GPU: {self.use_gpu}")
        self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu)
        logger.info("EasyOCR inicializado correctamente")

    def extract(self, image_path: str, image_data: Optional[np.ndarray] = None) -> OCRResult:
        """
        Extrae texto de una imagen usando EasyOCR.

        Args:
            image_path: Ruta de referencia de la imagen. Se usa si image_data es None.
            image_data: Array NumPy preprocesado (realzado o binarizado). Opcional.
        """
        try:
            if image_data is not None:
                results = self.reader.readtext(image_data)
            else:
                if not Path(image_path).exists():
                    raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
                results = self.reader.readtext(image_path)

            if not results:
                logger.warning(f"EasyOCR no detecto texto en: {image_path}")
                return OCRResult(text="", confidence=0.0, boxes=[])

            texts = []
            boxes = []
            confidences = []

            for (bbox, text, confidence) in results:
                texts.append(text)
                confidences.append(confidence)
                boxes.append({"bbox": bbox, "text": text, "confidence": confidence})

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            full_text = "\n".join(texts)

            logger.info(f"OCR completado: {len(texts)} cajas, confianza: {avg_confidence:.2f}")
            return OCRResult(text=full_text, confidence=avg_confidence, boxes=boxes)

        except Exception as e:
            logger.error(f"Error en OCR: {e}")
            raise