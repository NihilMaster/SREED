from pathlib import Path
from typing import List
import logging

from src.application.ports.document_processor import DocumentProcessor
from src.domain.models import DocumentResult, DocumentStatus
from src.infrastructure.adapters.ocr.opencv_preprocessor import OpenCVPreprocessor
from src.infrastructure.adapters.ocr.easyocr_extractor import EasyOCRExtractor
from src.infrastructure.adapters.ocr.pyzbar_extractor import PyzbarQRExtractor
from src.infrastructure.adapters.ocr.pymupdf_adapter import PyMuPDFAdapter

logger = logging.getLogger(__name__)


class OCRRealDocumentProcessor(DocumentProcessor):
    """
    Procesador OCR real con estrategia resiliente de multiples pasadas:

    Paso 1: OCR sobre imagen original.
    Paso 2: Si la confianza es baja, OCR sobre imagen realzada (CLAHE + upscale).
    Paso 3: Si sigue baja, OCR sobre imagen binarizada.

    Se conserva el mejor resultado por confianza.
    """

    OCR_MIN_CONFIDENCE = 0.6

    def __init__(self, languages: list = None, use_gpu: bool = True):
        """
        Args:
            languages: Idiomas para OCR (ej: ['es', 'en'])
            use_gpu: Si usar GPU para EasyOCR
        """
        self.preprocessor = OpenCVPreprocessor()
        self.ocr = EasyOCRExtractor(languages=languages, use_gpu=use_gpu)
        self.qr_extractor = PyzbarQRExtractor()
        self.pdf_adapter = PyMuPDFAdapter()
        logger.info("OCRRealDocumentProcessor inicializado")

    def process(self, source_path: Path) -> DocumentResult:
        try:
            if not source_path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {source_path}")

            file_path_str = str(source_path)

            if source_path.suffix.lower() == '.pdf':
                image_paths = self.pdf_adapter.pdf_to_images(file_path_str)
            else:
                image_paths = [file_path_str]

            all_texts = []
            all_qr_codes = []
            total_confidence = 0.0
            passes_used = []

            for img_path in image_paths:
                try:
                    best_result, used_pass = self._ocr_best_pass(img_path)

                    if best_result is not None and best_result.is_valid():
                        all_texts.append(best_result.text)
                        total_confidence += best_result.confidence
                        passes_used.append(used_pass)

                    qr_codes = self.qr_extractor.extract(img_path)
                    all_qr_codes.extend(qr_codes)

                except Exception as e:
                    logger.error(f"Error procesando imagen {img_path}: {e}")

            full_text = "\n\n".join(all_texts)
            avg_confidence = total_confidence / len(image_paths) if image_paths else 0.0

            qr_data = [
                {"data": qr.data, "type": qr.qr_type}
                for qr in all_qr_codes
            ]

            payload = {
                "documento": source_path.name,
                "tamano_bytes": source_path.stat().st_size,
                "extractor": "ocr_real",
                "raw_text": full_text,
                "qr": qr_data,
                "num_pages": len(image_paths),
                "num_qr_codes": len(all_qr_codes),
                "ocr_confidence": avg_confidence,
                "ocr_passes": passes_used,
                "estructura": {
                    "proveedor": None,
                    "nit": None,
                    "fecha": None,
                    "moneda": None,
                    "subtotal": None,
                    "impuestos": None,
                    "total": None,
                },
            }

            logger.info(
                f"Procesamiento completado: {len(all_texts)} textos, "
                f"{len(all_qr_codes)} QR, confianza: {avg_confidence:.2f}, "
                f"pasadas: {passes_used}"
            )

            return DocumentResult(
                source_path=source_path,
                status=DocumentStatus.SUCCESS,
                message="OCR completado exitosamente",
                payload=payload,
            )

        except Exception as e:
            logger.error(f"Error en procesamiento OCR: {e}")
            return DocumentResult(
                source_path=source_path,
                status=DocumentStatus.FAILED,
                message=f"Error en OCR: {str(e)}",
                payload={"exception": str(e), "exception_type": type(e).__name__},
            )

    def _ocr_best_pass(self, img_path: str):
        """
        Ejecuta hasta 3 pasadas de OCR y devuelve (mejor_resultado, nombre_pasada).
        """
        # Paso 1: imagen original
        best = self.ocr.extract(img_path)
        used = "original"

        if best.confidence >= self.OCR_MIN_CONFIDENCE:
            return best, used

        # Paso 2: imagen realzada
        prep = self.preprocessor.process(img_path)
        if prep.image_data is not None:
            alt = self.ocr.extract(img_path, image_data=prep.image_data)
            if alt.confidence > best.confidence:
                best = alt
                used = "enhanced"

            # Paso 3: binarizada (plan B)
            if best.confidence < self.OCR_MIN_CONFIDENCE:
                bin_img = self.preprocessor.binarize(prep.image_data)
                if bin_img is not None:
                    alt2 = self.ocr.extract(img_path, image_data=bin_img)
                    if alt2.confidence > best.confidence:
                        best = alt2
                        used = "binarized"

        logger.info(f"Mejor pasada OCR para {img_path}: {used} ({best.confidence:.2f})")
        return best, used