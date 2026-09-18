import cv2
import pyzbar.pyzbar as pyzbar
from pathlib import Path
from typing import List
import logging

from src.application.ports.qr_extractor import QRExtractorPort, QRCode

logger = logging.getLogger(__name__)


class PyzbarQRExtractor(QRExtractorPort):
    """
    Extractor QR resiliente: prueba pyzbar y el detector de OpenCV sobre
    multiples variantes de la imagen (escalas y umbrales), porque los QR
    densos de la DIAN suelen requerir escalado para decodificarse.
    """

    def extract(self, image_path: str) -> List[QRCode]:
        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

            img = cv2.imread(image_path)
            if img is None:
                return []

            for name, candidate in self._build_candidates(img):
                codes = self._decode_pyzbar(candidate)
                if not codes:
                    codes = self._decode_opencv(candidate)
                if codes:
                    logger.info(f"QR detectado con estrategia: {name} ({len(codes)} códigos)")
                    return codes

            logger.info(f"No se detectaron códigos QR en: {image_path}")
            return []

        except Exception as e:
            logger.warning(f"Error extrayendo QR (no crítico): {e}")
            return []

    def _build_candidates(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        up2 = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        up3 = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, up2_otsu = cv2.threshold(up2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return [
            ("original", img),
            ("gris", gray),
            ("otsu", otsu),
            ("up2", up2),
            ("up3", up3),
            ("up2_otsu", up2_otsu),
        ]

    def _decode_pyzbar(self, candidate) -> List[QRCode]:
        try:
            decoded = pyzbar.decode(candidate)
        except Exception:
            return []
        codes = []
        for obj in decoded:
            rect = obj.rect
            try:
                data = obj.data.decode("utf-8")
            except Exception:
                data = str(obj.data)
            codes.append(QRCode(data=data, qr_type=obj.type, position=(rect.left, rect.top, rect.width, rect.height)))
        return codes

    def _decode_opencv(self, candidate) -> List[QRCode]:
        try:
            detector = cv2.QRCodeDetector()
            data, points, _ = detector.detectAndDecode(candidate)
        except Exception:
            return []
        if data:
            return [QRCode(data=data, qr_type="QRCODE", position=None)]
        return []