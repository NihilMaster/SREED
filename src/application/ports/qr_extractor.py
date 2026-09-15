from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class QRCode:
    """Código QR detectado"""
    data: str
    qr_type: str
    position: Optional[tuple] = None  # (x, y, w, h)


class QRExtractorPort(ABC):
    """Puerto para extracción de códigos QR"""
    
    @abstractmethod
    def extract(self, image_path: str) -> List[QRCode]:
        """Extrae códigos QR de una imagen"""
        pass