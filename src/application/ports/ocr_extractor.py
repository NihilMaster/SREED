from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class OCRResult:
    """Resultado de la extracción OCR"""
    text: str
    confidence: float
    boxes: List[dict]  # Lista de cajas de texto con coordenadas
    
    def is_valid(self) -> bool:
        """Verifica si el OCR produjo texto válido"""
        return len(self.text.strip()) > 0 and self.confidence > 0.1


class OCRPort(ABC):
    """Puerto para motores de OCR"""
    
    @abstractmethod
    def extract(self, image_path: str) -> OCRResult:
        """Extrae texto de una imagen"""
        pass