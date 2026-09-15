from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class PreprocessingResult:
    """Resultado del preprocesamiento"""
    output_path: str
    applied_operations: list = field(default_factory=list)
    original_size: tuple = (0, 0)
    processed_size: tuple = (0, 0)
    image_data: Optional[np.ndarray] = None  # Array NumPy en memoria (para OCR)


class ImagePreprocessorPort(ABC):
    """Puerto para preprocesamiento de imágenes"""

    @abstractmethod
    def process(self, image_path: str) -> PreprocessingResult:
        """Preprocesa una imagen para mejorar el OCR"""
        pass