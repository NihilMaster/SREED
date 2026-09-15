import pymupdf
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class PyMuPDFAdapter:
    """Adaptador para convertir PDFs a imágenes"""

    def __init__(self, temp_dir: str = "resources/temp", dpi: int = 200):
        """
        Args:
            temp_dir: Directorio temporal para imágenes extraídas
            dpi: Resolución de las imágenes (mayor = mejor calidad pero más lento)
        """
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def pdf_to_images(self, pdf_path: str) -> List[str]:
        """
        Convierte un PDF a lista de imágenes (una por página).
        Guarda las imágenes en la carpeta temporal.

        Args:
            pdf_path: Ruta al archivo PDF

        Returns:
            Lista de rutas a las imágenes generadas
        """
        try:
            if not Path(pdf_path).exists():
                raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

            # Abrir PDF
            doc = pymupdf.open(pdf_path)
            image_paths = []

            # Procesar cada página
            for page_num in range(len(doc)):
                page = doc[page_num]

                # Renderizar página a imagen
                mat = pymupdf.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                # Guardar imagen
                image_path = self.temp_dir / f"{Path(pdf_path).stem}_page_{page_num + 1}.png"
                pix.save(str(image_path))
                image_paths.append(str(image_path))

                logger.info(f"Página {page_num + 1} convertida a imagen")

            doc.close()

            logger.info(f"PDF convertido a {len(image_paths)} imágenes")
            return image_paths

        except Exception as e:
            logger.error(f"Error convirtiendo PDF: {e}")
            raise

    def is_pdf(self, file_path: str) -> bool:
        """Verifica si un archivo es un PDF válido"""
        try:
            if not Path(file_path).exists():
                return False

            doc = pymupdf.open(file_path)
            is_valid = len(doc) > 0
            doc.close()
            return is_valid

        except Exception:
            return False