import pyzbar.pyzbar as pyzbar
from PIL import Image
from pathlib import Path
from typing import List
import logging

from src.application.ports.qr_extractor import QRExtractorPort, QRCode

logger = logging.getLogger(__name__)


class PyzbarQRExtractor(QRExtractorPort):
    """Extractor de códigos QR usando pyzbar"""
    
    def extract(self, image_path: str) -> List[QRCode]:
        """
        Extrae códigos QR y códigos de barras de una imagen
        
        Returns:
            Lista de QRCode detectados
        """
        try:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
            
            # Abrir imagen con PIL
            img = Image.open(image_path)
            
            # Decodificar códigos
            decoded_objects = pyzbar.decode(img)
            
            if not decoded_objects:
                logger.info(f"No se detectaron códigos QR en: {image_path}")
                return []
            
            qr_codes = []
            for obj in decoded_objects:
                # Extraer posición
                rect = obj.rect
                position = (rect.left, rect.top, rect.width, rect.height)
                
                # Decodificar datos
                try:
                    data = obj.data.decode('utf-8')
                except:
                    data = str(obj.data)
                
                qr_codes.append(QRCode(
                    data=data,
                    qr_type=obj.type,
                    position=position
                ))
                
                logger.info(f"Detectado {obj.type}: {data[:50]}...")
            
            logger.info(f"Total códigos detectados: {len(qr_codes)}")
            return qr_codes
            
        except Exception as e:
            logger.warning(f"Error extrayendo QR (no crítico): {e}")
            # No es crítico si falla QR, continuar sin ellos
            return []