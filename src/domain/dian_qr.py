from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

DIAN_QR_URL = "https://catalogo-vpfe.dian.gov.co/document/searchqr?documentkey="

DIAN_QR_KEYS = (
    "NumFac", "FecFac", "HorFac", "NitFac", "DocAdq",
    "ValFac", "ValIva", "ValOtroIm", "ValTolFac", "CUFE",
)

NUMERIC_KEYS = ("ValFac", "ValIva", "ValOtroIm", "ValTolFac")


def parse_dian_qr(qr_data: str) -> Optional[Dict[str, Any]]:
    """
    Parseo tolerante del QR fiscal DIAN.

    El payload real viene con bytes de control y basura intercalada
    (ej: 'https://catal c0 .ogo-vpfe...'), por lo que se extrae cada
    clave por regex en lugar de asumir lineas limpias.
    """
    if not qr_data or "NumFac" not in qr_data:
        return None

    parsed: Dict[str, Any] = {}

    for key in DIAN_QR_KEYS:
        match = re.search(rf"{key}\s*:\s*([^\n\r]+)", qr_data)
        if match:
            parsed[key] = match.group(1).strip()

    if not parsed.get("NumFac"):
        return None

    for key in NUMERIC_KEYS:
        if key in parsed:
            parsed[key] = _to_float(parsed[key])

    if "CUFE" in parsed:
        # El CUFE valido es puramente hexadecimal: se limpia cualquier basura
        parsed["CUFE"] = re.sub(r"[^0-9a-fA-F]", "", parsed["CUFE"])

    return parsed


def apply_qr_fiscal_crosscheck(
    estructura: Dict[str, Any],
    qr_fiscal: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    """
    El QR fiscal esta amparado por el CUFE firmado ante la DIAN:
    sus valores son fuente autoritativa y preferimos esos sobre OCR/LLM.

    Semantica asumida (anexo tecnico DIAN):
    - ValFac  = valor total de la factura (impuestos incluidos)
    - ValIva + ValOtroIm = impuestos
    - subtotal = ValFac - impuestos
    """
    nueva = dict(estructura)
    cambios: List[str] = []

    def set_field(field: str, value: Any) -> None:
        if value is not None and nueva.get(field) != value:
            cambios.append(field)
            nueva[field] = value

    if qr_fiscal.get("NitFac"):
        set_field("nit", str(qr_fiscal["NitFac"]))
    if qr_fiscal.get("FecFac"):
        set_field("fecha", str(qr_fiscal["FecFac"]))
    if qr_fiscal.get("NumFac"):
        set_field("numero_factura", str(qr_fiscal["NumFac"]))

    val_fac = qr_fiscal.get("ValFac")
    if val_fac is not None:
        impuestos = round((qr_fiscal.get("ValIva") or 0.0) + (qr_fiscal.get("ValOtroIm") or 0.0), 2)
        set_field("total", val_fac)
        set_field("impuestos", impuestos)
        set_field("subtotal", round(val_fac - impuestos, 2))

    return nueva, cambios


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = re.sub(r"[^0-9.\-]", "", str(value))
    try:
        return float(cleaned)
    except ValueError:
        return None