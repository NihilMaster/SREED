from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

NUMERIC_FIELDS = ("subtotal", "impuestos", "total")
ITEM_NUMERIC_FIELDS = ("cantidad", "precio_unitario", "total_item")
FECHA_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _as_float(value: Any) -> Optional[float]:
    """Convierte valores numericos o strings con formato local a float."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None

    cleaned = value.strip()
    if not cleaned:
        return None

    cleaned = re.sub(r"[A-Za-z$€£]", "", cleaned).replace(" ", "")
    if not cleaned:
        return None

    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts) == 2 and len(parts[1]) == 3:
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(",", ".")
    elif "." in cleaned:
        parts = cleaned.split(".")
        if len(parts) == 2 and len(parts[1]) == 3:
            cleaned = cleaned.replace(".", "")

    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_invoice_structure(structure: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte strings numericos a float para consumo downstream."""
    normalized = dict(structure)

    for field in NUMERIC_FIELDS:
        if field in normalized:
            value = _as_float(normalized[field])
            if value is not None:
                normalized[field] = value

    items = normalized.get("items")
    if isinstance(items, list):
        new_items = []
        for item in items:
            if not isinstance(item, dict):
                continue
            new_item = dict(item)
            for field in ITEM_NUMERIC_FIELDS:
                value = _as_float(new_item.get(field))
                if value is not None:
                    new_item[field] = value
            new_items.append(new_item)
        normalized["items"] = new_items

    return normalized


def validate_invoice_structure(structure: Dict[str, Any]) -> List[str]:
    """Validaciones deterministicas posteriores al LLM."""
    issues: List[str] = []

    subtotal = _as_float(structure.get("subtotal"))
    impuestos = _as_float(structure.get("impuestos"))
    total = _as_float(structure.get("total"))

    if total is None:
        issues.append("total_ausente")

    if total is not None and subtotal is not None:
        if impuestos is not None:
            esperado = subtotal + impuestos
            if abs(esperado - total) > 0.05:
                issues.append(
                    f"suma_inconsistente: subtotal+impuestos={esperado:.2f} vs total={total:.2f}"
                )
        elif abs(subtotal - total) > 0.05:
            issues.append("impuestos_no_declarados: total difiere de subtotal pero impuestos es null")

    items = structure.get("items")
    if isinstance(items, list) and items and subtotal is not None:
        suma_items = sum(
            _as_float(i.get("total_item")) or 0.0
            for i in items if isinstance(i, dict)
        )
        if suma_items > 0 and abs(suma_items - subtotal) > 0.05:
            issues.append(f"items_no_suman: {suma_items:.2f} vs subtotal={subtotal:.2f}")

    fecha = structure.get("fecha")
    if fecha is not None and not FECHA_RE.match(str(fecha)):
        issues.append(f"fecha_formato_invalido: {fecha}")

    for field in NUMERIC_FIELDS:
        raw = structure.get(field)
        if raw is not None and not isinstance(raw, (int, float)):
            issues.append(f"tipo_no_numerico: {field}")

    return issues