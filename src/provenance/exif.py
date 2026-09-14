"""Safe EXIF metadata extraction.

Only camera / editing-software / orientation / dimensions fields are exposed.
Sensitive location fields (GPS) are deliberately excluded.

**EXIF absence is NOT evidence of AI generation.**  Many legitimate pipelines
strip metadata, and many generators preserve none.  This module only reports
what is present.
"""

from __future__ import annotations

from typing import Dict, Optional

from PIL import Image, ExifTags

# EXIF tags we are willing to surface (all non-sensitive).
SAFE_TAGS = {
    "Make", "Model", "Software", "DateTime", "DateTimeOriginal",
    "DateTimeDigitized", "Orientation", "ImageWidth", "ImageLength",
    "ExposureTime", "FNumber", "ISOSpeedRatings", "LensMake", "LensModel",
    "Artist", "Copyright", "ImageDescription", "ProcessingSoftware",
}

# Mapping of PIL numeric tags to names, where the numeric tag is the IFD id.
def _tag_name_map():
    mapping = {}
    try:
        mapping = {v: k for k, v in ExifTags.TAGS.items()}
    except Exception:
        pass
    return mapping


def extract_exif(image: Image.Image) -> Dict[str, object]:
    """Extract a small, safe subset of EXIF fields.

    Returns ``{"present": bool, "fields": {...}}``.  Never raises — a missing
    or corrupt EXIF block returns ``present: False``.
    """
    result: Dict[str, object] = {"present": False, "fields": {}}
    try:
        exif = image.getexif()
    except Exception:
        return result

    if not exif:
        return result

    tag_names = _tag_name_map()
    fields: Dict[str, object] = {}
    for tag_id, value in exif.items():
        name = tag_names.get(tag_id, str(tag_id))
        if name in SAFE_TAGS and value is not None:
            # Normalise IFDRational values to strings for JSON safety.
            fields[name] = _safe_value(value)

    if fields:
        result["present"] = True
        result["fields"] = fields
    return result


def _safe_value(value) -> object:
    try:
        if hasattr(value, "numerator") and hasattr(value, "denominator"):
            return float(value)
    except Exception:
        pass
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="replace").rstrip("\x00")
        except Exception:
            return None
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


def summarize_exif(fields: Dict[str, object]) -> Dict[str, object]:
    """Human-facing summary of EXIF fields for the UI."""
    return {
        "camera_make": fields.get("Make"),
        "camera_model": fields.get("Model"),
        "software": fields.get("Software"),
        "datetime_original": fields.get("DateTimeOriginal") or fields.get("DateTime"),
        "orientation": fields.get("Orientation"),
    }
