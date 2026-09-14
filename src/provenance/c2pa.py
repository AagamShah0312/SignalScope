"""Best-effort C2PA / Content Credentials detection.

C2PA manifests are stored in a JUMBF box embedded in the file container
(a 'uuid' box in JPEG/PNG, or a 'jumb' box in MP4/HEIC).  A full C2PA
verification requires the official `c2pa` library and trust-store; to avoid
making the project fragile we implement a lightweight, honest *presence check*
that scans file bytes for the JUMBF superbox markers.

What "not detected" means (and what it does NOT mean):

* "Not detected" == "no machine-readable Content Credentials were found".
* It does **not** mean the image is AI-generated.
* It does **not** validate the cryptographic signature of any credential.

Provenance evidence is always kept separate from the visual model verdict.
"""

from __future__ import annotations

from typing import Dict, Optional

# JUMBF superbox markers ("jumb", "jumd") and the C2PA "c2pa" label appear as
# plain ASCII in the file when C2PA credentials are embedded.
_JUMBF_MARKERS = (b"jumb", b"jumd", b"c2pa", b"JUMBF")


def detect_c2pa(contents: bytes) -> Dict[str, object]:
    """Return a presence summary for the given file bytes."""
    if not contents:
        return {"present": False, "status": "not_detected",
                "note": "No machine-readable provenance credentials were found."}

    present = any(marker in contents for marker in _JUMBF_MARKERS)
    if present:
        return {
            "present": True,
            "status": "present_unverified",
            "note": (
                "Content Credentials markers were detected but not "
                "cryptographically verified by SignalScope."
            ),
        }
    return {
        "present": False,
        "status": "not_detected",
        "note": "No machine-readable provenance credentials were found.",
    }
