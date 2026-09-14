"""Security-focused tests for SignalScope.

Covers the threat surface that matters for a public image-upload service:

* input validation & upload limits (size / extension / MIME / content spoofing)
* decompression-bomb defence (header dimension checks before pixel decode)
* path-traversal protection (static files + Grad-CAM output names)
* metadata privacy (EXIF GPS redaction)
* error-response hygiene (no internal details / stack traces leaked)
* responsible-claims hygiene (fixed verdict vocabulary, no absolute accusations)
* CORS behaviour

These tests fail if a security guard is removed or weakened.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.preprocessing as preprocessing
from app import api as api_module
from app.api import app
from app.preprocessing import ImageValidationError, validate_file
from src.explainability.explanation import build_verdict
from src.explainability.gradcam import save_visualization
from src.provenance.c2pa import detect_c2pa
from src.provenance.exif import _safe_value, extract_exif


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def _png_bytes(size=(200, 200), color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Input validation & upload limits
# ---------------------------------------------------------------------------
class TestInputValidation:
    def test_empty_file_rejected(self):
        with pytest.raises(ImageValidationError):
            validate_file(b"", filename="x.png")

    def test_oversized_byte_payload_rejected(self):
        with pytest.raises(ImageValidationError) as exc:
            validate_file(b"0" * 2048, filename="x.png", max_bytes=1024)
        assert "large" in str(exc.value).lower()

    @pytest.mark.parametrize("ext", [".exe", ".html", ".php", ".txt", ".svg"])
    def test_unsupported_extension_rejected(self, ext, sample_png_bytes):
        with pytest.raises(ImageValidationError):
            validate_file(sample_png_bytes, filename=f"attack{ext}")

    def test_unsupported_mime_rejected(self, sample_png_bytes):
        with pytest.raises(ImageValidationError):
            validate_file(sample_png_bytes, filename="x.png", mime_type="text/html")

    def test_content_spoofing_rejected(self):
        # A file that claims to be a PNG but contains HTML must be rejected.
        with pytest.raises(ImageValidationError):
            validate_file(b"<html><script>alert(1)</script></html>",
                          filename="x.png", mime_type="image/png")

    def test_tiny_image_rejected(self):
        with pytest.raises(ImageValidationError):
            validate_file(_png_bytes(size=(15, 15)), filename="x.png")

    def test_oversized_dimension_rejected(self):
        # 8001 > MAX_DIMENSION (8000); small pixel count so the check is cheap.
        with pytest.raises(ImageValidationError) as exc:
            validate_file(_png_bytes(size=(8001, 16)), filename="x.png")
        assert "dimension" in str(exc.value).lower()

    def test_too_many_pixels_rejected(self, monkeypatch):
        monkeypatch.setattr(preprocessing, "MAX_PIXELS", 10_000)
        with pytest.raises(ImageValidationError):
            validate_file(_png_bytes(size=(200, 200)), filename="x.png")

    def test_valid_image_still_accepted(self, sample_png_bytes):
        img = validate_file(sample_png_bytes, filename="x.png", mime_type="image/png")
        assert img.mode == "RGB"


# ---------------------------------------------------------------------------
# Decompression-bomb defence
# ---------------------------------------------------------------------------
class TestDecompressionBomb:
    def test_bomb_error_is_caught_not_propagated(self, monkeypatch):
        # Lower Pillow's own guard so a modest PNG triggers the bomb check.
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10_000)
        # 200x200 = 40_000 px > 2 * 10_000 -> DecompressionBombError at open.
        with pytest.raises(ImageValidationError):
            validate_file(_png_bytes(size=(200, 200)), filename="x.png")

    def test_header_dimensions_checked_before_decode(self, monkeypatch):
        # A header claiming a huge image must be rejected before pixel decode.
        monkeypatch.setattr(preprocessing, "MAX_PIXELS", 10_000)
        huge = _png_bytes(size=(500, 500))  # 250k px > 10k limit
        with pytest.raises(ImageValidationError):
            validate_file(huge, filename="x.png")


# ---------------------------------------------------------------------------
# Path traversal
# ---------------------------------------------------------------------------
class TestPathTraversal:
    @pytest.mark.parametrize("url", [
        "/files/../../etc/passwd",
        "/files/..%2f..%2fetc%2fpasswd",
        "/files/../config.yaml",
    ])
    def test_static_files_traversal_blocked(self, client, url):
        assert client.get(url).status_code == 404

    def test_gradcam_output_name_cannot_escape(self, tmp_path):
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        cam = np.full((32, 32), 0.5, dtype=np.float32)
        result = save_visualization(arr, cam, str(tmp_path), base_name="../../evil")
        for value in result.values():
            assert value and value.startswith(str(tmp_path.resolve()))
        # Nothing must be written outside the output directory.
        assert not (tmp_path.parent / "evil_original.jpg").exists()

    def test_api_visualization_stays_in_safe_dir(self, client, sample_png_bytes):
        r = client.post("/predict", files={"file": ("../../evil.png", sample_png_bytes, "image/png")})
        assert r.status_code == 200
        overlay = r.json()["visualization"]["overlay"]
        assert overlay.startswith("/files/api/")
        assert ".." not in overlay


# ---------------------------------------------------------------------------
# API behaviour under abuse
# ---------------------------------------------------------------------------
class TestApiAbuse:
    def test_upload_bomb_rejected_413(self, client, monkeypatch):
        monkeypatch.setitem(api_module.cfg.inference, "max_upload_mb", 1)
        payload = b"0" * (2 * 1024 * 1024)  # 2 MB
        r = client.post("/predict", files={"file": ("x.png", payload, "image/png")})
        assert r.status_code == 413

    def test_unsupported_extension_rejected_via_api(self, client, sample_png_bytes):
        r = client.post("/predict", files={"file": ("x.exe", sample_png_bytes, "image/png")})
        assert r.status_code == 400

    def test_error_response_leaks_no_internals(self, client):
        r = client.post("/predict", files={"file": ("x.png", b"\x89PNG corrupt", "image/png")})
        assert r.status_code == 400
        body = r.text
        assert "Traceback" not in body
        assert "/home/" not in body
        assert "File \"" not in body

    def test_model_unavailable_returns_503(self, client, monkeypatch):
        monkeypatch.setattr(api_module, "predictor", None)
        monkeypatch.setattr(api_module, "model_error", "test model missing")
        r = client.post("/predict", files={"file": ("x.png", b"data", "image/png")})
        assert r.status_code == 503

    def test_filename_is_not_reflected(self, client, sample_png_bytes):
        r = client.post("/predict", files={
            "file": ("<script>alert(1)</script>.png", sample_png_bytes, "image/png")})
        assert "<script>" not in r.text


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
class TestCors:
    def test_preflight_returns_permissive_origin(self, client):
        r = client.options("/predict", headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "POST",
        })
        assert r.status_code == 200
        assert r.headers.get("access-control-allow-origin") == "*"


# ---------------------------------------------------------------------------
# Metadata privacy (EXIF)
# ---------------------------------------------------------------------------
class TestMetadataPrivacy:
    def _image_with_gps(self) -> Image.Image:
        import piexif
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Make: b"AcmeCamera",
                piexif.ImageIFD.Model: b"X100",
            },
            "GPS": {
                piexif.GPSIFD.GPSLatitude: ((40, 1), (26, 1), (46, 1)),
                piexif.GPSIFD.GPSLatitudeRef: b"N",
                piexif.GPSIFD.GPSLongitude: ((79, 1), (58, 1), (59, 1)),
                piexif.GPSIFD.GPSLongitudeRef: b"W",
            },
        }
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (120, 130, 140)).save(
            buf, format="JPEG", exif=piexif.dump(exif_dict))
        buf.seek(0)
        return Image.open(buf)

    def test_gps_never_exposed(self):
        result = extract_exif(self._image_with_gps())
        fields = result["fields"]
        assert result["present"] is True
        # Non-sensitive camera fields are surfaced...
        assert fields.get("Make") == "AcmeCamera"
        # ...but location data must never be.
        for key in fields:
            assert "GPS" not in key and "gps" not in key
        assert "GPSLatitude" not in fields
        assert "GPSLongitude" not in fields

    def test_bytes_value_null_stripped(self):
        assert _safe_value(b"AcmeCamera\x00\x00") == "AcmeCamera"


# ---------------------------------------------------------------------------
# Provenance honesty (C2PA)
# ---------------------------------------------------------------------------
class TestC2paHonesty:
    def test_presence_never_claimed_as_verified(self):
        result = detect_c2pa(b"\x00jumb\x00c2pa\x00")
        assert result["present"] is True
        assert result["status"] == "present_unverified"
        assert "not cryptographically verified" in result["note"]


# ---------------------------------------------------------------------------
# Responsible-claims hygiene
# ---------------------------------------------------------------------------
class TestResponsibleClaims:
    def test_verdict_vocabulary_is_fixed(self):
        assert build_verdict(0.99, 0.8, 0.2)["label"] == "Likely AI-generated"
        assert build_verdict(0.01, 0.8, 0.2)["label"] == "Likely real"
        assert build_verdict(0.50, 0.8, 0.2)["label"] == "Inconclusive"

    def test_no_absolute_accusations(self, client, sample_png_bytes):
        r = client.post("/predict", files={"file": ("x.png", sample_png_bytes, "image/png")})
        body = r.json()
        summary = body["explanation"]["summary"]
        # Responsible language only — never a definitive accusation.
        for banned in ("FAKE", "definitely", "100% proof", "guaranteed"):
            assert banned.lower() not in summary.lower()
