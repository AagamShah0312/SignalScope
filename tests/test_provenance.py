"""Tests for provenance (EXIF + C2PA) handling."""

from __future__ import annotations

from src.provenance.c2pa import detect_c2pa
from src.provenance.exif import extract_exif, summarize_exif


def test_exif_absent_handled_safely(sample_rgb_image):
    result = extract_exif(sample_rgb_image)
    assert result["present"] is False
    assert result["fields"] == {}


def test_exif_summary_empty():
    summary = summarize_exif({})
    assert summary["camera_make"] is None


def test_c2pa_absent_handled_safely():
    result = detect_c2pa(b"plain image bytes with no markers")
    assert result["present"] is False
    assert result["status"] == "not_detected"
    # "Not detected" must NOT be presented as "AI-generated".
    assert "ai" not in result["note"].lower()


def test_c2pa_empty_bytes():
    result = detect_c2pa(b"")
    assert result["present"] is False


def test_c2pa_marker_detection():
    result = detect_c2pa(b"\x00\x01jumb\x00c2pa\x00\xff")
    assert result["present"] is True
