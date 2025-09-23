import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("FOGSIGHT_API_KEY", "test-key")
os.environ.setdefault("FOGSIGHT_CREDENTIALS_PATH", str(ROOT / "demo-credentials.json"))

from app import _coerce_positive_float, _normalize_voiceover_text, _plan_audio_muxing


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, None),
        ("", None),
        ("not a number", None),
        (-1, None),
        (0, None),
        (1, 1.0),
        ("3.5", 3.5),
        ("3s", 3.0),
        ("2 秒", 2.0),
        ("1.5 second", 1.5),
        ("6000ms", 6.0),
        ("6000 毫秒", 6.0),
        ("durationMs: 4500", 4.5),
        ("Delay=750ms", 0.75),
        (6000, 6.0),
        ("6000", 6.0),
        ("120000", 120.0),
    ],
)
def test_coerce_positive_float(raw, expected):
    result = _coerce_positive_float(raw)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert math.isclose(result, expected, rel_tol=1e-6)


def test_values_without_units_remain_seconds():
    assert _coerce_positive_float(90) == pytest.approx(90.0)
    assert _coerce_positive_float("42") == pytest.approx(42.0)


def test_normalize_voiceover_text_collapses_lines():
    text = "第一句。\n第二句\n\n第三句\r\nFinally, an English line"
    normalized = _normalize_voiceover_text(text)
    assert "第一句。" in normalized
    assert "第二句。" in normalized
    assert "第三句。" in normalized
    assert "Finally, an English line" in normalized
    assert "\n" not in normalized


@pytest.mark.parametrize(
    "video_duration, audio_duration, expected_filters, expected_shortest",
    [
        (13.26, 12.76, ["apad"], True),
        (10.0, 12.5, [], True),
        (10.0, 10.0, [], False),
        (10.0, 9.9, ["apad"], True),
        (None, 12.0, ["apad"], True),
        (10.0, None, ["apad"], True),
    ],
)
def test_plan_audio_muxing(video_duration, audio_duration, expected_filters, expected_shortest):
    filters, include_shortest = _plan_audio_muxing(video_duration, audio_duration)
    assert filters == expected_filters
    assert include_shortest is expected_shortest
