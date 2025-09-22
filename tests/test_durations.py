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

from app import _coerce_positive_float


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
