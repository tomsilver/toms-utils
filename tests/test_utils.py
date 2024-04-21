"""Tests for utils.py."""

from dataclasses import dataclass

import numpy as np

from tomsutils.utils import _DISABLED_cached_property_until_field_change, wrap_angle


def test_cached_property_until_field_change():
    """Tests for cached_property_until_field_change()."""
    # pylint: disable=comparison-with-callable

    num_times_invoked = 0

    @dataclass
    class A:
        """A test class."""

        x: int
        y: int

        @_DISABLED_cached_property_until_field_change
        def xy(self) -> int:
            """Returns x * y."""
            nonlocal num_times_invoked
            num_times_invoked += 1
            return self.x * self.y

    a = A(x=3, y=4)
    assert a.xy == 12
    assert num_times_invoked == 1
    assert a.xy == 12
    assert num_times_invoked == 1
    a.x = 5
    assert a.xy == 20
    assert num_times_invoked == 2
    assert a.xy == 20
    assert num_times_invoked == 2
    a.y = -2
    assert a.xy == -10
    assert num_times_invoked == 3
    assert a.xy == -10
    assert num_times_invoked == 3
    a.x = 3
    a.y = 4
    assert a.xy == 12
    assert num_times_invoked == 4
    assert a.xy == 12
    assert num_times_invoked == 4


def test_wrap_angle():
    """Tests for wrap_angle()."""
    assert np.isclose(wrap_angle(0.0), 0.0)
    assert np.isclose(wrap_angle(np.pi / 2), np.pi / 2)
    assert np.isclose(wrap_angle(-np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(np.pi), np.pi)
    assert np.isclose(wrap_angle(-np.pi), -np.pi)
    assert np.isclose(wrap_angle(2 * np.pi), 0.0)
    assert np.isclose(wrap_angle(3 * np.pi / 2), -np.pi / 2)
    assert np.isclose(wrap_angle(5 * np.pi / 2), np.pi / 2)
