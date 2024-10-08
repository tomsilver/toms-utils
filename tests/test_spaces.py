"""Tests for spaces.py."""

from tomsutils.spaces import EnumSpace


def test_enum_space():
    """Tests for EnumSpace()."""
    elements = ["cat", "dog", "horse"]
    space = EnumSpace(elements)
    space.seed(123)
    elm = space.sample()
    assert elm in elements
    assert space.contains(elm)
    assert not space.is_np_flattenable
