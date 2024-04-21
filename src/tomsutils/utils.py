"""Utilities."""

from dataclasses import fields
from functools import cached_property

import matplotlib.pyplot as plt
import numpy as np

from tomsutils.structs import Image

_NOT_FOUND = object()


class _DISABLED_cached_property_until_field_change(cached_property):
    """Decorator that caches a property in a dataclass until any field is
    changed.

    This descriptor is currently disabled because it does not play well
    with pylint. For example, see
    https://stackoverflow.com/questions/74523859/

    It is left here in case future versions of python / pylint have better
    support for custom property-like descriptors.
    """

    def __get__(self, instance, owner=None):
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property_until_field_change instance "
                "without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except AttributeError:  # not all objects have __dict__
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        field_key = f"_cached_property_until_field_change_{self.attrname}_field"
        cur_field_vals = tuple(getattr(instance, f.name) for f in fields(instance))
        last_field_vals = cache.get(field_key, _NOT_FOUND)
        prop_key = f"_cached_property_until_field_change_{self.attrname}_property"
        if cur_field_vals == last_field_vals:
            return cache[prop_key]
        # Fields were updated, so we need to recompute and update both caches.
        new_prop_val = self.func(instance)
        cache[field_key] = cur_field_vals
        cache[prop_key] = new_prop_val
        return new_prop_val


def fig2data(fig: plt.Figure, dpi: int) -> Image:
    """Convert matplotlib figure into Image."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).copy()
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    data[..., [0, 1, 2, 3]] = data[..., [1, 2, 3, 0]]
    return data
