"""Utilities."""

import hashlib
import os
from dataclasses import fields
from functools import cached_property
from pathlib import Path
from typing import Any, Collection, Tuple

import graphviz
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


def fig2data(fig: plt.Figure) -> Image:
    """Convert matplotlib figure into Image."""
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())  # type: ignore


def wrap_angle(angle: float) -> float:
    """Wrap an angle in radians to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def get_signed_angle_distance(target: float, source: float) -> float:
    """Given two angles between [-pi, pi], get the smallest signed angle d s.t.

    source + d = target.
    """
    assert -np.pi <= source <= np.pi
    assert -np.pi <= target <= np.pi
    a = target - source
    return (a + np.pi) % (2 * np.pi) - np.pi


def draw_dag(edges: Collection[Tuple[str, str]], outfile: Path) -> None:
    """Draw a DAG using graphviz."""
    if not outfile.parent.exists():
        os.makedirs(outfile.parent)
    intermediate_dot_file = outfile.parent / outfile.stem
    assert not intermediate_dot_file.exists()
    dot = graphviz.Digraph(format=outfile.suffix[1:])
    nodes = {e[0] for e in edges} | {e[1] for e in edges}
    for node in nodes:
        dot.node(node)
    for node1, node2 in edges:
        dot.edge(node1, node2)
    dot.render(outfile.stem, directory=outfile.parent)
    os.remove(intermediate_dot_file)
    print(f"Wrote out to {outfile}")


def consistent_hash(obj: Any) -> int:
    """A hash function that is consistent between sessions, unlike hash()."""
    obj_str = repr(obj)
    obj_bytes = obj_str.encode("utf-8")
    hash_hex = hashlib.sha256(obj_bytes).hexdigest()
    hash_int = int(hash_hex, 16)
    # Mimic Python's built-in hash() behavior by returning a 64-bit signed int.
    # This makes it comparable to hash()'s output range.
    return hash_int if hash_int < 2**63 else hash_int - 2**6
