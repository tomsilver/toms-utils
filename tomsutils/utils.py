"""Utilities."""

from dataclasses import fields
from functools import cached_property
from types import GenericAlias

_NOT_FOUND = object()


class cached_property_until_field_change(cached_property):
    """Decorator that caches a property in a dataclass until a field is
    changed."""

    def __get__(self, instance, owner=None):
        ### Same as cached_property ###
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use cached_property instance without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except (
            AttributeError
        ):  # not all objects have __dict__ (e.g. class defines slots)
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        ### Modified from cached_property ###
        f_vals = tuple(getattr(instance, f.name) for f in fields(instance))
        key = (self.attrname, f_vals)  # cached_property usesd self.attrname only
        val = cache.get(key, _NOT_FOUND)
        if val is _NOT_FOUND:
            val = self.func(instance)
            try:
                cache[key] = val
            except TypeError:
                msg = (
                    f"The '__dict__' attribute on {type(instance).__name__!r} instance "
                    f"does not support item assignment for caching {key!r} property."
                )
                raise TypeError(msg) from None
        return val

    __class_getitem__ = classmethod(GenericAlias)
