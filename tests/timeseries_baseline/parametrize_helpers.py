# Derived from Unit8 Darts (darts/tests/parametrize_helpers.py), Apache-2.0.
# Copyright 2019 Unit8 SA. Additional contributions by the Darts contributors.
# Modified for native Anomx imports and configuration names.
# See THIRD_PARTY_NOTICES.md and licenses/DARTS-APACHE-2.0.txt.

from itertools import product as _product


def param_product(*iterables):
    """Return the Cartesian product of *iterables as a list for pytest.mark.parametrize."""
    return list(_product(*iterables))


def param_zip(*iterables, strict=False):
    """Return zip(*iterables) as a list for pytest.mark.parametrize."""
    return list(zip(*iterables, strict=strict))


def param_list(iterable):
    """Materialize an iterable as a list for pytest.mark.parametrize."""
    return list(iterable)
