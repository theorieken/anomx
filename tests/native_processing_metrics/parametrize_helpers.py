# Copyright 2019 Unit8 SA; additional contributions by Darts contributors.
# Licensed under the Apache License, Version 2.0 (see licenses/DARTS-APACHE-2.0.txt).
# Derived from Unit8 Darts, commit dcdde2414f07f13b3bbb095428d9c6cc41492588.
# Modified for Anomx: native namespace and integration with Anomx data APIs.

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
