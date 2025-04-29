# Copyright 2025 The meqpy Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Thin file for controlling the shapes of LY from meq."""
from collections.abc import Iterable, Sequence
from typing import Any, Final

import jax
import numpy as np
import oct2py


def _flatten_oct2py_struct(
    s: oct2py.Struct,
) -> tuple[Iterable[Any], Sequence[Any]]:
  flattened_data = [(jax.tree_util.GetAttrKey(k), s[k]) for k in s.keys()]
  aux_data = list(s.keys())
  return flattened_data, aux_data


def _unflatten_oct2py_struct(
    aux_data: Sequence[Any], flat_data: Iterable[Any]
) -> oct2py.Struct:
  s = oct2py.Struct()
  for k, v in zip(aux_data, flat_data):
    s[k] = v
  return s


jax.tree_util.register_pytree_with_keys(
    oct2py.Struct,
    _flatten_oct2py_struct,
    _unflatten_oct2py_struct,
)


# Copied from meq/meqsize.m
# These lists are complete. If you want to change a fields handling add it to
# a list below rather than deleting it from scalar or vector keys. When adding
# please keep it alphabetical.

# NOTE: The LY structures we deal with can have essentially two forms, either
# they are a single timeslice or they represent the trajectory over time.
# Due to how matlab works (everything is a matrix), a singleton time dimension
# doesn't need to be defined and can be indexed into regardless.
# Numpy and python do not allow this kind of behaviour. What this means for this
# file is that "scalars" are things that have shape (t) but for a single
# timeslice that will be a (1) however for compatibility with numpy slicing its
# more convenient to drop the time dimension and have scalars as ().
# Similarly for "vectors" they are (N, t) but for single timeslice keeping that
# singleton time dimension results in a lot of code focused on squeezing.
# In the following code "with_time" will indicate when we are processing an LY
# that has non-singleton time dimension (like is generated from reconstruction
# via meq).
SCALAR_KEYS: Final[frozenset[str]] = frozenset([
    'bp', 'bpli2', 'bt', 'chi', 'chih', 'cycle', 'dz', 'dzg', 'err', 'Ft',
    'Ft0', 'Ini', 'Ip', 'isconverged', 'lB', 'li', 'mkryl', 'mu', 'nA', 'nB',
    'nfeval', 'niter', 'rBt', 'res', 'resC', 'rese', 'resFx', 'resp', 'resy',
    'rIp', 'rst', 'shot', 't', 'Vp', 'Wk', 'WN', 'Wp', 'Wt', 'Wt0', 'zIp'])


VECTOR_KEYS: Final[frozenset[str]] = frozenset([
    'ag', 'Bm', 'bpD', 'Brn', 'Brrn', 'Brzn', 'btD', 'Bzn', 'Bzrn', 'Bzzn',
    'dr2FA', 'dr2FX', 'drzFA', 'drzFX', 'dz2FA', 'dz2FX', 'F0', 'F1', 'FA',
    'FB', 'Fe', 'Fedot', 'Ff', 'Fn', 'FR', 'Ft0D', 'FtD', 'FW', 'FX', 'Ia',
    'Iarel', 'Ie', 'IniD', 'IpD', 'Is', 'Iu', 'Iv', 'liD', 'lp', 'Lp', 'lX',
    'nX', 'Parel', 'q95', 'qA', 'qmin', 'rA', 'raqmin', 'rB', 'rbary', 'rIpD',
    'Rp', 'rX', 'Tarel', 'Uf', 'Um', 'Va', 'Vn', 'VpD', 'WkD', 'WND', 'WpD',
    'Wt0D', 'WtD', 'zA', 'zB', 'zIpD', 'zX'])


# These keys have shape (N, D) but D *might* be squeezed so we ensure it exists
MATRIX_WITH_DOMAIN_KEYS: Final[frozenset[str]] = frozenset([
    'aminor', 'AQ', 'aW', 'delta', 'deltal', 'deltau', 'epsilon', 'FtPQ', 'IpQ',
    'iqQ', 'ItQ', 'iTQ', 'jtorQ', 'kappa', 'LpQ', 'PpQ', 'PQ', 'Q0Q', 'Q1Q',
    'Q2Q', 'Q3Q', 'Q4Q', 'Q5Q', 'raQ', 'rbQ', 'rgeom', 'rrmax', 'rrmin',
    'rzmax', 'rzmin', 'signeo', 'SlQ', 'TQ', 'TTpQ', 'VpQ', 'VQ', 'zgeom',
    'zrmax', 'zrmin', 'zzmax', 'zzmin'])
# TODO(b/342136658): having aW here is a bit of a hack due to reset/step LY
# differences, remove when fixed upstream

# These have shape (M, N, D) but D might be missing
MATRIX_3D_WITH_DOMAIN_KEYS: Final[frozenset[str]] = frozenset(['rq', 'zq'])


# These modify the default behaviour (Scalars -> ())
# Keys added here will have 1 dimension instead of being floats.
SCALAR_KEYS_AS_VECTORS: Final[frozenset[str]] = frozenset([
    'rBt',  # Used in observations, it is convenient to have as a vector.
])
# Keys added here will not be squeezed eg scalars will have shape [1, 1]
# vectors will be either [N, 1] or [1, N]
# this area is to stop keys added above from being processed.
# TODO(b/335613929): Deal with these special cases after next meq upgrade.
FIELD_KEYS_TO_IGNORE: Final[frozenset[str]] = frozenset(['dz', 'dzg'])

TO_SCALAR: Final[frozenset[str]] = SCALAR_KEYS - (
    FIELD_KEYS_TO_IGNORE | SCALAR_KEYS_AS_VECTORS
)
TO_VECTOR: Final[frozenset[str]] = (
    VECTOR_KEYS | SCALAR_KEYS_AS_VECTORS
) - FIELD_KEYS_TO_IGNORE


def maybe_raise_shape_error(
    key: str, goal: int, value: np.ndarray, new_value: np.ndarray
) -> None:
  """Check the new value has expected number of dimensions or raise error."""
  if np.ndim(new_value) != goal:
    err_msg = (
        f'{key} should be {goal} dimensions got {np.ndim(new_value)} with'
        f' shape {new_value.shape} (was {value.shape}) before shaping.'
    )
    raise ValueError(err_msg)


def process_ly_entry(key_path: Any, value: Any) -> Any:
  """Correct shape of LY field from an single timeslice LY."""
  if not isinstance(value, np.ndarray):
    return value

  # NOTE: Someone may one day stumble on a problem here where the last element
  # of the key path doesn't have a name attribute. Because I know this should
  # be an oct2py struct the last key must be a GetAttrKey but if an LY is
  # constructed thats a Dict for instance this might become a GetDictKey.
  # Protecting against that now doesn't seem worth the effort but if you future
  # reader hit it, read
  # https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html#key-paths
  key = key_path[-1].name

  if key in TO_SCALAR:
    new_value = value.squeeze()
    maybe_raise_shape_error(key, 0, value, new_value)
    return new_value
  elif key in TO_VECTOR:
    new_value = np.atleast_1d(np.squeeze(value))
    maybe_raise_shape_error(key, 1, value, new_value)
    return new_value
  elif key in MATRIX_WITH_DOMAIN_KEYS:
    if np.ndim(value) == 1:
      new_value = value[:, None]
    else:
      new_value = value
    maybe_raise_shape_error(key, 2, value, new_value)
    return new_value
  elif key in MATRIX_3D_WITH_DOMAIN_KEYS:
    if np.ndim(value) == 2:
      new_value = value[:, :, None]
    else:
      new_value = value
    maybe_raise_shape_error(key, 3, value, new_value)
    return new_value
  else:
    return value


def process_ly_entry_with_time(key_path: Any, value: Any) -> Any:
  """Correct shape of LY field from an LY with time dimension on each field."""
  if not isinstance(value, np.ndarray):
    return value

  # NOTE: Someone may one day stumble on a problem here where the last element
  # of the key path doesn't have a name attribute. Because I know this should
  # be an oct2py struct the last key must be a GetAttrKey but if an LY is
  # constructed thats a Dict for instance this might become a GetDictKey.
  # Protecting against that now doesn't seem worth the effort but if you future
  # reader hit it, read
  # https://jax.readthedocs.io/en/latest/jax-101/05.1-pytrees.html#key-paths
  key = key_path[-1].name

  # When the time dimension is included scalars -> vectors
  # and vectors should be at least 2 dimensional.
  if key in TO_SCALAR:
    # When squeezing a Float with time (1, t) should be the shape
    # we just need to squeeze to (t) but protect against t being singleton.
    new_value = np.atleast_1d(np.squeeze(value))
    maybe_raise_shape_error(key, 1, value, new_value)
    return new_value
  elif key in TO_VECTOR:
    # When you have a Vector with time we should have shape (N,t) or (1, N, t).
    # In case 1, We would want to leave it untouched even if N or t was 1 but in
    # case 2 we want to squeeze out the first dimension. Since we want the time
    # dimension to remain, we can squeeze all but the last dimension then re-add
    # a dimension before the time if needed with np.at_least_2d. If N is not 1
    # it won't do anything, but if it was, then we re-add a singleton.
    squeeze_dims = tuple([i for i, s in enumerate(value.shape[:-1]) if s == 1])
    new_value = np.atleast_2d(np.squeeze(value, axis=squeeze_dims))
    maybe_raise_shape_error(key, 2, value, new_value)
    return new_value
  elif key in MATRIX_WITH_DOMAIN_KEYS:
    # Matricies with domain with time are 3 dimensional, time should always
    # be the last dimension. So here we add the squeezed out domain dimension in
    # the right place if it was lost.
    if np.ndim(value) == 2:
      new_value = value[:, None, :]
    else:
      new_value = value
    maybe_raise_shape_error(key, 3, value, new_value)
    return new_value
  elif key in MATRIX_3D_WITH_DOMAIN_KEYS:
    if np.ndim(value) == 3:
      new_value = value[:, :, None, :]
    else:
      new_value = value
    maybe_raise_shape_error(key, 4, value, new_value)
    return new_value
  else:
    return value


def check_ly_entry_shape(key_path: Any, value: Any) -> bool:
  """Check that a timeslice LY field has the expected number of dimensions."""
  key = key_path[-1].name
  if key in FIELD_KEYS_TO_IGNORE:
    return True
  elif key in TO_SCALAR:
    return np.ndim(value) == 0
  elif key in TO_VECTOR:
    return np.ndim(value) == 1
  elif key in MATRIX_WITH_DOMAIN_KEYS:
    return np.ndim(value) == 2
  elif key in MATRIX_3D_WITH_DOMAIN_KEYS:
    return np.ndim(value) == 3
  elif not isinstance(value, np.ndarray):
    return True
  else:
    return True


def check_ly_entry_shape_with_time(key_path: Any, value: Any) -> bool:
  """Check that a multi timestep LY field has the expected number of dimensions."""
  key = key_path[-1].name
  if key in FIELD_KEYS_TO_IGNORE:
    return True
  elif key in TO_SCALAR:
    return np.ndim(value) == 1
  elif key in TO_VECTOR:
    return np.ndim(value) == 2
  elif key in MATRIX_WITH_DOMAIN_KEYS:
    return np.ndim(value) == 3
  elif key in MATRIX_3D_WITH_DOMAIN_KEYS:
    return np.ndim(value) == 4
  elif not isinstance(value, np.ndarray):
    return True
  else:
    return True


def process_ly(ly: oct2py.Struct, with_time: bool = False) -> oct2py.Struct:
  """Conditionally squeeze the fields of ly so they are of standard type."""
  # TODO(b/342136658) remove this when aW is a correctly shaped array
  ly.aW = np.asarray(ly.aW)
  if with_time:
    return jax.tree_util.tree_map_with_path(process_ly_entry_with_time, ly)
  else:
    return jax.tree_util.tree_map_with_path(process_ly_entry, ly)


def check_ly_shape(ly: oct2py.Struct, with_time: bool = False) -> bool:
  """Checks if ly shapes conform to expectations post squeezing."""
  if with_time:
    correct_shapes = jax.tree_util.tree_map_with_path(
        check_ly_entry_shape_with_time, ly)
  else:
    correct_shapes = jax.tree_util.tree_map_with_path(
        check_ly_entry_shape, ly)
  return jax.tree.all(correct_shapes)
