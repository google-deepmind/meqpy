# Copyright 2024 The meqpy Authors.
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

"""Utilities for running Octave."""

from collections.abc import Mapping, Sequence
from importlib import resources
import itertools
import logging as std_logging
import os
from typing import Any

from absl import logging
import numpy as np
import oct2py


_PATHS = (
    'meq',
    'genlib',
    'octave_optim/inst',
    'octave_control',
)


def _add_octave_paths(
    octave: oct2py.Oct2Py, paths: Sequence[str], root_dir: str | None = None
) -> oct2py.Oct2Py:
  """Import paths for octave."""
  if root_dir is None:
    root_dir = os.environ['MAT_ROOT']

  for path in paths:
    logging.info('adding %s to octave paths', path)
    octave.addpath(octave.genpath(os.path.join(root_dir, path)))

  logging.info('adding meqpy.matlab to octave paths')
  with resources.as_file(resources.files('meqpy.matlab')) as meq_matlab_path:
    logging.info('adding %s to octave paths', meq_matlab_path)
    octave.addpath(octave.genpath(str(meq_matlab_path.absolute())))

  return octave


def create_meq_oct2py_instance(
    octave_paths: Sequence[str] = (),
    logger: std_logging.Logger = logging.get_absl_logger(),
    root_dir: str | None = None,
) -> oct2py.Oct2Py:
  """Instantiate oct2py and import MEQ paths."""
  octave = oct2py.Oct2Py(logger=logger, keep_matlab_shapes=True)
  octave = _add_octave_paths(
      octave, tuple(itertools.chain(_PATHS, octave_paths)), root_dir)
  return octave


def push_to_struct(
    octave: oct2py.Oct2Py,
    struct_name: str,
    data: Mapping[str, Any] | None,
    safe: bool = True) -> None:
  """Assign the values from a dict of data to the fields of an octave struct.

  Args:
    octave: the octave instance to use.
    struct_name: the name of the struct to assign to.
    data: a dict mapping field names to values to assign.
    safe: if True, check that all the fields exist before assigning.

  Raises:
    ValueError: if a field name in `data` is not a field of the struct.
  """
  if not data:
    # If data is empty, we don't need to push anything.
    return

  # Run through the data to add, creating a list of temporary variable names to
  # assign to, a list of the values to assign and a string of commands to
  # execute in octave. We can then group the work into three commands: check
  # that the fields exists, push the values to temporaries and then assign to
  # the struct fields and clear the temporaries.
  var_names = []
  var_data = []
  octave_cmd = ''
  for name, value in data.items():
    var_names.append(name)
    var_data.append(value)
    # Builds the command which is applied at the end to assign the temp values
    # to the struct fields. Done here to avoid reiterating over the data.
    octave_cmd += f'{struct_name}.{name} = {name}_temp; clear {name}_temp;\n'

  # Adds an explicit isfield test on all the field names to avoid silent
  # failures, setting a field that doesn't exist won't error but the user is
  # likely to have made a mistake.
  var_names_octave_compatible = set(var_names)
  if safe:
    is_field = octave.eval(
        f'isfield({struct_name}, {var_names_octave_compatible});'
    )
    # Reformat the returned is_field, if setting a single field then it's a
    # scalar, if a list then octave returns a nested list.
    is_field = (
        np.asarray([is_field])
        if isinstance(is_field, int)
        else np.squeeze(is_field)
    )
    if is_field.all():
      octave.push([f'{name}_temp' for name in var_names], var_data)
      octave.eval(octave_cmd)
    else:
      missing_fields = [
          name
          for name, exists in zip(var_names_octave_compatible, is_field)
          if not exists
      ]
      raise ValueError(
          f'Struct {struct_name} is missing fields: {missing_fields}'
      )
  else:
    octave.push([f'{name}_temp' for name in var_names], var_data)
    octave.eval(octave_cmd)


def pull_from_struct(
    octave: oct2py.Oct2Py,
    struct_name: str,
    param_name: str) -> Any:
  """Pulls a value from an octave struct.

  NOTE: if the value is a numpy array it will be squeezed to remove the
  extra dimension added by octave

  Args:
    octave: the octave instance to use.
    struct_name: the name of the struct to pull from.
    param_name: the name of the field to pull.

  Returns:
    The value of the field.
  """
  return octave.eval(f'{struct_name}.{param_name};', nout=1)


def maybe_remove_from_struct(
    octave: oct2py.Oct2Py,
    struct_name: str,
    param_names: Sequence[str]) -> None:
  """Removes a sequence of values from an octave struct if they exist.

  Args:
    octave: the octave instance to use.
    struct_name: the name of the struct to remove from.
    param_names: the name of the fields to remove.

  Returns:
    The value of the field.
  """
  if not param_names:
    return
  cmd = ''
  for var in param_names:
    cmd += (
        f"if isfield({struct_name}, '{var}'); {struct_name} ="
        f" rmfield({struct_name}, '{var}'); end;\n"
    )
  octave.eval(cmd)


def meq_params_dict_to_str(
    meq_params: dict[str, Any]) -> str:
  """
  Converts a dictionary of MEQ parameters into a formatted string representation, 
  for use in e.g. FGE/FBT calls. 
  
  Args:
    meq_params (dict[str, Any]): A dictionary of MEQ parameters to be converted.
  Returns:
    str: A formatted string representation of the input dictionary.
  Raises:
    ValueError: If a value in the dictionary is of an unsupported type.
  Example:
    meq_params = {
        "cde": "OhmTor_rigid_0D",
        "anajac": True,
        "debug": 2,
        "agcon": ["Wk", "qA"],
    }
    result = meq_params_dict_to_str(meq_params)
    print(result)  # Output: "'cde', 'OhmTor_rigid_0D', 'anajac', 1, 'debug', 2, 'agcon', {'Wk','qA'}"
  """
    

    s = ''
    for k, v in meq_params.items():

        # Parse string entries. Example: {"cde": "OhmTor_rigid_0D"} --> "'cde', 'OhmTor_rigid_0D'"
        if isinstance(v, str):
            s += f"'{k}', '{v}', "      

        # Parse bool entries. Example: {"anajac": True} --> "'anajac', 1"
        elif isinstance(v, bool):       
            s += f"'{k}', {int(v)}, "   

        # Parse numeric entries. Example: {"debug": 2} --> "'debug', 2"
        elif isinstance(v, (int, float, complex)):
            s += f"'{k}', {v}, "        
        
        # Parse list of strings. Example: {"agcon": ["Wk", "qA"]} --> "'agcon', {'Wk','qA'}"
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):        
            cell_array_str = "{" + ",".join([f"'{i}'" for i in v]) + "}"   
            s += f"'{k}', {cell_array_str}, "      
        else:
          raise ValueError(
              f"Unsupported type for key '{k}': type({v}) = {type(v)}. Supported types are: "
              "str, bool, int, float, complex, list of str."
          )          
    s = s[:-2]  # Remove the last comma and space
    
    return s