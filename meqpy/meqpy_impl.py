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

"""Basic python interface for common tasks using MEQ.

This uses oct2py to run MEQ.

Adds the ability to:
* initialise, run and configure FBT
* initialise FGE either directly or from FBT results
* run fge in an step by step fashion
* save and load various inputs for debugging.
"""

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
import enum
import json
import os
import pathlib
import tempfile
from typing import Any, Final

from absl import logging
from meqpy import file_utils
from meqpy import ly_post_processing
from meqpy import octave_utils
import numpy as np
import oct2py

import shutil
import os


_FGETK_ENV_VARS_TO_SAVE: Final[tuple[str, ...]] = (
    "LPfge",
    "LGfge",
    "State_prev",
    "Actions",
    "LXfge_sequence",
    "num_steps",
)

_FBTT_VARS_TO_SAVE: Final[tuple[str, ...]] = (
    "LPfbt",
    "LGfbt",
    "LXfbt",
)

# These are fields which have a corresponding per domain field, it is the
# per-domain fields, and not these, which should be used to override params.
# NOTE: This list could be missing some fields, especially new fields which
# may be added.
_SINGLET_DOUBLET_FIELDS: Final[frozenset[str]] = frozenset(
    ["Ip", "bp", "bt", "Wk", "li"]
)


class MeqSource(enum.Enum):
  """Source of the MEQ data."""

  MEQ_DIRECT = "meq-direct"  # Use MEQ methods directly to init FGE and FBT


class MeqPy:
  """Class for interfacing MEQ to python.

  This class holds the octave instance which holds and maintains the internal
  meq state. MEQ should driven by the methods on this class rather than
  running direct commands.
  """

  def __init__(self, *, extra_paths: Sequence[str] = (),
               root_dir: str | None = None):
    """Initializes MEQ Oct2py instance with the required paths.

    Args:
      extra_paths: paths to add to the octave instance, it is assumed that the
        paths are relative to the root directory.
      root_dir: the root directory to use for the octave instance, this will be
        appended to the paths.
    """
    logging.info("Creating MEQ Oct2py instance.")
    self._is_open = False
    self._octave = octave_utils.create_meq_oct2py_instance(
        extra_paths, root_dir=root_dir)
    self._is_open = True

    self._env_num_steps = 0

  def __del__(self):
    self.cleanup()

  def cleanup(self) -> None:
    """Cleanup the MEQ Oct2py instance."""
    if self._is_open:
      self._octave.eval("clear all;")
      self._octave.exit()
      self._is_open = False

  def octave_eval(self, cmd: str, log: bool = True, nout: int = 0, **kwargs):
    """Wrapper for octave eval with optional logging.

    Most MEQ functionality should be done through the methods on this class
    rather than direct octave calls. This wrapper is provided for convenience
    when debugging or for advanced use cases.

    Args:
      cmd: the octave command to run.
      log: whether to log the command being run.
      nout: how many outputs to return, if 0 then returns the whole result.
      **kwargs: additional arguments to pass to octave.eval.

    Returns:
      The result of the octave eval in either a tuple or a single object
      depending on the value of nout.
    """
    if log:
      logging.info("Executing octave command: %s", cmd)

    return self._octave.eval(
        cmd, verbose=True, nout=nout, **kwargs
    )

  def init_fge(
      self,
      tokamak: str,
      shot: int,
      time: float,
      source: MeqSource,
      cde: str | None = None,
      default_meq_params: MutableMapping[str, Any] | None = None,
      default_fbt_params: MutableMapping[str, Any] | None = None,
  ) -> None:
    """Initialize FGE creating L and LX structures.

    Note: L and LX are maintained in the octave instance as Lfge and LXfge for
    clarity.

    Args:
      tokamak: the tokamak to configure fge to.
      shot: the shot number of the initial condition to initialise the tokamak
        with.
      time: the time at which to start the simulation at.
      source: where to get the data about the tokamak geometry and initial
        condition.
      cde: the cde to use when initializing fge, if not provided then the
        default will be used.
      default_meq_params: default parameters to override in L.P for fge and fbt.
      default_fbt_params: additional default params to use in fbt.
    """
    default_meq_params = default_meq_params or {}
    default_fbt_params = default_fbt_params or {}
    combined_fbt_params = default_fbt_params | default_meq_params
    self._init_fge(
        tokamak,
        shot,
        time,
        source,
        cde,
        default_meq_params,
        combined_fbt_params,
    )

  def _init_fge(
      self,
      tokamak: str,
      shot: int,
      time: float,
      source: MeqSource,
      cde: str | None = None,
      default_meq_params: MutableMapping[str, Any] | None = None,
      combined_fbt_params: MutableMapping[str, Any] | None = None,
  ) -> None:
    """Initialise FGE creating L and LX structures."""
    del cde  # Unused when using MEQ_DIRECT, used for other sources.
    match source:
      case MeqSource.MEQ_DIRECT:
        self._init_fge_directly(tokamak, shot, default_meq_params)
        self._set_fge_inputs_using_fbt(tokamak, shot, time, combined_fbt_params)
      case _:
        raise ValueError(f"Unsupported source: {source}")

  def init_fbt(
      self,
      tokamak: str,
      shot: int,
      time: float,
      source: MeqSource,
      default_fbt_params: MutableMapping[str, Any] | None = None,
      cde: str | None = None,
  ) -> None:
    """Initialise FBT creating L (Lfbt) and LX (LXfbt) structures."""
    self._init_fbt(
        tokamak,
        shot,
        time,
        source,
        cde,
        default_fbt_params,
    )

  def _init_fbt(
      self,
      tokamak: str,
      shot: int,
      time: float,
      source: MeqSource,
      cde: str | None = None,
      default_fbt_params: MutableMapping[str, Any] | None = None,
  ) -> None:
    """Initialise FBT creating L (Lfbt) and LX (LXfbt) structures."""
    del cde  # Unused when using MEQ_DIRECT
    match source:
      case MeqSource.MEQ_DIRECT:
        self._init_fbt_directly(tokamak, shot, default_fbt_params)
        self._set_fbt_inputs_directly(time)
      case _:
        raise ValueError(f"Unsupported source: {source}")

  def init_fge_from_fbt_result(
      self,
      tokamak: str,
      shot: int,
      time: float,
      source: MeqSource,
      default_meq_params: MutableMapping[str, Any] | None = None,
      cde: str | None = None,
  ):
    """Initialise FGE using the result of a previous FBT run.

    This allows controlling the running of FBT (setting custom parameters/
    inputs) for things like shape override etc and then using the result of this
    run to initialise FGE.

    To run this you should already have run something similar to:
    ```
    instance = meqpy.MeqPy()
    instance.init_fbt(tokamak, shot, time, source)
    # Optionally set fbt parameters
    instance.run_fbt()

    # Then initialise fge using this method
    instance.init_fge_from_fbt_result(tokamak, shot, time, source)
    ```

    Args:
      tokamak: the tokamak to configure fge to.
      shot: the shot number of the initial condition to initialise the tokamak
        with.
      time: the time at which to start the simulation at.
      source: where to get the data about the tokamak geometry and initial
        condition.
      default_meq_params: default parameters to override in L.P for fge.
      cde: Which CDE to use.
    """
    if not self.octave_eval("exist('LYfbt', 'var');"):
      raise ValueError("FBT must have been run for this to be possible.")

    self._init_fge_static(
        tokamak,
        shot,
        time,
        source,
        cde,
        default_meq_params,
    )
    self.set_fge_input_from_fbt_run(time)

  def _init_fge_static(
      self,
      tokamak: str,
      shot: int,
      time: float,
      source: MeqSource,
      default_meq_params: MutableMapping[str, Any] | None = None,
      cde: str | None = None,
  ) -> None:
    """Initialise FGE using the result of a previous FBT run."""
    del cde, time  # Unused when using MEQ_DIRECT
    match source:
      case MeqSource.MEQ_DIRECT:
        self._init_fge_directly(tokamak, shot, default_meq_params)
      case _:
        raise ValueError(f"Unsupported source: {source}")

  def _init_fge_directly(self, tokamak: str, shot: int,
                         default_meq_params: Mapping[str, Any] | None = None):
    """Initialize FGE directly, calculates the precomputed quanties stored in L.

    This only works when the appropriate machine description / parameter
    files are in the path.

    Args:
      tokamak: the tokamak to configure fge to.
      shot: the shot number of the initial condition to initialise the tokamak
        with. Different shots across the same tokamak may have different
        geometry and parameters.
      default_meq_params: default parameters to override in L.P for fge.
    """
    meq_params_str = octave_utils.meq_params_dict_to_str(default_meq_params)
    self.octave_eval(f"Lfge = fge('{tokamak}',{shot},0,{meq_params_str});")

  def _set_fge_inputs_using_fbt(
      self, tokamak: str, shot: int, time: float,
      default_meq_params: Mapping[str, Any] | None = None,
  ) -> None:
    """Initialize FGE initial condition using FBT."""
    self._init_fbt_directly(tokamak, shot, default_meq_params)
    self._set_fbt_inputs_directly(time)
    self.run_fbt()
    self.set_fge_input_from_fbt_run(time)

  def _init_fbt_directly(self, tokamak: str, shot: int,
                         default_meq_params: Mapping[str, Any] | None = None):
    """Initialize FBT directly, calculates the precomputed quanties (L)."""    
    meq_params_str = octave_utils.meq_params_dict_to_str(default_meq_params)
    self.octave_eval(f"Lfbt = fbt('{tokamak}',{shot},0,{meq_params_str});")

  def _set_fbt_inputs_directly(self, time: float):
    self.octave_eval(f"""
        fbtxtok = str2func(['fbtx' Lfbt.P.tok]); % fbttok function handle
        LXfbt = fbtxtok({time},Lfbt); % tokamak specific
        LXfbt = fbtx(Lfbt,LXfbt); % add defaults""")

  def run_fbt(self) -> oct2py.Struct:
    """Runs FBT returning LY if converged, else raising an error."""
    result = self.try_run_fbt()
    if result is None:
      raise ValueError("FBT failed to converge.")
    return result

  def try_run_fbt(self) -> oct2py.Struct | None:
    """Runs FBT returning LY if converged, else None."""
    self.octave_eval("LYfbt = fbtt(Lfbt, LXfbt);")
    if not self.octave_eval("~isempty(LYfbt) && LYfbt.isconverged;"):
      return None
    ly = self._octave.pull("LYfbt")
    return ly_post_processing.process_ly(ly)

  def display_fbt_inputs(self):
    self.octave_eval("fbtxdisp(Lfbt, LXfbt);")

  def set_fge_input_from_fbt_run(self, time: float):
    self.octave_eval("LX0 = meqxconvert(Lfbt,LYfbt,Lfge);")

    # Call fgex to populate a time-sequence of LX for time-dependent FGE sim
    self.octave_eval(f"LXfge = fgex(Lfge.P.tok,{time},Lfge,LX0);", log=True)

  def set_fge_parameters(
      self, *,
      parameters: Mapping[str, Any] | None = None,
      geometry: Mapping[str, Any] | None = None,
  ) -> None:
    """Sets FGE params (L.P) and geometry (L.G) and redoes necessary init.

    Note: parameters should be updated in one go such that consolidation is done
      once to keep the workflow efficient.

    Args:
      parameters: mapping from string which is the parameter name in L.P that we
        want to update and set it to the value.
      geometry: mapping from string with is the parameter name in L.G that we
        want to update and set it to the value.
    """
    octave_utils.push_to_struct(self._octave, "Lfge.P", parameters)
    octave_utils.push_to_struct(self._octave, "Lfge.G", geometry)

    # Run consolidation to regenerate values in L after updating L.P and L.G to
    # maintain consistency.
    self.octave_eval("Lfge = fgec(Lfge.P,Lfge.G);")

  def set_fbt_parameters(
      self,
      *,
      parameters: Mapping[str, Any] | None = None,
      geometry: Mapping[str, Any] | None = None,
  ) -> None:
    """Sets FBT params (L.P) and geometry (L.G) and redoes necessary init."""
    octave_utils.push_to_struct(self._octave, "Lfbt.P", parameters)
    octave_utils.push_to_struct(self._octave, "Lfbt.G", geometry)

    # Run consolidation to regenerate values in L after updating L.P and L.G to
    # maintain consistency.
    self.octave_eval("Lfbt = fbtc(Lfbt.P,Lfbt.G);")

  def set_fbt_inputs(self, inputs: Mapping[str, Any]) -> None:
    """Sets LXfbt values, note use `D` values such as `IpD` when both exist."""
    to_remove = _check_correct_fields_and_get_to_remove(inputs)
    octave_utils.push_to_struct(self._octave, "LXfbt", inputs, safe=False)
    octave_utils.maybe_remove_from_struct(self._octave, "LXfbt", to_remove)
    self.octave_eval("LXfbt = fbtx(Lfbt, LXfbt);")

  def set_fbt_control_point(
      self, control_points: Mapping[str, np.ndarray]
  ) -> None:
    """Clear existing control points and set new values."""
    # Assign new values
    octave_utils.push_to_struct(
        self._octave, "LXfbt", {f"gp{n}": v for n, v in control_points.items()}
    )
    # consolidate
    self.octave_eval("LXfbt = fbtx(Lfbt,LXfbt);")

  def get_from_fge_l(self, param_name: str) -> Any:
    return octave_utils.pull_from_struct(self._octave, "Lfge", param_name)

  def get_fge_parameter(self, param_name: str) -> Any:
    return octave_utils.pull_from_struct(self._octave, "Lfge.P", param_name)

  def get_fge_geometry(self, param_name: str) -> Any:
    return octave_utils.pull_from_struct(self._octave, "Lfge.G", param_name)

  def get_fge_input(self, input_name: str) -> Any:
    return octave_utils.pull_from_struct(self._octave, "LXfge", input_name)

  def get_fbt_parameter(self, param_name: str) -> Any:
    return octave_utils.pull_from_struct(self._octave, "Lfbt.P", param_name)

  def get_fbt_geometry(self, param_name: str) -> Any:
    return octave_utils.pull_from_struct(self._octave, "Lfbt.G", param_name)

  def get_fbt_input(self, input_name: str) -> Any:
    return octave_utils.pull_from_struct(self._octave, "LXfbt", input_name)

  def get_fbt_control_points(self) -> oct2py.Struct:
    """Get all the parameters related to the fbt control points."""
    gps = self.octave_eval(
        """
        f = fieldnames(LXfbt)';
        gps = f(startsWith(f, 'gp'));
    """,
        nout=1,
    )[0]
    gps = [gp.replace("gp", "") for gp in gps]
    cmd = ""
    for gp in gps:
      cmd += f"a.{gp} = LXfbt.gp{gp};\n"
    cmd += "a;"
    result = self.octave_eval(cmd, nout=1)
    self.octave_eval("clear a;")

    return result

  def _run_fget(self) -> None:
    """Test-only, runs time-dependent fge to test convergence."""
    self.octave_eval("LYfge = fget(Lfge,LXfge);", log=True)

  def init_fgetk_environment(
      self,
      control_timestep: float,
      simulator_timestep: float,
      input_overrides: Mapping[str, np.ndarray] | None = None,
  ) -> tuple[oct2py.Struct, str]:
    """Initializes fgetk_environment."""
    self._env_num_steps = control_timestep / simulator_timestep
    if abs(self._env_num_steps - int(self._env_num_steps)) > 1e-12:
      raise ValueError(
          "Control timestep must be a multiple of simulator timestep. Was"
          f" {control_timestep} / {simulator_timestep} = {self._env_num_steps}"
      )
    self._env_num_steps = round(self._env_num_steps)
    self._octave.push(
        ["num_steps", "dt"], [self._env_num_steps, simulator_timestep]
    )
    self.octave_eval("LXfge_working = LXfge;")
    if input_overrides:
      to_remove = _check_correct_fields_and_get_to_remove(input_overrides)
      octave_utils.maybe_remove_from_struct(
          self._octave, "LXfge_working", to_remove)
      self._process_input_overrides("LXfge_working", input_overrides,
                                    stack_inputs=False)
      self.octave_eval(
          "LXfge_working = fgex(Lfge.P.tok,LXfge.t,Lfge,LXfge_working);",
          log=True,
      )
    ly, stop = self.octave_eval(
        """
        LXfge_sequence = meqlpack(repmat(LXfge_working, 1, num_steps));

        % Initialize environment
        [LYfge,Stop,State] = fgetk_environment([],[],Lfge,LXfge_working,dt,'init');
        """,
        nout=2,
    )
    return ly_post_processing.process_ly(ly), str(stop)

  def step_fgetk_environment(
      self, actions: np.ndarray,
      input_overrides: Mapping[str, np.ndarray] | None = None
  ) -> tuple[oct2py.Struct, str]:
    """Performs a control step in the fgetk_environment.

    NOTE: in the case when the control timestep and simulator timestep are
    different this will have the effect of doing multiple steps in the fgetk
    environment.

    Args:
      actions: the actions to be applied in the environment. These should match
        the ordering provided by `get_fgetk_input_names()`.
      input_overrides: a mapping from input name to value to override the
        current value of that input in the environment (LX).

    Returns:
      The result (LY) of the previous timestep. This ensures that it matches the
      reality of the tokamak, we calculate the n+1 actions based on the state
      at the beginning of the n-th timestep (aka as a result of the n-1
      timestep).
    """
    self._process_input_overrides("LXfge_sequence", input_overrides)
    self._octave.push("Actions", actions.reshape(-1, 1))

    # Define LX of an appropriate length and run
    octave_cmd = """
    LXfge_sequence.t = State.LYt.t + State.dt*(1:num_steps);
    State_prev = State;
    [LYfge,Stop,State] = fgetk_environment_multi_step(...
                      State_prev,Actions,Lfge,LXfge_sequence,num_steps);"""
    ly, stop = self.octave_eval(octave_cmd, nout=2)
    return ly_post_processing.process_ly(ly), str(stop)

  def _process_input_overrides(
      self,
      var_name: str,
      input_overrides: Mapping[str, np.ndarray] | None = None,
      stack_inputs: bool = True,
  ) -> None:
    """Process input overrides and set them in the given LXfge struct."""
    if not input_overrides:
      return
    if stack_inputs:
      input_overrides = {
          k: np.stack([v] * self._env_num_steps, axis=-1)
          for k, v in input_overrides.items()
      }
    octave_utils.push_to_struct(
        self._octave, var_name, input_overrides)

  def get_fgetk_circuit_names(self) -> Sequence[str]:
    return self.octave_eval("Lfge.G.dima;", nout=1).squeeze()

  def save_last_fgetk_env_call_state(self, file_path: str | os.PathLike[str]):
    """Saves everything needed to rerun the last environment step.

    NOTE: Lfge.P is saved as LPfge with the function handles converted to
      strings and Lfge.G is saved as LGfge. All other variables are saved as
      they were input to `fgetk_environment_multi_step`.

    Args:
      file_path: where to save the data.
    """
    self._strip_function_handles("Lfge.P", "LPfge")
    self.octave_eval("LGfge = Lfge.G;")
    self.save_to_file(file_path, _FGETK_ENV_VARS_TO_SAVE)

  def _restore_fgetk_env_state(self, file_path: str | os.PathLike[str]):
    """Restores the state saved by `save_last_fgetk_env_call_state`.

    NOTE: This also recreates the function handles of the function converted to
    strings. All variables are restored as they were input to
    `fgetk_environment_multi_step` so this can be rerun as required.

    Args:
      file_path: where to save the data.
    """
    self.load_from_file(file_path, _FGETK_ENV_VARS_TO_SAVE)
    # Restore the function handles
    cmd = """
    % special doublet basis function case
    if strcmp(LPfge.bfct,'bfgenD') | strcmp(LPfge.bfct,'bfdoublet')
      for iD = 1:size(LPfge.bfp,1)
        % bfp also contains function handles
        LPfge.bfp{iD,1} = str2func(LPfge.bfp{iD,1});
      end
    end
    for var = {'bfct','infct', 'eqfct'}
      if isfield(LPfge,var) && ischar(LPfge.(var{:}))
        LPfge.(var{:}) = str2func(LPfge.(var{:}));
      end
    end
    Lfge = fgec(LPfge,LGfge);
    """
    self.octave_eval(cmd)

  def run_fgetk_env_from_loaded_file(
      self, file_path: str | os.PathLike[str]
  ) -> oct2py.Struct:
    """Loads and runs an environment step with the saved values."""
    self._restore_fgetk_env_state(file_path)
    ly = self.octave_eval(
        """LY = fgetk_environment_multi_step(...
                      State_prev,Actions,Lfge,LXfge_sequence,num_steps);""",
        nout=1,
    )
    return ly_post_processing.process_ly(ly)

  def save_run_fbt_inputs(self, file_path: str | os.PathLike[str]):
    """Saves everything needed to rerun an run_fbt call.

    NOTE: Lfbt.P is saved as LPfbt with the function handles converted to
      strings and Lfbt.G is saved as LGfbt.
    Args:
      file_path: where to save the data.
    """
    logging.info("Saving run_fbt inputs to %s", file_path)
    self._strip_function_handles("Lfbt.P", "LPfbt")
    self.octave_eval("LGfbt = Lfbt.G;")

    self.save_to_file(file_path, _FBTT_VARS_TO_SAVE)

  def _restore_fbt_state(self, file_path: str | os.PathLike[str]):
    """Restores the state saved by `save_run_fbt_inputs`.

    Args:
      file_path: where to save the data.
    """
    self.load_from_file(file_path, _FBTT_VARS_TO_SAVE)
    # Restore the function handles
    cmd = """
    % special doublet basis function case
    if strcmp(LPfbt.bfct,'bfgenD') | strcmp(LPfbt.bfct,'bfdoublet')
      for iD = 1:size(LPfbt.bfp,1)
        % bfp also contains function handles
        LPfbt.bfp{iD,1} = str2func(LPfbt.bfp{iD,1});
      end
    end
    for var = {'bfct','infct', 'eqfct'}
      if isfield(LPfbt,var) && ischar(LPfbt.(var{:}))
        LPfbt.(var{:}) = str2func(LPfbt.(var{:}));
      end
    end
    Lfbt = fbtc(LPfbt,LGfbt);
    """
    self.octave_eval(cmd)

  def _strip_function_handles(
      self,
      input_variable: str,
      output_variable: str,
      function_name: str = "string",
  ) -> None:
    self.octave_eval(
        f"{output_variable} ="
        f" strip_function_handles({input_variable},'{function_name}');"
    )

  def save_to_file(
      self,
      file_path: str,
      variables_to_save: Sequence[str],
      fields_as_variables: bool = False,
  ) -> None:
    """Writes a set of variables to the given `.mat` file.

    Args:
      file_path: Path of the output file.
      variables_to_save: Name of the matlab variables to be saved.
      fields_as_variables: If True, will save structure fields from
        `variables_to_save` as individual variables. E.g. `my_var` has fields
        `my_field1` and `my_field2`. If True, the saved structure will not
        have a field `my_var`, but instead directly `my_field1` and `my_field2`.
        Note that in this case, `variables_to_save` is expected to be of
        length 1 if we want to save all fields in the variable. Extra arguments
        in the `variables_to_save` list will be interpreted as an explicit list
        of variable fields to be saved.'
    """
    if not variables_to_save:
      raise ValueError("variables_to_save must not be empty.")
    if fields_as_variables:
      root_var = variables_to_save[0]
      if not self.octave_eval(f"exist('{root_var}', 'var');"):
        raise ValueError(
            f"Variable {root_var} does not exist in octave."
        )
      for field in variables_to_save[1:]:
        if not self.octave_eval(f"isfield({root_var}, '{field}');"):
          raise ValueError(
              f"Variable {root_var} does not have field {field}."
          )
    else:
      for var in variables_to_save:
        if not self.octave_eval(f"exist('{var}', 'var');"):
          raise ValueError(f"Variable {var} does not exist in octave.")
    # NOTE: A temporary file is needed as octave cannot access all non-local
    # files.
    with tempfile.NamedTemporaryFile(
        "wb", suffix=".mat", dir=choose_temporary_directory()
    ) as temporary_file:
      vars_to_save_str = " ".join(variables_to_save)
      struct_str = "-struct" if fields_as_variables else ""
      self.octave_eval(
          f"save {temporary_file.name} -v7 {struct_str} {vars_to_save_str};"
      )
      temporary_file.flush()
      file_utils.safe_copy(
          pathlib.Path(temporary_file.name), pathlib.Path(file_path)
      )

  def load_from_file(
      self, file_path: str, variables_to_load: Sequence[str]
  ) -> None:
    """Loads a `.mat` file and reads the specified variables into octave."""
    # NOTE: A temporary file is needed as octave cannot access all non-local
    # files.
    with tempfile.TemporaryDirectory(dir=choose_temporary_directory()
                                     ) as temporary_dir:
      filename = os.path.join(temporary_dir, "file.mat")
      shutil.copyfile(file_path, filename)
      self.octave_eval(f"loaded_values = load('{filename}');")
    cmd = ""
    for var in variables_to_load:
      cmd += f"{var} = loaded_values.{var};"
    cmd += "clear loaded_values;"
    self.octave_eval(cmd)

  def save_fields_to_json(
      self,
      file_path: str | pathlib.Path,
      structure: str,
      field_names: Sequence[str]) -> None:
    """Saves the specified fields of a matlabstructure to a JSON file."""
    fields = {}
    for field in field_names:
      fields[field] = np.squeeze(octave_utils.pull_from_struct(
          self._octave, structure, field)).tolist()
    with open(file_path, "w") as f:
      json.dump(fields, f)


def _check_correct_fields_and_get_to_remove(
    values: Iterable[str],
) -> Sequence[str]:
  """Validates that we are using the per domain fields."""
  to_remove = []
  for value in values:
    # We only allow setting of per domain fields and rely on others being
    # calculated so that we can use the same code for singlets and doublets.
    if value in _SINGLET_DOUBLET_FIELDS:
      raise ValueError(f"Input {value} is not supported, please use {value}D.")
    elif value.endswith("D") and value[:-1] in _SINGLET_DOUBLET_FIELDS:
      to_remove.append(value[:-1])
  return to_remove


def choose_temporary_directory() -> str | None:
  """Returns which temporary directory to use.

  We prefer to use tmpfs, if available. This allows us to avoid allocating
  dedicated RAM for a local filesystem when we only use it for this short
  period.
  """
  tmpfs_path = os.path.join(os.getcwd(), "tmpfs")
  if os.path.isdir(tmpfs_path):
    logging.info(
        "Chose %s as a temporary directory for octave inputs", tmpfs_path
    )
    return tmpfs_path
  logging.info("Using default tmp directory for octave inputs")
  return None
