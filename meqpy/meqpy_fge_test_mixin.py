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

"""FGE mixin for meqpy tests."""
import functools
import os
import tempfile
from typing import Final

from meqpy import meqpy_impl
from meqpy import test_utils
import numpy as np
from oct2py import utils as oct2py_utils

_CONTROL_TIMESTEP: Final[float] = 0.0001
_SIMULATOR_TIMESTEP: Final[float] = 0.00002

# Disabling lint warnings as it is a test mixin and these don't apply to tests.
# pylint: disable=protected-access, missing-function-docstring


@functools.cache
def _get_fge_instance(
    tokamak: str,
    shot: int,
    time: float,
    source: meqpy_impl.MeqSource,
    cde: str | None,
) -> meqpy_impl.MeqPy:
  """Returns a possibly cached initialised meq instance.

  This contains all the expensive setup of the instance creating and
  initialising fge. We set this up to cache the instance and when asking for the
  same instance based on the params use the same issue.

  NOTE: the caching behavior of this means that state is maintained across
  different tests, changes to the meq state will be maintained across tests and
  these tests can occur in any order. Where possible reset the internal state
  after any changes occur.

  Args:
    tokamak: the tokamak to initialise for
    shot: the shot number to setup for
    time: the time at which to start the simulation
    source: how to setup the parameters in the meq instance
    cde: which CDE to use

  Returns a meq_instance with initialized FGE.
  """
  meq_instance = meqpy_impl.MeqPy()
  # Setting debug=2 for help debugging (including for the initialisation)
  meq_instance.init_fge(
      tokamak, shot, time, source, default_meq_params={'debug': 2}, cde=cde
  )
  # Setting anajac,usepreconditioner for speed of fge env
  meq_instance.set_fge_parameters(parameters={'anajac': True,
                                              'usepreconditioner': True})
  return meq_instance


class MeqPyFGETestMixin(test_utils.TestCaseProtocol):
  """Mixin to run FGE related tests on MeqPy."""

  def get_meq_fge_instance(self):
    """Gets an meqpy instance initialized for running FGE."""
    raise NotImplementedError(
        '_get_meq_instance needs to be implemented by the subclass.'
    )

  def _get_time(self):
    raise NotImplementedError(
        '_get_time needs to be implemented by the subclass.'
    )

  def _get_shot_number(self):
    raise NotImplementedError(
        '_get_shot_number needs to be implemented by the subclass.'
    )

  def test_init_and_run(self):
    meq_instance = self.get_meq_fge_instance()

    meq_instance._run_fget()

  def test_get_and_set_fge_parameters(self):
    meq_instance = self.get_meq_fge_instance()

    original_debug_value = meq_instance.get_fge_parameter('debug')
    original_use_preconditioner_value = meq_instance.get_fge_parameter(
        'usepreconditioner')

    target_debug_value = original_debug_value + 1
    target_use_preconditioner_value = not(original_use_preconditioner_value)
    meq_instance.set_fge_parameters(
        parameters={
            'debug': target_debug_value,
            'usepreconditioner': target_use_preconditioner_value})
    self.assertEqual(
        meq_instance.get_fge_parameter('debug'), target_debug_value)
    self.assertEqual(meq_instance.get_fge_parameter('usepreconditioner'),
                     target_use_preconditioner_value)

    # Reset the original value for usage across other tests
    meq_instance.set_fge_parameters(
        parameters={
            'debug': original_debug_value,
            'usepreconditioner': original_use_preconditioner_value})

  def test_get_and_set_fge_geometry(self):
    meq_instance = self.get_meq_fge_instance()

    original_vamax_value = meq_instance.get_fge_geometry('Vamax')
    original_iamax_value = meq_instance.get_fge_geometry('Iamax')

    target_vamax_value = original_vamax_value + 10.
    target_iamax_value = original_iamax_value + 10.
    meq_instance.set_fge_parameters(
        geometry={'Vamax': target_vamax_value,
                  'Iamax': target_iamax_value})
    np.testing.assert_array_equal(
        target_vamax_value, meq_instance.get_fge_geometry('Vamax'))
    np.testing.assert_array_equal(
        target_iamax_value, meq_instance.get_fge_geometry('Iamax'))

    meq_instance.set_fge_parameters(
        geometry={'Vamax': original_vamax_value,
                  'Iamax': original_iamax_value})

  def test_setting_field_which_does_not_exist_raises_error(
      self):
    meq_instance = self.get_meq_fge_instance()

    with self.assertRaisesRegex(
        ValueError, 'foo'):
      meq_instance.set_fge_parameters(
          parameters={'foo': 1, 'debug': 2})

  def test_getting_parameter_which_does_not_exist_raises_error(
      self):
    meq_instance = self.get_meq_fge_instance()

    with self.assertRaisesRegex(
        oct2py_utils.Oct2PyError, 'structure has no member \'foo\''):
      meq_instance.get_fge_parameter('foo')

  def test_getting_geometry_which_does_not_exist_raises_error(
      self):
    meq_instance = self.get_meq_fge_instance()

    with self.assertRaisesRegex(
        oct2py_utils.Oct2PyError, 'structure has no member \'foo\''):
      meq_instance.get_fge_geometry('foo')

  def test_getting_input_which_does_not_exist_raises_error(
      self):
    meq_instance = self.get_meq_fge_instance()

    with self.assertRaisesRegex(
        oct2py_utils.Oct2PyError, 'structure has no member \'foo\''):
      meq_instance.get_fge_input('foo')

  def test_init_and_step_fgetk_env(self):
    meq_instance = self.get_meq_fge_instance()
    original_bp = meq_instance.get_fge_input('bpD')
    _, stop = meq_instance.init_fgetk_environment(
        control_timestep=_CONTROL_TIMESTEP,
        simulator_timestep=_SIMULATOR_TIMESTEP,
        input_overrides={'bpD': original_bp * 1.001})
    self.assertFalse(stop)

    num_actions = len(meq_instance.get_fgetk_circuit_names())
    actions = np.full((num_actions,), 10)

    for i in range(3):
      ly, stop = meq_instance.step_fgetk_environment(actions, {})
      self.assertFalse(stop)
      # The simulator returns the result of last timestep, this is so that it
      # matches reality when we base decisions on the basis of the measurements
      # from taking the previous action.
      self.assertAlmostEqual(ly.t, self._get_time() + i * _CONTROL_TIMESTEP)

    with self.subTest('use_parameter_variation'):
      meq_instance.step_fgetk_environment(actions, {'bpD': original_bp * 1.2})

  def test_init_and_step_fgetk_env_with_delay(self):
    # Introduces action delay and then checks that the actions are delayed by
    # the expected amount.
    meq_instance = self.get_meq_fge_instance()
    num_actions = len(meq_instance.get_fgetk_circuit_names())
    delay_steps = [3] * num_actions

    # Set some delay_steps to uneven values
    delay_steps[0] = 0
    delay_steps[1] = 1
    delay_steps[-2] = 1
    delays = np.asarray(delay_steps, dtype=np.float32) * _CONTROL_TIMESTEP

    meq_instance.set_fge_parameters(
        geometry={
            'Vadelay': delays})
    meq_instance.init_fgetk_environment(
        control_timestep=_CONTROL_TIMESTEP,
        simulator_timestep=_SIMULATOR_TIMESTEP,
        input_overrides={}
    )
    results = []
    first_action = 10.
    first_actions = np.full((num_actions,), first_action)
    second_action = -10.
    second_actions = np.full((num_actions,), second_action)

    for _ in range(5):
      ly, stop = meq_instance.step_fgetk_environment(first_actions)
      self.assertFalse(stop)
      results.append(np.squeeze(ly.Va))

    for _ in range(5):
      ly, stop = meq_instance.step_fgetk_environment(second_actions)
      self.assertFalse(stop)
      results.append(np.squeeze(ly.Va))

    results = np.stack(results).transpose()
    for i, delay in enumerate(delay_steps):
      result = results[i]
      # Check that the first value plus the number of delay steps is not equal
      # to the first action. We add in the first action as we receive the
      # results of the previous step as output so are essentially always delayed
      # by 1 step.
      self.assertFalse(np.any(np.equal(result[:delay + 1], first_action)))
      # Check that the 5 next values match the first action.
      np.testing.assert_array_equal(result[delay + 1: delay + 5 + 1],
                                    first_action)
      # Check that the remaining values match the second action.
      np.testing.assert_array_equal(result[delay + 5 + 1: 10], second_action)

  def test_save_and_load_simple_parameter(self):
    meq_instance = self.get_meq_fge_instance()
    bp = meq_instance.get_fge_input('bpD')

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = os.path.join(temp_dir, 'foo.mat')

      meq_instance.save_to_file(temp_file, ('LXfge',))
      meq_instance.octave_eval('clear LXfge;')
      meq_instance.load_from_file(temp_file, ('LXfge',))

      np.testing.assert_array_equal(meq_instance.get_fge_input('bpD'), bp)

  def test_save_and_load_fgetk_env_state(self):
    meq_instance = self.get_meq_fge_instance()
    meq_instance.init_fgetk_environment(
        control_timestep=_CONTROL_TIMESTEP,
        simulator_timestep=_SIMULATOR_TIMESTEP,
        input_overrides={})
    actions = np.random.random((len(meq_instance.get_fgetk_circuit_names()),))
    initial_ly, _ = meq_instance.step_fgetk_environment(actions)

    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = os.path.join(temp_dir, 'foo.mat')

      meq_instance.save_last_fgetk_env_call_state(temp_file)

      new_instance = meqpy_impl.MeqPy()
      ly = new_instance.run_fgetk_env_from_loaded_file(temp_file)

      self.assertSameElements(ly, initial_ly)
      for k in initial_ly:
        np.testing.assert_array_equal(ly[k], initial_ly[k])
