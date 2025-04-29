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

"""FBT mixin for meqpy tests."""
import functools
import os
import tempfile

from meqpy import meqpy_impl
from meqpy import test_utils
import numpy as np


# Disabling lint warnings as it is a test mixin and these don't apply to tests.
# pylint: disable=protected-access, missing-function-docstring


@functools.cache
def _get_fbt_instance(
    tokamak: str,
    shot: int,
    time: float,
    source: meqpy_impl.MeqSource,
    cde: str | None,
) -> meqpy_impl.MeqPy:
  """Fbt equivalent of the above method `_get_fge_instance`."""
  fbt_instance = meqpy_impl.MeqPy()
  # Setting debug=2 for help debugging
  fbt_instance.init_fbt(
      tokamak, shot, time, source, default_fbt_params={'debug': 2}, cde=cde
  )
  return fbt_instance


class MeqPyFBTestMixin(test_utils.TestCaseProtocol):
  """Mixin to run FBT related tests on MeqPy."""

  def get_meq_fbt_instance(self) -> meqpy_impl.MeqPy:
    raise NotImplementedError(
        'get_meq_fbt_instance needs to be implemented by the subclass.'
    )

  def _get_tokamak(self) -> str:
    raise NotImplementedError(
        '_get_tokamak needs to be implemented by the subclass.'
    )

  def _get_shot_number(self) -> int:
    raise NotImplementedError(
        '_get_shot needs to be implemented by the subclass.'
    )

  def _get_time(self) -> float:
    raise NotImplementedError(
        '_get_time needs to be implemented by the subclass.'
    )

  def _get_source(self) -> meqpy_impl.MeqSource:
    raise NotImplementedError(
        '_get_source needs to be implemented by the subclass.'
    )

  def test_save_and_load_fbt_state(self):
    meq_instance = self.get_meq_fbt_instance()
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = os.path.join(temp_dir, 'foo.mat')
      initial_ly = meq_instance.run_fbt()
      meq_instance.save_run_fbt_inputs(temp_file)

      new_instance = meqpy_impl.MeqPy()
      new_instance._restore_fbt_state(temp_file)
      ly = new_instance.run_fbt()

      self.assertSameElements(ly, initial_ly)
      for k in {'Ia', 'FB', 'FA', 'isconverged'}:
        np.testing.assert_array_almost_equal(ly[k], initial_ly[k])

  def test_init_and_run_fbt_instance(self):
    meq_instance = self.get_meq_fbt_instance()
    ly = meq_instance.run_fbt()

    original_gpr = meq_instance.get_fbt_input('gpr')
    target_gpr = original_gpr + 0.01

    meq_instance.set_fbt_inputs({'gpr': target_gpr})
    new_ly = meq_instance.run_fbt()

    np.testing.assert_array_less(ly.rA, new_ly.rA)

  def test_init_and_run_fbt_instance_with_new_control_points(self):
    meq_instance = self.get_meq_fbt_instance()
    ly = meq_instance.run_fbt()

    # get control points and shift them to the right
    gps = meq_instance.get_fbt_control_points()
    gps['r'] = gps['r'] + 0.01

    # run FBT with new control points
    meq_instance.set_fbt_control_point(gps)
    new_ly = meq_instance.run_fbt()

    # Check that plasma has moved to the right
    np.testing.assert_array_less(ly.rA, new_ly.rA)

  def test_init_fge_from_fbt(self):
    meq_instance = self.get_meq_fbt_instance()
    meq_instance.run_fbt()
    meq_instance.init_fge_from_fbt_result(
        tokamak=self._get_tokamak(),
        shot=self._get_shot_number(),
        time=self._get_time(),
        source=self._get_source(),
    )
    meq_instance._run_fget()
