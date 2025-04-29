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

from absl.testing import absltest

from meqpy import octave_utils
import numpy as np
import oct2py


class OctaveUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.octave = oct2py.Oct2Py()

  def test_push_a_single_value_to_a_struct(self):
    self.octave.eval('a.foo = 1; a.bar = 2;')

    octave_utils.push_to_struct(self.octave, 'a', {'foo': 3})

    self.assertEqual(octave_utils.pull_from_struct(
        self.octave, 'a', 'foo'), 3)

  def test_push_a_collection_of_values_to_a_struct(self):
    self.octave.eval('a.foo = 1; a.bar = 2;')

    octave_utils.push_to_struct(self.octave, 'a', {'foo': 3, 'bar': 4})

    self.assertEqual(octave_utils.pull_from_struct(self.octave, 'a', 'foo'), 3)
    self.assertEqual(octave_utils.pull_from_struct(self.octave, 'a', 'bar'), 4)

  def test_push_a_string_to_a_struct(self):
    self.octave.eval("a.foo = 'test';")

    octave_utils.push_to_struct(self.octave, 'a', {'foo': 'temp'})
    self.assertEqual(
        octave_utils.pull_from_struct(self.octave, 'a', 'foo'), 'temp'
    )

  def test_push_an_array_to_a_struct(self):
    self.octave.eval('a.foo = [1, 2, 3];')

    octave_utils.push_to_struct(self.octave, 'a', {'foo': [4, 5, 6]})
    np.testing.assert_array_equal(
        np.squeeze(octave_utils.pull_from_struct(self.octave, 'a', 'foo')),
        [4, 5, 6]
    )

  def test_push_bool_to_struct(self):
    self.octave.eval('a.foo = true;')

    octave_utils.push_to_struct(self.octave, 'a', {'foo': False})
    self.assertFalse(octave_utils.pull_from_struct(self.octave, 'a', 'foo'))

  def test_pull_and_push_a_cell_to_struct(self):
    self.octave.eval('a.foo = {"1", "2", "3"};')

    a = octave_utils.pull_from_struct(self.octave, 'a', 'foo').tolist()[0]
    a.remove('2')
    octave_utils.push_to_struct(self.octave, 'a', {'foo': a})
    result = octave_utils.pull_from_struct(self.octave, 'a', 'foo').tolist()[0]
    self.assertSequenceEqual(result, ['1', '3'])

  def test_try_to_push_a_value_to_a_missing_field(self):
    self.octave.eval('a.foo = 1;')

    with self.assertRaisesRegex(ValueError, 'bar'):
      octave_utils.push_to_struct(self.octave, 'a', {'foo': 3, 'bar': 2})

  def test_try_to_pull_a_missing_field(self):
    self.octave.eval('a.foo = 1;')
    with self.assertRaisesRegex(oct2py.utils.Oct2PyError, 'bar'):
      octave_utils.pull_from_struct(self.octave, 'a', 'bar')

if __name__ == '__main__':
  absltest.main()
