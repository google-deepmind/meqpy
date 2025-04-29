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

import json
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
from meqpy import meqpy_impl
import oct2py


class MeqpyTest(parameterized.TestCase):

  def test_save_to_json(self):
    meq_instance = meqpy_impl.MeqPy()
    meq_instance._octave.eval('Foo.a = 1;')
    meq_instance._octave.eval('Foo.b = [1.2, 3.4];')
    meq_instance._octave.eval('Foo.c = "bar";')

    with tempfile.TemporaryDirectory() as temp_dir:
      filepath = temp_dir+ '/test.json'
      meq_instance.save_fields_to_json(filepath, 'Foo', ['a', 'b'])
      with open(filepath, 'r') as f:
        loaded = json.load(f)
      self.assertIn('a', loaded)
      self.assertIn('b', loaded)
      self.assertNotIn('c', loaded)
      self.assertEqual(loaded['a'], 1.)
      self.assertSequenceEqual(loaded['b'], [1.2, 3.4])

  def test_save_to_file_ok(self):
    meq_instance = meqpy_impl.MeqPy()
    meq_instance._octave.eval('Foo = 2;')
    meq_instance._octave.eval('Bar = {};')
    meq_instance._octave.eval('Bar.baz = 3;')
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = temp_dir + '/test.mat'
      meq_instance.save_to_file(temp_file, ['Foo', 'Bar'])
      meq_instance._octave.eval('clear all;')
      meq_instance.load_from_file(
          temp_file, variables_to_load=['Foo', 'Bar']
      )
      self.assertEqual(meq_instance._octave.eval('Foo', nout=1), 2)
      self.assertEqual(meq_instance._octave.eval('Bar.baz', nout=1), 3)

  def test_save_to_file_fields_as_variables_ok(self):
    meq_instance = meqpy_impl.MeqPy()
    meq_instance._octave.eval('Foo = {};')
    meq_instance._octave.eval('Foo.bar = 2;')
    meq_instance._octave.eval('Foo.baz = 3;')
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = temp_dir + '/test.mat'
      meq_instance.save_to_file(
          temp_file, ['Foo'], fields_as_variables=True
      )
      meq_instance._octave.eval('clear all;')
      meq_instance.load_from_file(
          temp_file, variables_to_load=['bar', 'baz']
      )
      self.assertEqual(meq_instance._octave.eval('bar', nout=1), 2)
      self.assertEqual(meq_instance._octave.eval('baz', nout=1), 3)

  def test_save_to_file_fields_as_variables_select_fields(self):
    meq_instance = meqpy_impl.MeqPy()
    meq_instance._octave.eval('Foo = {};')
    meq_instance._octave.eval('Foo.bar = 2;')
    meq_instance._octave.eval('Foo.baz = 3;')
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = temp_dir + '/test.mat'
      meq_instance.save_to_file(
          temp_file, ['Foo', 'bar'], fields_as_variables=True
      )
      meq_instance._octave.eval('clear all;')
      meq_instance.load_from_file(temp_file, variables_to_load=['bar'])
      self.assertEqual(meq_instance._octave.eval('bar', nout=1), 2)
      with self.assertRaises(oct2py.utils.Oct2PyError):
        meq_instance.load_from_file(temp_file, variables_to_load=['baz'])

  @parameterized.parameters(
      dict(fields_as_variables=False), dict(fields_as_variables=True)
  )
  def test_save_to_file_empty_variables_to_save(self, fields_as_variables):
    meq_instance = meqpy_impl.MeqPy()
    meq_instance._octave.eval('Foo = {};')
    meq_instance._octave.eval('Bar = {};')
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = temp_dir + '/test.mat'
      with self.assertRaises(ValueError):
        meq_instance.save_to_file(
            temp_file, [], fields_as_variables=fields_as_variables
        )

  def test_save_to_file_missing_variable(self):
    meq_instance = meqpy_impl.MeqPy()
    meq_instance._octave.eval('Foo = {};')
    meq_instance._octave.eval('Bar = {};')
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = temp_dir + '/test.mat'
      with self.assertRaises(ValueError):
        meq_instance.save_to_file(temp_file, ['Foo', 'Bar', 'Baz'])

  def test_save_to_file_missing_root_variable(self):
    meq_instance = meqpy_impl.MeqPy()
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = temp_dir + '/test.mat'
      with self.assertRaises(ValueError):
        meq_instance.save_to_file(
            temp_file, ['Foo'], fields_as_variables=True
        )

  def test_save_to_file_missing_field(self):
    meq_instance = meqpy_impl.MeqPy()
    meq_instance._octave.eval('Foo = {};')
    meq_instance._octave.eval('Foo.bar = 2;')
    with tempfile.TemporaryDirectory() as temp_dir:
      temp_file = temp_dir + '/test.mat'
      with self.assertRaises(ValueError):
        meq_instance.save_to_file(
            temp_file, ['Foo', 'bar', 'baz'], fields_as_variables=True
        )


if __name__ == '__main__':
  absltest.main()
