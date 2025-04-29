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
import pathlib
import tempfile

from absl.testing import absltest


from meqpy import file_utils


class FileHelpersTest(absltest.TestCase):

  def test_simple(self):
    with tempfile.TemporaryDirectory() as temp_dir:
      path = pathlib.Path(temp_dir)
      input_file = path / "input.json"
      data = {"a": 1, "b": 2}
      with open(input_file, "w") as f:
        json.dump(data, f)
      output_file = path / "output.json"
      file_utils.safe_copy(pathlib.Path(input_file), pathlib.Path(output_file))
      with open(output_file, "r") as f:
        self.assertEqual(json.load(f), data)


if __name__ == "__main__":
  absltest.main()
