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

"""Tools for moving files around."""

import pathlib
import random
import string

import shutil
import os


def safe_copy(from_path: pathlib.PurePath, to_path: pathlib.PurePath) -> None:
  """Copies the file to the target directory and then does an atomic rename.

  Because renames are atomic, this avoids the problem of two processes
  attempting to copy to the same filename simultaneously, which causes
  poisoned filehandles.

  If the target path is /foo/bar.baz, the intermediate path is of the form
  /foo/bar-abcdefghi.baz.

  Args:
    from_path: path to copy from.
    to_path: path to copy to. If the path does not exist, it will be created
      with permissions 0o775.
    immutable: Whether the file should be immutable. Setting to true allows
      replacement but no appending. It should also load slightly faster.
  """
  if not os.path.exists(from_path):
    return
  if not os.path.exists(to_path.parent):
    os.makedirs(to_path.parent, mode=0o775)
  random_addition = ''.join(random.sample(string.ascii_letters, 9))
  new_name = f'{to_path.stem}-{random_addition}{to_path.suffix}'
  tmp_path = to_path.with_name(new_name)
  shutil.copyfile(from_path, tmp_path)
  os.rename(tmp_path, to_path)
