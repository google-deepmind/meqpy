"""Tools for moving files around."""

import pathlib
import random
import string

import shutil
import os


def safe_copy(from_path: pathlib.PurePath, to_path: pathlib.PurePath,
              ) -> None:
  """Copies the file to the target directory and then does an atomic rename.

  Because renames are atomic, this avoids the problem of two processes
  attempting to copy to the same filename simultaneously, which causes
  poisoned filehandles.

  If the target path is /foo/bar.baz, the intermediate path is of the form
  /foo/bar-abcdefghi.baz.

  Args:
    from_path: path to copy from.
    to_path: path to copy to.
    immutable: Whether the file should be immutable. Setting to true allows
      replacement but no appending. It should also load slightly faster.
  """
  if not os.path.exists(from_path):
    return
  if not os.path.exists(to_path.parent):
    os.makedirs(to_path.parent)
  random_addition = ''.join(random.sample(string.ascii_letters, 9))
  new_name = f'{to_path.stem}-{random_addition}{to_path.suffix}'
  tmp_path = to_path.with_name(new_name)
  shutil.copyfile(from_path, tmp_path)
  os.rename(tmp_path, to_path)
