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
