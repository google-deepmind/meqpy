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

"""Utils for writing test mixins."""

from typing import Protocol

# pylint:disable=invalid-name


class TestCaseProtocol(Protocol):
  """Protocol which implements the methods we need from googletest.TestCase."""

  def assertEqual(self, a, b):
    ...

  def assertAlmostEqual(self, a, b, places=None):
    ...

  def assertRaisesRegex(self, exception_type, regex):
    ...

  def assertFalse(self, a):
    ...

  def assertSameElements(self, a, b):
    ...

  def subTest(self, name):
    ...

  def skipTest(self, reason: str):
    ...

# pylint:enable=invalid-name

