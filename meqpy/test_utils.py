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

