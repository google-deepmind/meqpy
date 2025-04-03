from absl.testing import absltest
from meqpy import meqpy_fbt_test_mixin
from meqpy import meqpy_fge_test_mixin
from meqpy import meqpy_impl


class AnamakTestMixin(meqpy_fge_test_mixin.MeqPyFGETestMixin,
                      meqpy_fbt_test_mixin.MeqPyFBTestMixin):

  def get_meq_fge_instance(self) -> meqpy_impl.MeqPy:
    return meqpy_fge_test_mixin._get_fge_instance(
        tokamak=self._get_tokamak(), shot=self._get_shot_number(),
        time=self._get_time(), source=self._get_source(),
        cde=None)

  def get_meq_fbt_instance(self) -> meqpy_impl.MeqPy:
    return meqpy_fbt_test_mixin._get_fbt_instance(
        tokamak=self._get_tokamak(), shot=self._get_shot_number(),
        time=self._get_time(), source=self._get_source(),
        cde=None)

  def _get_time(self) -> float:
    return 0.0

  def _get_source(self) -> meqpy_impl.MeqSource:
    return meqpy_impl.MeqSource.MEQ_DIRECT

  def _get_tokamak(self) -> str:
    return 'ana'


class AnamakOneTest(absltest.TestCase, AnamakTestMixin):

  def _get_shot_number(self) -> int:
    return 1


class AnamakEightTwoTest(absltest.TestCase, AnamakTestMixin):

  def _get_shot_number(self) -> int:
    return 82

if __name__ == '__main__':
  absltest.main()
