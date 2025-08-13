from collections import Counter
from collections.abc import Sequence

from scipy.stats import chisquare  # pyright: ignore[reportUnknownVariableType]

from methodkit.picking.roulette_wheel import RouletteWheelSelector

# random chosen parameters
_CANDIDATE_COUNT = 5
_SELECT_TIMES = 10000


def test_roulette_wheel() -> None:
    """
    Test whether the roulette wheel works correctly.

    In this case, we do not update the fitness values, and the chosen ones should conform to the uniform
    distribution. We use scipy to verify the result.
    """
    selector = RouletteWheelSelector([None] * _CANDIDATE_COUNT)
    values: list[int] = []
    for _ in range(_SELECT_TIMES):
        index, _ = selector.select_indexed()
        values.append(index)
    assert _is_uniform_distribution(values), "the chosen values are not uniformly distributed"


def _is_uniform_distribution(values: Sequence[int]) -> bool:
    """
    Tests whether the given values are uniformly distributed.

    Args:
        values: The values to test.
    """
    observation = list(Counter(values).values())
    expectation = [len(values) / len(observation)] * len(observation)
    result = chisquare(observation, expectation)
    return result.pvalue.item() > 0.05
