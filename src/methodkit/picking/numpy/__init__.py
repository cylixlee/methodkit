from importlib.metadata import PackageNotFoundError

from methodkit.utils.conditional import package_installed

if not package_installed("numpy"):
    raise PackageNotFoundError("numpy is required for this module")


from .roulette_wheel import *  # noqa: F403
