from importlib.metadata import PackageNotFoundError

from methodkit.utils.conditional import package_installed

if not package_installed("torch"):
    raise PackageNotFoundError("torch is not installed. Please install torch to use this module.")


from .random_start import *  # noqa: F403
