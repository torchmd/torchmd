from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("torchmd")
except PackageNotFoundError:
    # package is not installed
    pass
