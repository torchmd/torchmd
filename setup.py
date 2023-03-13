import versioneer
from setuptools import setup

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    zip_safe=False,
)
