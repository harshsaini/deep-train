import os

import setuptools


PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(PACKAGE_ROOT, 'README.md')) as f:
    README = f.read()

with open(os.path.join(PACKAGE_ROOT, 'requirements.txt')) as f:
    REQUIREMENTS = [r.strip() for r in f.readlines()]

with open(os.path.join(PACKAGE_ROOT, 'VERSION')) as f:
    VERSION = f.read().strip()

setuptools.setup(
    name='deep_train',
    version=VERSION,
    long_description=README,
    packages=setuptools.find_packages(
        exclude=('tests', 'tests.*', 'examples', 'examples.*'),
    ),
    python_requires='>= 3.7',
    install_requires=REQUIREMENTS,
    author='harshsaini',
    author_email='harshsaini90@gmail.com',
    url='https://github.com/harshsaini/deep-train/tree/master/',
    platforms='Posix; MacOS X; Windows',
    include_package_data=True,
    zip_safe=True,
    license='MIT License',
)
