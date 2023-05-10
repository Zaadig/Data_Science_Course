# python3 setup.py build_ext --inplace

from setuptools import setup
from Cython.Build import cythonize

setup(
    name='The Matrix app',
    ext_modules=cythonize("the_matrix.pyx"),
    zip_safe=False,
)
