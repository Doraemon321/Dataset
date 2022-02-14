# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 10:41:39 2021

@author: PQD
"""

from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_path = os.path.dirname(os.path.abspath(__file__))
include_path = os.path.join(this_path, 'include')
source_path = os.path.join(this_path, 'src')
sources = glob.glob(os.path.join(source_path, '*.cpp')) + glob.glob(os.path.join(source_path, '*.cu'))

setup(
      name='ext',
      version='0.2',
      ext_modules=[
              CUDAExtension('ext', sources=sources, include_dirs=[include_path]),
              ],
    cmdclass={'build_ext': BuildExtension}
    )