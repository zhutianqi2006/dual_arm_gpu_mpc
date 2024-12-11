import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, 'include')]
sources = glob.glob('*.cu') + glob.glob('*.cpp')

setup(
    name='dq_test',
    version='1.0.0',  # 也考虑添加版本号
    ext_modules=[
        CUDAExtension(
            name='dq_torch',  # import name
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },

    package_data={
        '': ['*.pyi'], 
    },

    packages=[''],
)
