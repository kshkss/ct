from setuptools import setup, find_packages, Extension

setup(
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data = True,
    package_data = {'': ['libct.so']},
    scripts=['scripts/z-project', 'scripts/recon', 'scripts/rotation-center'],
    python_requires='>=3.6, <4',
    install_requires=['numpy>=1.17', 'scipy>=1.1.0', 'tifffile>=2019.7.26', 'tqdm'],
    )

