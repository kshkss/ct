from setuptools import setup, find_packages, Extension

setup(
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data = True,
    package_data = {'': ['libct.so']},
    scripts=['scripts/mean', 'scripts/recon'],
    python_requires='>=3.6, <4',
    install_requires=['numpy>=1.17'],
    )

