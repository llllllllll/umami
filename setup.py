from setuptools import setup, find_packages
import sys

long_description = ''

if 'upload' in sys.argv:
    with open('README.rst') as f:
        long_description = f.read()

setup(
    name='umami',
    version='0.3.0',
    description='Bayesian optimization of machine learning models.',
    author='Jasper Snoek, Hugo Larochelle, Ryan P. Adams, Joe Jevnik',
    author_email='joejev@gmail.com',
    packages=find_packages(),
    long_description=long_description,
    license='GPL-3+',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3+ (GPLv3+)',
        'Natural Language :: English',
        'Programming Language :: Python :: 2 :: Only',
        'Topic :: Scientific/Engineering',
    ],
    url='https://github.com/llllllllll/umami',
    install_requires=[
        'numpy',
        'protobuf',
        'scipy',
        'weave',
    ],
)
