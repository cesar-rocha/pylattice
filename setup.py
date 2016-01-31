from setuptools import setup, Extension

VERSION='0.1.4'

DISTNAME='pyqg'
URL='http://github.com/pyqg/pyqg'
# how can we make download_url automatically get the right version?
DOWNLOAD_URL='https://github.com/pyqg/pyqg/tarball/v%s' % VERSION
AUTHOR='pyqg team'
AUTHOR_EMAIL='pyqg-dev@googlegroups.com'
LICENSE='MIT'

DESCRIPTION='python lattice models of advection-diffusion'
LONG_DESCRIPTION="""
"""

CLASSIFIERS = [
    'Development Status :: 1 - Beta',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2.7',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Physics',
    'Topic :: Scientific/Engineering :: Atmospheric Science'
]


def readme():
    with open('README.md') as f:
        return f.read()

setup(name='pylattice',
      version='0.1d',
      description='Pythonian lattice model of advection=diffusion\
              analysis',
      url='http://github.com/crocha700/pylattice',
      author='Cesar B Rocha',
      author_email='crocha@ucsd.edu',
      license='MIT',
      packages=['pylattice'],
      install_requires=[
          'numpy',
      ],
      test_suite = 'nose.collector',
      zip_safe=False)
