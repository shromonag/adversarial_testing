from setuptools import setup, find_packages

setup(name='active_testing',
      install_requires=[
          'numpy',
          'GPy',
          'GPyOpt',
          'scipy'
      ],
      packages=find_packages(),
)
