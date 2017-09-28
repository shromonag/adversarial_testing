from setuptools import setup, find_packages

setup(name='active_testing',
      author='Shromona Ghosh',
      author_email='shromona.ghosh@berkeley.edu',
      install_requires=[
          'numpy',
          'GPy',
          'GPyOpt',
          'scipy'
      ],
      packages=find_packages(),
)
