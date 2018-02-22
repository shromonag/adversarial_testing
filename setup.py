from setuptools import setup, find_packages

setup(name='adversarial_testing',
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
