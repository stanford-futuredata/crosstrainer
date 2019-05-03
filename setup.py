from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='crosstrainer',
      version='0.1.2',
      description='CrossTrainer: Practical Domain Adaptation with Loss Reweighting',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/stanford-futuredata/crosstrainer',
      author='Justin Yu-wei Chen',
      author_email='jyc8889@gmail.com',
      license='MIT',
      packages=['crosstrainer'],
      test_suite='nose.collector',
      tests_require=['nose'],
      install_require=[
            'markdown',
            'sklearn',
            'numpy',
            'scipy',
      ],
      )
