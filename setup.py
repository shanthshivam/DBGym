"""
setup.py
Set up DBGym reposity.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="dbgym",
    version="0.1.0",
    author="Jiaxuan You",
    author_email="jiaxuan@cs.stanford.edu",
    description="DBGym: platform for relational database benchmark RDBench",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JiaxuanYou/DBGym",
    packages=find_packages(exclude=['docs']),
    install_requires=[
      'matplotlib',
      'networkx',
      'numpy',
      'pandas',
      'scikit_learn',
      'torch',
      'torch_geometric',
      'xgboost',
      'yacs',
      'requests'
    ],
    license='MIT',
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)
