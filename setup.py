"""
setup.py
Set up DBGym reposity.
Tears of the era.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name="dbgym",
    version="0.1.2",
    author="DBGym Team",
    author_email="dbgym@googlegroups.com",
    description="DBGym: deep learning platform for databases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JiaxuanYou/DBGym",
    packages=find_packages(exclude=['docs']),
    install_requires=[
        'numpy', 'pandas', 'scikit_learn', 'tensorboard', 'torch',
        'torch_geometric', 'tqdm', 'xgboost', 'yacs', 'requests'
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
