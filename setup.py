import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dbgym",
    version="0.1.0",
    author="Jiaxuan You",
    author_email="jiaxuan@cs.stanford.edu",
    description="DBGym: platform for relational database benchmark RDBench",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JiaxuanYou/DBGym",
    packages=setuptools.find_packages(),
    install_requires=[
      'matplotlib',
      'networkx',
      'numpy',
      'pandas',
      'scikit_learn',
      'torch',
      'torch_geometric',
      'xgboost'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=tbf',
)
