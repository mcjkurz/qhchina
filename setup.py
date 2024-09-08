# setup.py

from setuptools import setup, find_packages

setup(
    name="qhchina",
    version="0.0.2",
    author="Maciej Kurzynski",
    author_email="makurz@gmail.com",
    description="The basic package for Quantitative Humanities · China Lab",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)