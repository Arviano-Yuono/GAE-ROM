from setuptools import setup, find_packages

setup(
    name="gae-rom",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "meshio>=5.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "isort>=5.0",
            "flake8>=4.0",
            "mypy>=0.9",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    author="Muhammad Arviano Yuono",
    author_email="arvianoyuono@gmail.com",
    description="Graph Autoencoder for Reduced Order Modeling of Navier-Stokes Equations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Arviano-Yuono/GAE-ROM",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
) 