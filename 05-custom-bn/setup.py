from setuptools import find_packages, setup

setup(
    name="python-custom-ops-bn",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
    description="Custom BatchNorm operator implementation using PyTorch tensor operations",
)
