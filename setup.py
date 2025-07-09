from setuptools import setup, find_packages

setup(
    name="dwfpy_ux",
    version="0.3.9",
    author="Sean Mendoza",
    author_email="sean.mendoza@mac.com",
    description="Digilent Waveforms API facade with optional UIX",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/snmendoza/dwfpy-ux",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6,<3.13",
    install_requires=[
        "numpy<2.0.0",
        "pandas>=1.3.0",
        "PyQt5>=5.15.0",
        "pyqtgraph>=0.13.0",
        "pglive",
        "ipython>=7.0.0",
        "ipywidgets>=7.0.0",
        "ipykernel>=6.0.0",
    ],
) 