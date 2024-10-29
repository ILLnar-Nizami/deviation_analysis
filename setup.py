import os
from setuptools import setup, find_packages

with open(os.path.join("docs", "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="deviation-analysis",
    version="0.2.0",
    description="Analysis tool for floor vs ceiling corner deviations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "requests>=2.26.0",
        "line_profiler>=3.5.1",
        "memory_profiler>=0.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "pre-commit>=2.15.0",
            "pdoc3>=0.9.2",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
        ],
        "profile": [
            "memory-profiler>=0.60.0",
            "line-profiler>=3.5.1",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "deviation-analysis=src.main:main",
        ],
    },
)
