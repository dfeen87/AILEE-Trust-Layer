from setuptools import setup, find_packages

setup(
    name="ailee-trust-layer",
    version="1.4.0",
    description="A deterministic trust and governance layer for AI decision systems",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Don Michael Feeney Jr.",
    license="MIT",
    python_requires=">=3.9",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=[
        # Intentionally minimal.
        # Core AILEE pipeline has no hard external dependencies.
    ],
    extras_require={
        "monitoring": [],
        "serialization": [],
        "replay": [],
        "domains": [],
        "dev": ["pytest", "black", "mypy"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    keywords=[
        "AI trust",
        "responsible AI",
        "AI governance",
        "decision systems",
        "safety-critical AI",
        "AILEE",
    ],
    project_urls={
        "Source": "https://github.com/dfeen87/ailee-trust-layer",
        "Documentation": "https://github.com/dfeen87/ailee-trust-layer",
        "Issues": "https://github.com/dfeen87/ailee-trust-layer/issues",
    },
)
