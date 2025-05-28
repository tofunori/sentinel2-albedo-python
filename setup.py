from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="sentinel2-albedo",
    version="1.0.0",
    author="Claude AI Assistant",
    author_email="",
    description="Python implementation of high spatial resolution albedo retrieval from Sentinel-2 and MODIS BRDF data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tofunori/sentinel2-albedo-python",
    project_urls={
        "Bug Tracker": "https://github.com/tofunori/sentinel2-albedo-python/issues",
        "Documentation": "https://github.com/tofunori/sentinel2-albedo-python/docs",
        "Source Code": "https://github.com/tofunori/sentinel2-albedo-python",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "notebooks": [
            "ipykernel>=6.0.0",
            "jupyter>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "s2-albedo=sentinel2_albedo.cli:main",
        ],
    },
    keywords="sentinel2, modis, albedo, brdf, remote-sensing, satellite, earth-observation",
    include_package_data=True,
    zip_safe=False,
)
