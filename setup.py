import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="jmawirat_ml_helpers",
    version="1.0.1",
    author="Jonel Mawirat",
    author_email="jmawirat@outlook.com",
    description="Time-series data splitting, feature engineering, and stationarity transformations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jonelmawirat/ml_helpers",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "matplotlib>=3.0.0",
        "statsmodels>=0.12.0"
    ],
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    license="MIT",
    keywords="time-series splitting feature-engineering stationarity machine-learning",
)

