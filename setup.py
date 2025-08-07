import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="crimpy",
    version="0.0.1",
    author="Andrew Wheeler",
    author_email="andrew.wheeler@crimede-coder.com",
    description="Crime analysis in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apwheele/crimpy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'matplotlib',
        'pandas',
        'scikit-learn',
        'geopandas',
        'folium',
        'requests',
        'jupyter',
        'importlib_resources',
    ],
    tests_require=[
        'pytest',
    ],
    package_data={
        'crimpy': ['*.png', '*.csv.zip'],
    },
)