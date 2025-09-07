import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="crimepy",
    version="0.0.1",
    author="Andrew Wheeler",
    author_email="andrew.wheeler@crimede-coder.com",
    description="Crime analysis in python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/apwheele/crimepy",
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
        'scipy>1.11',
        'geopandas',
        'folium',
        'requests',
        'jupyter',
        'importlib_resources',
        'geopandas',
        'shapely',
        'matplotlib-scalebar',
        'matplotlib-map-utils',
        'osmnx',
        'beautifulsoup4',
        'folium',
        'networkx',
        'contextily',
        'ipycytoscape',
        'pulp',
        'highspy',
        'libpysal',
        'openpyxl',
        'tabulate',
        'TableauScraper'
    ],
    tests_require=[
        'pytest',
    ],
    package_data={
        'crimepy': ['*.png', '*.csv.zip'],
    },
)