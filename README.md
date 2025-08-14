# crimpy -- a python package for crime analysis

A python package to help conduct different various types of crime analysis. Mostly a collection of various functions I have used over the years.

Andrew Wheeler
[Crime De-Coder LLC](https://crimede-coder.com/)

![](/crimepy/CDCWLineRec.PNG)

## Installation

To install this package from GitHub, you can use pip:

```bash
pip install git+https://github.com/apwheele/crimepy.git
```

For now, I would suggest installing editable, since it is in a very early stage. E.g.

```bash
git clone https://github.com/apwheele/crimepy.git
cd ./crimepy
pip install -e .
```

And then just periodically do a `git pull` to get the most recent version.

## Examples

See the notebooks folder for example analyses:

 - [Aoristic analysis](./notebooks/AoristicAnalysis.ipynb)
 - [DBScan hotspots](./notebooks/DBScanHotspots.ipynb)
 - [Prioritizing Call Ins via Dominant Sets](./notebooks/DominantSetNetwork.ipynb)
 - [Time Series Charts](./notebooks/TimeSeriescharts.ipynb)

## ToDo

 - Funnel charts
 - Example Folium helpers
 - SPPT
 - WDD
 - e-test
 - Patrol district
 - synthetic control
 - tests
 - github actions
 - network spillovers
 - network experiment
 - ?survey duplicates?
 - nearby chains

## Getting Started with Python

Don't know where to get started? Check out my introductory book on python for crime analysts, [Data Science for Crime Analysis with Python](https://crimede-coder.com/blogposts/2024/PythonDataScience)

![](https://crimede-coder.com/images/CoverPage.png)

## References

 - hot spots Dallas
 - SPPT paper
 - tables and graphs
 - monitoring homicide
 - gang network call-ins
 - ??????