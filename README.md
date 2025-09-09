# crimepy -- a python package for crime analysis

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
 - [Time Series Charts](./notebooks/TimeSeriesCharts.ipynb)
 - [Patrol Districting with Workload Equality](./notebooks/PatrolDistricts.ipynb)
 - [SPPT Compare Pre to Post Crime](./notebooks/SPPT_Compare_PrePost.ipynb)
 - [Querying public Socrata or ESRI data](./notebooks/QueryData.ipynb)
 - [Example Interactive Maps with Folium](./notebooks/FoliumExample.ipynb)
 - [Weighted Displacement Difference to See Changes over Time](./notebooks/WDD_Examples.ipynb)

## ToDo

Additional examples in Notebooks and documentation (contributors welcome!)

 - Funnel charts
 - nearby chains
 - SPPT crime to police activity example
 - small sample Benford/Day of week functions & example
 - full docstrings for all functions
 - tests
 - github actions for code quality and pytest
 - try patrol districting with googles ORtools, see how fast it runs
 - simpler tests for districting
 - Folium example for lines and graduated circles
 - scan statistic (see [this blog post for R implementation](https://andrewpwheeler.com/2023/10/18/fitting-beta-binomial-in-python-poisson-scan-stat-in-r/))

Other additional methods *I may* add (if you want these, let me know):

 - synthetic control using Lasso ([blog post](https://andrewpwheeler.com/2023/10/04/synthetic-control-in-python-opioid-death-increases-in-oregon-and-washington/))
 - network spillovers optimal assignment ([blog post](https://andrewpwheeler.com/2019/09/27/optimal-treatment-assignment-with-network-spillovers/))
 - network experiment design ([Notebook](https://github.com/apwheele/Blog_Code/blob/master/Python/NetworkExperiment/NetworkExperimentDesign.ipynb))
 - ?survey duplicates? ([Notebook](https://github.com/apwheele/Blog_Code/blob/master/Python/SurveyMatch/DupsSurvey.ipynb))

Suggestions are always welcome -- direct contributions are a better way to get the items you want in the codebase.

## Getting Started with Python

Don't know where to get started? Check out my introductory book on python for crime analysts, [Data Science for Crime Analysis with Python](https://crimede-coder.com/blogposts/2024/PythonDataScience)

![](https://crimede-coder.com/images/CoverPage.png)

## References

 - Wheeler, A. P. (2016). Tables and graphs for monitoring temporal crime trends: Translating theory into practical crime analysis advice. [*International Journal of Police Science & Management*, 18(3), 159-172](https://journals.sagepub.com/doi/abs/10.1177/1461355716642781). [Preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2551472)

 - Wheeler, A. P. (2019). Creating optimal patrol areas using the p-median model. [*Policing: An International Journal*, 42(3), 318-333](https://www.emerald.com/insight/content/doi/10.1108/pijpsm-02-2018-0027/full/html). [Preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3109791)

 - Wheeler, A. P., & Kovandzic, T. V. (2018). Monitoring volatile homicide trends across US cities. [*Homicide Studies*, 22(2), 119-144](https://journals.sagepub.com/doi/abs/10.1177/1088767917740171). [Preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2977556)

 - Wheeler, A. P., McLean, S. J., Becker, K. J., & Worden, R. E. (2019). Choosing representatives to deliver the message in a group violence intervention. [*Justice Evaluation Journal*, 2(2), 93-117](https://www.tandfonline.com/doi/abs/10.1080/24751979.2019.1630661). [Preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2934325)

 - Wheeler, A. P., & Ratcliffe, J. H. (2018). A simple weighted displacement difference test to evaluate place based crime interventions. [*Crime Science*, 7(1), 11](https://link.springer.com/article/10.1186/s40163-018-0085-5). (Crime Science is an open journal)

 - Wheeler, A. P., & Reuter, S. (2021). Redrawing hot spots of crime in Dallas, Texas. [*Police Quarterly*, 24(2), 159-184](https://journals.sagepub.com/doi/abs/10.1177/1098611120957948). [Preprint](https://www.crimrxiv.com/pub/wmelrli9)

 - Wheeler, A. P., Riddell, J. R., & Haberman, C. P. (2021). Breaking the chain: How arrests reduce the probability of near repeat crimes. [*Criminal Justice Review*, 46(2), 236-258](https://journals.sagepub.com/doi/abs/10.1177/0734016821999707). [Preprint](https://osf.io/7tazd/download)

 - Wheeler, A. P., Steenbeek, W., & Andresen, M. A. (2018). Testing for similarity in area‐based spatial patterns: Alternative methods to Andresen's spatial point pattern test. [*Transactions in GIS*, 22(3), 760-774](https://onlinelibrary.wiley.com/doi/abs/10.1111/tgis.12341). [Preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3111822)