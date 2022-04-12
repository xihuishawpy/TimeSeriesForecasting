from pandas import Series
from pandas import DataFrame
from pandas import TimeGrouper
from matplotlib import pyplot
series = Series.from_csv('dataset.csv')
groups = series['1964':'1970'].groupby(TimeGrouper('A'))
years = DataFrame()
pyplot.figure()
n_groups = len(groups)
for i, (name, group) in enumerate(groups, start=1):
    pyplot.subplot((n_groups*100) + 10 + i)
    pyplot.plot(group)
pyplot.show()
