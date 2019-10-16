import numpy as np
import matplotlib.pyplot as plt


def plot_bar_chart(chartname, barvalues, barnames=[], barcolors=[], ylabel=''):
	ind = np.arange(len(barvalues))  # the x locations for the groups
	width = 0.35  # the width of the bars

	fig, ax = plt.subplots()
	ax.bar(ind, barvalues, width,color=barcolors[0])

	# Add some text for labels, title and custom x-axis tick labels, etc.
	ax.set_ylabel(ylabel)
	ax.set_title(chartname)
	ax.set_xticks(ind)
	ax.set_xticklabels(barnames)
	return ax


def plot_pie_chart(chartname, labels, values):
	# Pie chart, where the slices will be ordered and plotted counter-clockwise:
	labels = labels
	sizes = values
	
	fig1, ax1 = plt.subplots()
	ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
	        shadow=True, startangle=90)
	ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.