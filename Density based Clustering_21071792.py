# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:20:43 2023

@author: Anjikutty
"""

# Importing the required packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import seaborn as sb


# function to read file and to sort data

def read_csv(file_name):
    """
    This function is used to read the csv file from the directory and to import
    the data for the Density clustering.

    file_name :- the name of the csv file with data.   
    """

    file = pd.read_csv(file_name)
    dataFr = pd.DataFrame(file)
    print(dataFr.isnull().any().any())
    sort_data = dataFr.iloc[0:800, [5, 17, 29, 40]]
    return sort_data

# Function transform and to plot the data


def cluster_plot(title_1, title_2, image_1, image_2):
    """
    'Cluster_plot' function is used to standardize and to normalize the data by
    removing the mean and making the data to be more accurate.

    title_1 :- The title for the Line plot.
    title_2 :- the title for the Cluster plot.
    image_1 :- the destination to which the line plot should be saved.
    image_2 :- the destination to thich the scatter plot should be saved.
    """

    plot_data = read_csv("Sales_Transactions_Dataset_Weekly.csv")

    # standardizing the data
    plot_data = StandardScaler().fit_transform(plot_data)

    # Normalizing the data
    plot_data = normalize(plot_data)
    plot_data = pd.DataFrame(plot_data)

    x = plot_data[0]
    y = plot_data[2]

    # KNN method
    """
    By using KNN method its easy to find the data points which are close to the 
    chosen point. 
    """

    nearest_neigh = NearestNeighbors(n_neighbors=5).fit(plot_data)
    neighbor_dist, neighbor_index = nearest_neigh.kneighbors(plot_data)
    neighbor_dist = np.sort(neighbor_dist, axis=0)
    neighbor_dist = neighbor_dist[:, 4]
    plt.figure(figsize=(5, 3))
    plt.plot(neighbor_dist)
    plt.title(title_1)
    plt.show()
    plt.savefig(image_1, dpi=200)

    # Using DBSCAN Algorithm and doing Cluster plotting
    minPoints = 2*(len(plot_data.axes[1]))
    dbscan = DBSCAN(eps=0.15, min_samples=minPoints).fit(plot_data)

    # scatter plot using seaborn
    plt.figure()
    plt.title(title_2)
    p = sb.scatterplot(data=plot_data, x=x, y=y,
                       hue=dbscan.labels_, legend="full", palette="deep")
    sb.move_legend(p, "upper right", bbox_to_anchor=(
        1.10, 1.10), title="CLUSTERS")
    plt.show()
    plt.savefig(image_2, dpi=200)
    return


# Inoking the function by passing the parameters.
cluster_plot('The distance between the neighbor points',
             'Sales Transactions per Week', "line plot.png", "scatter plot.png")
