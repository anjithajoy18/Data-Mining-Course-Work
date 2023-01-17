# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 15:20:43 2023

@author: Anjikutty
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize


#function to read file and to sort data
def read_csv(file_name):
    
    file = pd.read_csv(file_name)
    dataFr = pd.DataFrame(file)
    print(dataFr.isnull().any().any())
    sort_data = dataFr.iloc[0:200,[1, 22, 34, 45]]
    return sort_data


def cluster_plot():
    
    #Getting the data loaded from csv file
    plot_data = read_csv("Sales_Transactions_Dataset_Weekly.csv")
    
    #standardizing the data
    plot_data = StandardScaler().fit_transform(plot_data)
    #plot_data = pd.DataFrame(plot_data)
    
    #Normalizing the data
    plot_data = normalize(plot_data)
    plot_data = pd.DataFrame(plot_data)
    
    x = plot_data[0]
    y = plot_data[2]
    

    #KNN method
    nearest_neigh = NearestNeighbors(n_neighbors = 3).fit(plot_data) 
    neighbor_dist, neighbor_index = nearest_neigh.kneighbors(plot_data)
    neighbor_dist = np.sort(neighbor_dist, axis = 0)
    neighbor_dist = neighbor_dist[:, 3]
    print(neighbor_dist)
    plt.figure(figsize = (5,3))
    plt.plot(neighbor_dist)
    plt.savefig("line plot.png")
    
    
    #Cluster plotting
    minPoints = 2*(len(plot_data.axes[1]))
    dbscan = DBSCAN(eps = 0.15, min_samples = minPoints).fit(plot_data)
    set(dbscan.labels_)
    
    plt.figure()
    plt.scatter(plot_data[:, 0],plot_data[:, 1])
    return


#calling the function to plot the distance.
cluster_plot()
