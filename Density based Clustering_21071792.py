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
    sort_data = dataFr.iloc[:,[1, 22, 54, 88]]
    return dataFr, sort_data


def cluster_plot():
    
    #Getting the data loaded from csv file
    data, plot_data = read_csv("Sales_Transactions_Dataset_Weekly.csv")
    
    #standardizing the data
    plot_data = StandardScaler().fit_transform(plot_data)
    
    #Normalizing the data
    plot_data = normalize(plot_data)
    #plot_data = pd.DataFrame(plot_data)
    print(plot_data)
    
    #KNN method
    nearest_neigh = NearestNeighbors(n_neighbors = 3).fit(plot_data)
    neighbor_dist = nearest_neigh.kneighbors(plot_data)
    neighbor_dist = np.sort(neighbor_dist, axis = 0)
    neighbor_dist = neighbor_dist[:, 4]
    plt.figure(figsize = (5,3))
    plt.plot(neighbor_dist)
    plt.savefig("line plot.png")
    
    Minpoints = 

    return


#calling the function to plot the distance.
cluster_plot()
