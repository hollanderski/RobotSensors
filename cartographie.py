# To use math functions
import numpy as np

# To read csv files
import pandas as pd

#To plot data
import matplotlib.pyplot as plt
import itertools

# Clustering function
from clustering import clustering

plt.figure()

fig = plt.gcf()
ax = plt.gca()

plt.ion()

df_lidar = pd.read_csv('scan.csv')
df_lidar['%time'] = (df_lidar['%time'] - df_lidar['%time'].tolist()[0])/1000000000
df_lidar.set_index('%time',inplace=True)
df_lidar.columns = [name[6:] for name in df_lidar]


min=df_lidar['range_min'][0]
max=df_lidar['range_max'][0]

for time, row in df_lidar.iterrows():
    n = 360  # <-- number of points
    ranges = np.array(row[10:10+n])

    angles=np.arange(row.angle_min,row.angle_max+row.angle_increment,row.angle_increment)
    x=ranges*np.cos(angles)
    y=ranges*np.sin(angles)

    # Enlever les points provenant du robot 
    masked = x==0
    x = np.ma.masked_array(x, mask = masked)
    y = np.ma.masked_array(y, mask = masked)
    scan=np.array([x,y])

    (clusters_id, clusters) = clustering(np.array(scan))

    cmap = plt.cm.get_cmap('hsv', 11) #11 #18
    plt.scatter(scan[0], scan[1], c=cmap(clusters_id))

    for clust in clusters :
        plt.plot(clust.poly[:,0], clust.poly[:,1], linewidth=1, color='black')
    


    plt.axis('square')
    
    plt.draw()
    plt.pause(0.2)
    plt.clf()
