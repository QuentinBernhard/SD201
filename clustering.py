import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime
import time

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

data = pd.read_csv('/Users/quents/Documents/git/SD201/shuffled_data.csv')
data.drop(columns = data.columns[0], inplace=True)
data.dropna()


def convertTime(str_time):
    timestamp = datetime.datetime.strptime(str_time,"%Y-%m-%d")
    tuple = timestamp.timetuple()
    timestamp = time.mktime(tuple)

    return(timestamp)


data['Date'] = data['Date'].apply(convertTime)



airlines = data['Airline'].unique()


origins = data['Origin'].unique()
destinations = data['Destination'].unique()
airports = np.unique(np.concatenate([origins, destinations]))

# print(airports)
# print(origins)
# print(destinations)

def convertAirlines(airline):
    return(np.where(airlines == airline)[0][0])

def convertAirport(airport):
    return(np.where(airports == airport)[0][0])

data['Airline'] = data['Airline'].apply(convertAirlines)

data['Origin'] = data['Origin'].apply(convertAirport)
data['Destination'] = data['Destination'].apply(convertAirport)

data = data.dropna()

data_cluster = data.drop(columns=['DepDelay', 'ArrDelay', 'ArrTime', 'AirTime'])



tsne = TSNE(2, random_state=53)             #53 correspond au random state avec le moins d'inertie (erreur)
tsne_result = tsne.fit_transform(data_cluster)

print(tsne_result.shape)

km = KMeans(
n_clusters=13, init='k-means++',
n_init=10, max_iter=300, 
tol=1e-04, random_state=0)

y_km = km.fit_predict(tsne_result)


fte_colors = {
        0: "#FF4E33",
        1: "#FF9633",
        2: "#FFEA33",
        3: "#94FF33",
        4: "#278C20",
        5: "#12DB8A",
        6: "#12DBD6",
        7: "#10AADB",
        8: "#0C67C3",
        9: "#0F21AF",
        10: "#8C2AE7",
        11: "#B213E7",
        12: "#E713E0",
        13: "#E6207A"
    }

km_colors = [fte_colors[label] for label in km.labels_]

clusters = [[]]*(max(km.labels_)+1)

index = data[data['ArrDelay'] > 0].index



fig = plt.figure(1)
ax = fig.add_subplot()

ax.scatter(tsne_result[:,0], tsne_result[:,1], c = km_colors, marker = 'o')
ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], c = 'red', marker = '+')

plt.show()
