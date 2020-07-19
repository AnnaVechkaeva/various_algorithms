import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
import sys

# to run the code:
# python kmean.py ./path/to/input/file.csv 2 plot_name.png

args = sys.argv

input_file = args[1]
n_clusters = int(args[2])
img_file = args[3]

# open data file
df = pd.read_csv(input_file)
data = StandardScaler().fit_transform(df)

def k_means(data, n_clusters, max_iter=100, random_state = 100):
    # set random centroids
    np.random.RandomState(random_state)
    random_idx = np.random.permutation(data.shape[0])
    centroids = data[random_idx[:n_clusters]]
    
    # compute distances
    distance = np.zeros((data.shape[0], n_clusters))
    for i in range(n_clusters):
        row_norm = norm(data - centroids[i, :], axis=1)
        distance[:, i] = np.square(row_norm)
        
    # compute clusters
    
    curr_clusters = np.argmin(distance, axis=1)
    
    for n in range(max_iter):
        # update centroids
        new_centroids = np.zeros((n_clusters, data.shape[1]))
        for i in range(n_clusters):
            labels = np.argmin(distance, axis=1)
            new_centroids[i, :] = np.mean(data[labels == i, :], axis=0)
        
        if np.all(new_centroids != centroids):
            centroids = new_centroids
            
            distance = np.zeros((data.shape[0], n_clusters))
            for i in range(n_clusters):
                row_norm = norm(data - centroids[i, :], axis=1)
                distance[:, i] = np.square(row_norm)
            new_custers = curr_clusters = np.argmin(distance, axis=1)
        else:
            return {"centroids": centroids, "clusters": new_custers}
        return {"centroids": centroids, "clusters": new_custers}

result = k_means(data, n_clusters)

print(result)

# Plot the clustered data
centroids = result["centroids"]
clusters = result["clusters"]

fig, ax = plt.subplots(figsize=(6, 6))
plt.scatter(data[clusters == 0, 0], data[clusters == 0, 1])
plt.scatter(data[clusters == 1, 0], data[clusters == 1, 1])
plt.scatter(centroids[:, 0], centroids[:, 1], s=200)

plt.savefig(img_file)
