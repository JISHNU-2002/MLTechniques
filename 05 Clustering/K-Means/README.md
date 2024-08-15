# K-Means
- **K-Means** is one of the most popular and widely used clustering algorithms in unsupervised learning
- The goal of K-Means is to partition a dataset into **K clusters**, where each data point belongs to the cluster with the nearest mean, also known as the cluster centroid
- The algorithm works by iteratively refining these cluster centroids to minimize the overall variance within each cluster

### K-Means Steps

1. **Initialize the Centroids**
	   - Choose the number of clusters, K
	   - Randomly initialize K centroids. These centroids are the initial cluster centers

2. **Assign Data Points to the Nearest Centroid**
	   - For each data point in the dataset, calculate the distance (usually Euclidean distance) to each of the K centroids
	   - Assign the data point to the cluster whose centroid is closest to it

3. **Update Centroids**
	   - Once all data points are assigned to clusters, calculate the new centroids by taking the mean of all data points in each cluster
	   - These new centroids are the updated cluster centers

4. **Repeat**
	   - Repeat the assignment and update steps until the centroids no longer change significantly or until a maximum number of iterations is reached
	   - This means the algorithm has converged, and the clusters are stable

5. **Final Clusters**
	   - The algorithm outputs the final clusters, with each data point assigned to a specific cluster

### Choosing the Right Number of Clusters (K)
- **Elbow Method**
	- Plot the inertia (sum of squared distances) against different values of K
	- The point at which the decrease in inertia slows down (forming an "elbow") is often considered the optimal K
  
- **Silhouette Score**
	- Measures how similar a point is to its own cluster compared to other clusters
	- A higher silhouette score indicates well-defined clusters

### Limitations of K-Means
- **Need to Specify K** : The number of clusters must be chosen beforehand, which may not always be obvious
- **Sensitivity to Initialization** : The final clusters can vary based on the initial random choice of centroids. Multiple runs with different initializations can help mitigate this
- **Assumption of Spherical Clusters** : K-Means assumes that clusters are spherical and of similar size. It might struggle with clusters of different shapes and densities
- **Outliers** : K-Means can be sensitive to outliers, as they can significantly affect the position of centroids
