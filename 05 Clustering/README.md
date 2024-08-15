# Clustering
- **Clustering** is a key technique in unsupervised learning, where the goal is to group a set of objects (data points) in such a way that objects in the same group (called a **cluster**) are more similar to each other than to those in other groups
- Unlike supervised learning, clustering doesn't rely on labeled data. Instead, it tries to find structure or patterns in the data by analyzing the inherent similarities or differences among data points

### Key Concepts in Clustering
1. **Similarity/Dissimilarity Measure**
   - Clustering algorithms often rely on a measure of similarity or distance between data points, such as Euclidean distance, Manhattan distance, or cosine similarity
   - The closer the data points are in the feature space, the more likely they are to belong to the same cluster
assignPtsToCluster(X,centroids) plotClusters(centroids) updateClusters(centroids)assignPtsToCluster(X,centroids) plotClusters(centroids) updateClusters(centroids)assignPtsToCluster(X,centroids) plotClusters(centroids) updateClusters(centroids)assignPtsToCluster(X,centroids) plotClusters(centroids) updateClusters(centroids)
2. **Centroids**
   - In some clustering algorithms like K-means, each cluster is represented by its centroid (the average of all points in the cluster)
   - The centroid acts as a representative point for the cluster

3. **Number of Clusters (K)**
   - Some algorithms, like K-means, require the user to specify the number of clusters in advance
   - Determining the right number of clusters can be challenging and often requires methods like the elbow method or silhouette analysis

4. **Cluster Assignments**
   - After clustering, each data point is assigned to a cluster
   - These assignments help to group similar data points together for further analysis or decision-making

### Applications of Clustering
- **Market Segmentation** : Identifying different customer segments for targeted marketing
- **Image Segmentation** : Grouping pixels in an image to identify objects or regions
- **Anomaly Detection** : Identifying outliers in data, such as fraudulent transactions
- **Document Clustering** : Grouping similar documents together for topics or themes
