# Dimensionality Reduction
- Process of reducing the number of input variables or features in a dataset while retaining as much information as possible
- It helps in simplifying models, reducing computation time, and mitigating the "curse of dimensionality" where models become more complex and less effective as the number of dimensions (features) increases

#### Advantages of Dimensionality Reduction
- Improves Model Performance
- **Reduces Overfitting** 
	- Fewer features mean less noise, wh can lead to more generalizable models
- Speeds Up Computation
- Visualization easier interpretation

# Curse of Dimensionality
- The **Curse of Dimensionality** describes the various phenomena that arise when analyzing and organizing data in high-dimensional spaces (with many features)
- As the number of dimensions increases, the volume of the space grows exponentially, making the data sparse
- This sparsity means that the data points are far apart from each other, making it difficult for algorithms to find patterns or clusters

### Effects of the Curse of Dimensionality
- Increased Complexity
- Overfitting
- Distance Metrics Become Less Informative- High-dimensional data is hard to visualize, but reducing it to 2 or 3 dimensions allows for 

# Dimensionality Reduction Methods
## Feature Selection
- **Feature Selection** is a technique used to select a subset of the most relevant features (or variables) from the original set of features in the dataset
- The aim is to reduce the number of features while preserving the predictive power of the model
#### Types of Feature Selection
1. **Filter Methods**
   - Filter methods evaluate the relevance of each feature independently of any machine learning algorithm
   - They use statistical techniques to rank features based on their correlation with the target variable

     - **Chi-Square Test** : Measures the association between categorical features and the target variable
     - **Correlation Coefficient** : Evaluates the linear relationship between continuous features and the target variable
     - **Mutual Information** : Measures the amount of information one feature provides about the target variable

2. **Wrapper Methods**
   - Wrapper methods evaluate subsets of features by training and testing a model on different combinations of features
   - These methods consider the interactions between features but can be computationally expensive

     - **Recursive Feature Elimination (RFE)** : Recursively removes the least important features and builds models on the remaining features to determine the best subset
     - **Forward Selection** : Starts with no features and adds features one by one, evaluating the model performance at each step
     - **Backward Elimination** : Starts with all features and removes the least significant features one by one, evaluating model performance

3. **Embedded Methods**
   - Embedded methods perform feature selection during the process of model training
   - These methods are less computationally expensive than wrapper methods and consider feature interactions

     - **Lasso Regression** : Adds a penalty term to the linear regression that shrinks the coefficients of less important features to zero, effectively performing feature selection
     - **Decision Trees** : Naturally perform feature selection by selecting the most important features to split the data at each node

## Feature Extraction
- **Feature Extraction** is a process of transforming the original features into a new set of features, which are often lower-dimensional and more informative
- Unlike feature selection, which involves selecting a subset of existing features, feature extraction creates new features from the original ones
#### Techniques for Feature Extraction
1. **Principal Component Analysis (PCA)**
   - PCA transforms the original features into a new set of orthogonal components (principal components) that capture the maximum variance in the data
   - The first few components can be used to reduce the dimensionality while retaining most of the information
   - PCA is widely used for dimensionality reduction, noise reduction, and visualization of high-dimensional data

2. **Linear Discriminant Analysis (LDA)**
   - LDA is a supervised method that transforms features to maximize the separation between different classes
   - It projects the data onto a lower-dimensional space where the classes are as separable as possible
   - LDA is commonly used in classification tasks where the goal is to reduce dimensionality while preserving class separability

3. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
   - t-SNE is a nonlinear dimensionality reduction technique that is particularly useful for visualizing high-dimensional data in 2D or 3D by preserving the local structure of the data
   - t-SNE is often used for exploratory data analysis to uncover patterns and clusters in complex datasets

4. **Autoencoders**
   - Autoencoders are a type of neural network that learns to compress the input data into a lower-dimensional representation and then reconstruct it
   - The compressed representation serves as a feature extraction
   - Autoencoders are used for tasks like data compression, anomaly detection, and dimensionality reduction
