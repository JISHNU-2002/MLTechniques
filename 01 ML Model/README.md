# 01 **The ML Model**
An ML model is a mathematical representation of a system that learns from data to make predictions or decisions

- **Supervised Learning** 
	- (e.g., Linear Regression, Decision Trees ) 
	- The model is trained on labeled data (input-output pairs)
- **Unsupervised Learning** 
	- (e.g., K-Means, PCA)
	- The model identifies patterns or structures in unlabeled data
- **Semi-Supervised Learning** 
	- Falls between supervised and unsupervised learning
	- It uses both labeled and unlabeled data to build a model, which is particularly useful when labeling data is expensive or time-consuming, but a large amount of unlabeled data is readily available
- **Reinforcement Learning** 
	- (e.g., Q-Learning)
	- The model learns through interactions with an environment to maximize a reward signal

### 02 **Model Parameters and Hyperparameters**
- **Model Parameters **
	- These are the internal variables that the model learns from the data during training
	- eg: in a linear regression model, the weights (coefficients) are the parameters
  
- **Hyperparameters** 
	- These are the settings or configurations specified before training
	- They control the learning process and must be tuned
	- eg: learning rate in gradient descent, the number of trees in a random forest, or the number of clusters in K-means

### 03 **Loss Function**
The loss function quantifies how well the model's predictions match the actual data. It measures the error between the predicted output and the true output

- **Mean Squared Error (MSE)** 
	- Used in regression, it measures the average squared difference between the predicted and actual values
- **Cross-Entropy Loss** 
	- Used in classification, it measures the difference between the predicted probability distribution and the actual distribution (labels)

### 04 **Gradient Descent**
Gradient Descent is an optimization algorithm used to minimize the loss function. The basic idea is to update the model parameters in the opposite direction of the gradient of the loss function with respect to the parameters

- **Compute Gradient :** Calculate the gradient of the loss function with respect to each parameter
- **Update Parameters :** Adjust the parameters by moving them slightly in the direction that reduces the loss
- **Learning Rate :** A hyperparameter that controls the size of the step taken during each update

### 05 **Train-Test Split**
Before training the model, the dataset is typically split into
- **Training Set :** Used to train the model
- **Testing Set :** Used to evaluate the model’s performance on unseen data
The train-test split helps to ensure that the model generalizes well to new data and is not simply memorizing the training data

### 06 **Training the Model**
- Training involves feeding the training data to the model and adjusting the parameters based on the loss function and gradient descent
- The training process continues until the model converges (i.e., further training does not significantly reduce the loss)

### 07 **Model Evaluation**
After training, the model is evaluated on the test set using various metrics depending on the task

- **Accuracy**
	- The proportion of correctly predicted labels (used in classification)
- **Precision and Recall**
	- Measures of how many true positives are correctly identified versus false positives (used in classification)
- **F1 Score** 
	- The harmonic mean of precision and recall
- **R-squared** 
	- A statistical measure that represents the proportion of the variance for a dependent variable that's explained by an independent variable (used in regression)
- **Confusion Matrix** 
	- A table used to describe the performance of a classification model

### 08 **Hyperparameter Tuning**
To improve the model's performance, hyperparameters are tuned
- **Grid Search :** Testing all possible combinations of hyperparameters
- **Random Search :** Randomly sampling hyperparameters
- **Bayesian Optimization :** Using a probabilistic model to choose hyperparameters more efficiently

### 09 **Cross-Validation**
- Cross-validation is a technique for assessing how the results of a model will generalize to an independent dataset 
- It involves partitioning the dataset into a set of training and validation sets multiple times, training the model on each training set, and evaluating it on the corresponding validation set

### 10 **Deployment, Monitoring and Maintenance**


# Loss Function

- In machine learning, a **loss function** (also known as a cost function or objective function) quantifies how well a model's predictions match the actual target values
- It measures the discrepancy between the predicted values and the true values
- The goal of training a machine learning model is to minimize the loss function, thereby improving the model's accuracy
- The optimization algorithm (like gradient descent) uses the loss function to update the model's parameters, aiming to minimize the loss

### ** Types of Loss Functions**
Different types of loss functions are used depending on the problem (regression or classification) and the specific characteristics of the model
#### **01 Loss Functions for Regression**
1. **Mean Squared Error (MSE)**
     $\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$
    
   - MSE calculates the average of the squared differences between the predicted and actual values
   - It's the most commonly used loss function for regression problems
   - MSE penalizes larger errors more heavily than smaller ones due to the squaring operation

```python
from sklearn.metrics import mean_squared_error
```

```python
def error(X,y,w): 
	n = X.shape[0] 
	e = 0 
	for i in range(n): 
		y_i = model(X[i],w) 
		e = e + (y[i] - y_i)**2 
	return e/(2*m)
```

2. **Mean Absolute Error (MAE)**
     $\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |y_i - \hat{y}_i|$
   - MAE calculates the average of the absolute differences between the predicted and actual values
   - It’s less sensitive to outliers than MSE
   - MAE gives equal weight to all errors

3. **Huber Loss**
   - Huber loss is a combination of MSE and MAE
   - It is quadratic for small errors and linear for large errors
   - It’s robust to outliers while still being sensitive to small errors
   - Balances sensitivity to outliers and small errors

#### **02 Loss Functions for Classification**
1. **Binary Cross-Entropy (Log Loss)**
     $\text{Log Loss} = -\frac{1}{m} \sum_{i=1}^{m} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$

   - Used for binary classification, log loss penalizes incorrect predictions
   - It measures the difference between the predicted probability and the actual label (0/1)
   - Highly sensitive to confident wrong predictions
   - Standard choice for logistic regression and binary classification tasks

2. **Categorical Cross-Entropy**
     $\text{Categorical Cross-Entropy} = -\sum_{i=1}^{m} \sum_{j=1}^{k} y_{ij} \log(\hat{y}_{ij})$
    
   - Used for multi-class classification problems
   - It calculates the cross-entropy loss between the predicted probability distribution and the true distribution
   - Generalization of binary cross-entropy for multiple classes

3. **Hinge Loss**
     $L(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})$

   - Used primarily for training Support Vector Machines (SVMs)
   - Hinge loss penalizes predictions that are on the wrong side of the decision boundary or within a margin
   - Encourages a large margin between classes

4. **Kullback-Leibler (KL) Divergence**
     $D_{KL}(P || Q) = \sum_{i} P(i) \log \left(\frac{P(i)}{Q(i)}\right)$

   - KL Divergence measures how one probability distribution diverges from a second, expected probability distribution
   - It’s often used in models like Variational Autoencoders (VAEs)
   - Sensitive to changes in distribution

# Model Evaluation

# 01 R² Score (Coefficient of Determination)
- A statistical measure that indicates how well the independent variables explain the variance in the dependent variable
- It is commonly used in the context of regression analysis to evaluate the performance of a predictive model
- R² is a measure of how well the regression model fits the data. It represents the proportion of the variance in the dependent variable that is predictable from the independent variables
- In simpler terms, R² tells you how much of the variation in the target variable is explained by the model

$R^2 = 1 - \frac{SS_{\text{residual}}}{SS_{\text{total}}}$

Where:
- $SS_{\text{residual}}$ : sum of squares of the residuals (errors)
- $SS_{\text{total}}$ : total sum of squares, which measures the total variance in the target variable

#### **Calculating R² Score**

$R^2 = 1 - \frac{\sum_{i=1}^{m} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{m} (y_i - \bar{y})^2}$

- $y_i$ : actual value
- $hat{y}_i$ : predicted value
- $bar{y}$is : mean of the actual values
- $m$ : number of data points

```python
from sklearn.metrics import r2_score
```

```python
def r2Score(y,y_pred):
	ymean = y.mean()
	num = np.sum((y-y_pred)**2)
	den = np.sum((y-ymean)**2)
	return 1-num/den
```

#### **Interpretation of R²**
- **R² = 1** : The model perfectly explains the variance in the target variable (i.e., all data points lie exactly on the regression line)
- **R² = 0** : The model does not explain any of the variance in the target variable (i.e., the model predictions are no better than simply using the mean of the target variable)
- **0 < R² < 1** : The model explains some portion of the variance in the target variable
- **R² < 0** : This can occur when the model is worse than a simple horizontal line (mean of the target variable), indicating a poor fit

#### **Use of R² in Model Evaluation**
R² is a useful metric for understanding how well your model is performing, especially in linear regression
- **High R²** : A high R² value indicates that the model explains a large portion of the variance in the target variable, but it doesn't necessarily mean the model is perfect
- **Low R²** : A low R² value indicates that the model doesn't explain much of the variance. This could be due to the model being too simple or not capturing the underlying pattern in the data

#### **Limitations of R²**
- **Overfitting** : A high R² doesn't guarantee that the model will generalize well to new data. The model might be overfitting the training data.
- **Comparison** : R² alone doesn't allow for comparison between models with different dependent variables or datasets with different scales.
- **Non-linear Models** : R² is more intuitive for linear models, and its interpretation might be less straightforward for non-linear models.


# 02 Cross-validation
- A statistical technique used to evaluate the performance of a machine learning model by splitting the dataset into multiple subsets. It helps to ensure that the model generalizes well to unseen data, rather than just fitting well to the training data. Cross-validation is particularly useful when the dataset is limited, as it allows for more efficient use of the data.

### Why Cross-Validation?
- **Generalization**: Cross-validation provides a more reliable estimate of a model's performance on new, unseen data by using different subsets of the data for training and testing.
- **Bias-Variance Trade-off**: It helps in balancing bias (error due to overly simplistic models) and variance (error due to overly complex models) by testing the model on various splits of the data.
- **Overfitting Detection**: It can detect overfitting, where a model performs well on the training data but poorly on new data, by revealing performance discrepancies across different data subsets.

### Types of Cross-Validation

1. **K-Fold Cross-Validation**:
   - **Description**: The dataset is divided into `k` equal-sized folds. The model is trained on `k-1` folds and tested on the remaining fold. This process is repeated `k` times, with each fold serving as the test set exactly once.
   - **Common Choice**: `k=10` is a common choice, known as 10-fold cross-validation.
   - **Output**: It produces `k` performance scores, which are then averaged to provide a final estimate.
   - **Pros**: Provides a good balance between bias and variance. Works well for most datasets.
   - **Cons**: Can be computationally expensive for large datasets or complex models.

   ![K-Fold Cross-Validation](https://scikit-learn.org/stable/_images/grid_search_cross_validation.png)
   *Source: scikit-learn documentation*

2. **Stratified K-Fold Cross-Validation**:
   - **Description**: Similar to K-Fold Cross-Validation, but it ensures that each fold has a similar distribution of classes as the original dataset, which is particularly useful for imbalanced datasets.
   - **Pros**: Preserves the distribution of target classes across folds, leading to more reliable performance estimates.
   - **Cons**: Can be computationally expensive, similar to regular K-Fold Cross-Validation.

3. **Leave-One-Out Cross-Validation (LOOCV)**:
   - **Description**: In LOOCV, each observation in the dataset is used once as a test set, and the model is trained on the remaining data. This results in `n` iterations, where `n` is the number of observations.
   - **Output**: Provides `n` scores, which are averaged for the final estimate.
   - **Pros**: Uses the maximum amount of data for training in each iteration, which is beneficial when the dataset is small.
   - **Cons**: Extremely computationally expensive for large datasets, as it requires training the model `n` times.

4. **Holdout Method**:
   - **Description**: The dataset is randomly split into two subsets: a training set and a testing set (commonly 70% training, 30% testing). The model is trained on the training set and evaluated on the testing set.
   - **Pros**: Simple and fast.
   - **Cons**: Results can vary depending on how the data is split, and it may not fully utilize the data.

5. **Repeated K-Fold Cross-Validation**:
   - **Description**: The K-Fold Cross-Validation process is repeated multiple times with different random splits of the data. The final performance score is averaged across all iterations.
   - **Pros**: Provides a more stable estimate of model performance by reducing the variability associated with a single K-Fold split.
   - **Cons**: Increases computational cost due to repeated model training and evaluation.

6. **Time Series Cross-Validation (Rolling/Sliding Window)**:
   - **Description**: For time series data, the data is split based on time, ensuring that the model is trained on past data and tested on future data. A common approach is to use a rolling or sliding window where the training set expands with each iteration, and the test set moves forward in time.
   - **Pros**: Respects the temporal order of the data, which is critical for time series predictions.
   - **Cons**: May result in less data for training in early iterations.

### How Cross-Validation Works
- **Step 1**: Split the dataset into `k` equal-sized folds (for K-Fold Cross-Validation).
- **Step 2**: Train the model on `k-1` folds and test it on the remaining fold.
- **Step 3**: Repeat the process for each fold, so each fold serves as the test set exactly once.
- **Step 4**: Calculate the performance metric (e.g., accuracy, F1 score) for each fold.
- **Step 5**: Average the performance metrics to get a final estimate of the model’s performance.

### Example of K-Fold Cross-Validation in Python:

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load dataset
X, y = load_iris(return_X_y=True)

# Define the model
model = RandomForestClassifier()

# Perform 10-fold cross-validation
scores = cross_val_score(model, X, y, cv=10)

# Output the scores
print("Cross-validation scores: ", scores)
print("Mean accuracy: ", scores.mean())
print("Standard deviation: ", scores.std())
```

### Interpretation of Results:
- **Scores**: An array of accuracy scores (or other metrics) for each fold.
- **Mean Accuracy**: The average accuracy across all folds, providing a general estimate of model performance.
- **Standard Deviation**: Indicates the variability of the model’s performance across different folds; lower values suggest more consistent performance.

### Advantages of Cross-Validation:
- **More Reliable Estimates**: By using different subsets of the data, cross-validation provides a more robust estimate of model performance.
- **Efficient Use of Data**: All data points are used for both training and testing, maximizing the utilization of the dataset.
- **Overfitting Detection**: Cross-validation helps detect overfitting by revealing how the model performs on unseen data.

### Disadvantages of Cross-Validation:
- **Computationally Expensive**: For large datasets or complex models, cross-validation can be computationally intensive due to repeated training and testing.
- **Not Always Suitable**: In some scenarios, like time series data, traditional cross-validation methods may not be appropriate and require specialized techniques.

Model evaluation metrics and methods are crucial for assessing the performance of machine learning models. They provide quantitative measures to help determine how well a model is making predictions and where it may be lacking. Different metrics and methods are used depending on the type of problem (classification, regression, etc.) and the specific goals of the analysis.

# **Model Evaluation Metrics**

#### A. **Classification Metrics**

1. **Accuracy**
   - **Definition**: The ratio of correctly predicted instances to the total instances. 
   - **Formula**: \(\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}\)
   - **Use Case**: Best for balanced datasets where the classes are of equal importance.

2. **Precision**
   - **Definition**: The ratio of correctly predicted positive observations to the total predicted positive observations.
   - **Formula**: \(\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}\)
   - **Use Case**: Useful when the cost of false positives is high (e.g., spam detection).

3. **Recall (Sensitivity or True Positive Rate)**
   - **Definition**: The ratio of correctly predicted positive observations to all observations in the actual class.
   - **Formula**: \(\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}\)
   - **Use Case**: Important when the cost of false negatives is high (e.g., cancer detection).

4. **F1 Score**
   - **Definition**: The harmonic mean of precision and recall, providing a balance between the two.
   - **Formula**: \(\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}\)
   - **Use Case**: Useful when there’s an uneven class distribution and a trade-off between precision and recall is needed.

5. **Confusion Matrix**
   - **Definition**: A table that summarizes the performance of a classification model by showing the true positives, true negatives, false positives, and false negatives.
   - **Use Case**: Provides a comprehensive overview of classification performance.

   |                | Predicted Positive | Predicted Negative |
   |----------------|--------------------|--------------------|
   | **Actual Positive** | True Positive (TP)    | False Negative (FN)   |
   | **Actual Negative** | False Positive (FP)   | True Negative (TN)    |

6. **ROC Curve and AUC (Area Under the Curve)**
   - **ROC Curve**: A graphical representation of a model's diagnostic ability, plotting the true positive rate (recall) against the false positive rate.
   - **AUC**: The area under the ROC curve, representing the likelihood that the model will distinguish between positive and negative classes.
   - **Use Case**: AUC is used to compare models; a higher AUC indicates better performance.

7. **Log Loss**
   - **Definition**: Measures the uncertainty of your predictions based on the probability assigned to the correct class.
   - **Formula**: \(\text{Log Loss} = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(p_i) + (1 - y_i) \log(1 - p_i)]\)
   - **Use Case**: Used for probabilistic classification models where outputs are probability values.

#### B. **Regression Metrics**

1. **Mean Absolute Error (MAE)**
   - **Definition**: The average of the absolute differences between predicted and actual values.
   - **Formula**: \(\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|\)
   - **Use Case**: Best for situations where all errors are equally significant.

2. **Mean Squared Error (MSE)**
   - **Definition**: The average of the squared differences between predicted and actual values.
   - **Formula**: \(\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2\)
   - **Use Case**: Penalizes larger errors more than MAE; useful when large errors are particularly undesirable.

3. **Root Mean Squared Error (RMSE)**
   - **Definition**: The square root of the MSE, providing an error measure in the same units as the target variable.
   - **Formula**: \(\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}\)
   - **Use Case**: Similar to MSE, but more interpretable because it’s in the same units as the original data.

4. **R-Squared (R²)**
   - **Definition**: The proportion of variance in the dependent variable that is predictable from the independent variables.
   - **Formula**: \(\text{R}^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}\)
   - **Use Case**: Indicates how well the independent variables explain the variance in the target variable. R² values range from 0 to 1, with higher values indicating better fit.

5. **Adjusted R-Squared**
   - **Definition**: A modified version of R² that adjusts for the number of predictors in the model. It’s particularly useful when comparing models with different numbers of predictors.
   - **Formula**: \(\text{Adjusted R}^2 = 1 - \left( \frac{(1-R^2)(n-1)}{n-p-1} \right)\)
   - **Use Case**: More accurate for evaluating models with different numbers of features.

### 2. **Model Evaluation Methods**

#### A. **Holdout Method**
- **Description**: The dataset is randomly split into two parts: a training set and a test set. The model is trained on the training set and evaluated on the test set.
- **Pros**: Simple and fast.
- **Cons**: The performance estimate can vary depending on how the data is split, and it may not fully utilize the data.

#### B. **Cross-Validation**
- **Description**: The data is split into `k` subsets (folds). The model is trained on `k-1` folds and tested on the remaining fold. This process is repeated `k` times, with each fold being used as the test set once.
- **Types**:
  - **K-Fold Cross-Validation**: A common form where the data is split into `k` equal parts.
  - **Stratified K-Fold**: Ensures each fold has the same proportion of classes, useful for imbalanced data.
  - **Leave-One-Out Cross-Validation (LOOCV)**: Each data point is used as a single test set; computationally expensive but uses the maximum amount of data.
- **Pros**: Provides a more reliable estimate of model performance.
- **Cons**: Can be computationally expensive.

#### C. **Bootstrap Method**
- **Description**: A resampling technique where multiple training datasets are created by sampling with replacement from the original data. The model is trained on these samples and evaluated on the out-of-bag data (data not included in the sample).
- **Pros**: Useful for small datasets and provides a measure of model stability.
- **Cons**: Can be computationally expensive and complex.

#### D. **Leave-P-Out Cross-Validation**
- **Description**: Similar to LOOCV, but instead of leaving one sample out, `p` samples are left out at each iteration. The process is repeated for all combinations.
- **Pros**: Thoroughly evaluates the model.
- **Cons**: Computationally expensive and typically used only for small datasets.

### 3. **Choosing the Right Metric and Method**

- **For Classification**: 
  - Use accuracy when classes are balanced.
  - Use precision, recall, or F1 score when classes are imbalanced.
  - Use ROC-AUC for probabilistic models.
- **For Regression**: 
  - Use MAE or MSE for continuous outcomes.
  - Use R² or adjusted R² for understanding variance explained by the model.
- **For Cross-Validation**: 
  - Use K-Fold Cross-Validation for most scenarios.
  - Use Stratified K-Fold for imbalanced data.
  - Use LOOCV or Leave-P-Out when the dataset is small.

The choice of evaluation metrics and methods should align with the specific goals of the project, the nature of the data, and the type of model being used. Each metric and method has its strengths and weaknesses, so understanding the context is crucial for making the best decision.