# Implementing core machine learning algorithms for revision purposes (supervised, unsupervised, semi-supervised, and reinforcement learning)

**Supervised Learning**:

  1. **Logistic Regression:** This project trains a model to classify breast tumors as malignant or benign from cell-measurement data.
  
  2. **Decision Tree:** This project trains a model to predict how productive garment factory teams will be, and then shows which factors drive that prediction.

  3. **KNN Classification:** This project uses KNN machine learning algorithm to classify breast tumors as benign or malignant based on measurements extracted from cell nuclei. (similar to logistic_regression.py)
  
  4. **Elastic Net Regression:** This project uses Elastic Net Regression to predict house prices from property characteristics while reducing overfitting and selecting the most useful features through combined L1 and L2 regularization.
  
  5. **Bayesian Regression:** This project leverages Bayesian regression to model insurance costs, providing a way to identify key drivers while offering reliable predictions for future medical expenses.
  
  6. **Lasso Regression:** This project uses Lasso Regression to predict diabetes from medical features, including data preprocessing, exploratory analysis, hyperparameter tuning, feature selection, model evaluation (accuracy, ROC-AUC, confusion matrix).
  
  7. **Quadratic Discriminant Analysis:** QDA for classification on Wine dataset using different covariance matrices per class to create quadratic decision boundaries.
  
  8. **Support Vector Regression:** SVM for predicting continuous values on Diabetes dataset, Energy Efficiency dataset, and Concrete Strength dataset using kernel tricks to handle non-linear relationships with epsilon-insensitive loss.
  9. **AdaBoost:** This project is an AdaBoost-based Customer Churn Prediction system that analyzes telecom customer data to identify customers likely to leave the service, enabling proactive retention strategies.
  10. **Lasso Regression**: A complete machine learning pipeline using L1 regularization to predict diabetes from medical features with automatic feature selection, hyperparameter tuning, and performance evaluation comparing Lasso to standard Logistic Regression.
  11. **Polynomial Regression**: This project employs a regression pipeline with polynomial feature expansion to capture non-linear relationships between meteorological/pollutant data and PM 2.5 levels, evaluating predictive accuracy through residual analysis and error metrics.

**Unsupervised Learning**

1. **Affinity Propagation**: This project segments customers into distinct personas by identifying representative "exemplar" individuals within the data, allowing for targeted marketing strategies based on observed spending habits and income levels.

2. **Balanced Iterative Reducing and Clustering using Hierarchies**: This project implements and evaluates the BIRCH algorithm on the Wine dataset to perform unsupervised clustering, utilizing PCA for visualization and metric-based analysis to compare clustering performance against true target classes.

3. **Density-Based Spatial Clustering of Applications with Noise**: This project identifies clusters based on regional density, allowing for the detection of non-spherical shapes and anomalous noise points.

4. **K-Means**: This project employs centroid-based partitioning to segment data into predefined, spherical clusters based on feature similarity.

5. **Hierarchical Agglomerative Clustering**: This project builds a bottom-up hierarchy visualized via a dendrogram, allowing for intuitive identification of customer segments by analyzing the Euclidean distance between merging groups.
    
6. **Mean Shift**: This project applies the Mean Shift clustering algorithm to the Mall Customers dataset to automatically discover customer segments based on annual income and spending behavior for targeted marketing analysis.
    
7. **Factor Analysis**: This project used Factor Analysis to the Big Five Personality Dataset to uncover latent personality traits and validate the five-factor psychological model through dimensionality reduction and factor extraction.

8. **Uniform Manifold Approximation and Projection**: Applied UMAP dimensionality reduction to the Iris dataset to transform four-dimensional flower measurements into a two-dimensional embedding that reveals natural species clusters and relationships.
9. **Non-Negative Matrix Factorization**: This project applies NMF to the  Breast Cancer Wisconsin dataset to extract 5 latent feature components and evaluate reconstruction accuracy.
10. **t-Distributed Stochastic Neighbor Embedding**: This project applies t-SNE to the Wine Recognition dataset to project 13 chemical features into a 2D space, revealing natural clusters among three wine cultivars.
    
11. **Gaussian Mixture Model**: This Python script implements Gaussian Mixture Model (GMM)-based semi-supervised learning on CIFAR-10 with a CNN feature extractor.


**Semi-Supervised Learning**

1. **Generative Adversarial Network (GAN)**: This project implements GAN on the MNIST dataset, training with only 1,000 labeled samples while leveraging unlabeled data to achieve competitive classification accuracy and generate realistic digit images.
  
2. **Graph-based Semi-Supervised Learning**: This project implements graph-based semi-supervised learning on MNIST using a neural network with graph Laplacian regularization, achieving competitive classification accuracy with only 100 labeled samples by leveraging the manifold structure of unlabeled data.


**Reinforcement Learning**

1. **Deep Q-Network (DQN)**: DQN agent that learns to balance a pole on a cart through reinforcement learning, with comprehensive training visualization and performance analysis.

2. **REINFORCE (a.k.a. Monte Carlo Policy Gradient)**: A policy gradient REINFORCE agent that learns to balance a pole on a cart through direct policy optimization, with comprehensive training visualization and analysis.

3. **Deep Deterministic Policy Gradient (DDPG)**: A DDPG actor-critic agent that learns to balance a pole on a cart with continuous action space, featuring experience replay and target networks.

4. **Twin Delayed Deep Deterministic Policy Gradient (TD3)**: A TD3 agent with dual critics and delayed policy updates for stable learning on CartPole.

5. **Soft Actor-Critic (SAC)**: A SAC agent with automatic temperature tuning that learns to balance CartPole through maximum entropy reinforcement learning.

6. **Proximal Policy Optimization (PPO)**: A PPO agent with clipped surrogate objective and multiple epochs of updates for stable and sample-efficient CartPole training.

7. **Q-Learning**: A Q-Learning agent that learns to navigate a warehouse grid efficiently while managing limited energy and avoiding hazards.













