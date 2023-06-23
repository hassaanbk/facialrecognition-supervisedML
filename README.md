# Facial Recognition using Random Forest and K-Means Clustering

This Jupyter Notebook code demonstrates the process of facial recognition using Random Forest and K-Means clustering. It utilizes the Olivetti Faces dataset from scikit-learn, which consists of 400 grayscale images of 40 individuals (10 images per individual).

## Prerequisites

Before running the code, ensure that you have the following libraries installed:

- scikit-learn
- matplotlib
- seaborn
- numpy
- joblib

You can install these libraries using pip:

```shell
pip install scikit-learn matplotlib seaborn numpy joblib
```

## Code Overview

1. Importing Required Libraries:
   - The necessary libraries for the code are imported, including scikit-learn modules, matplotlib, seaborn, numpy, and joblib.

2. Loading and Preprocessing the Dataset:
   - The Olivetti Faces dataset is loaded using the `fetch_olivetti_faces` function from scikit-learn.
   - The dataset is centered and locally centered to ensure that the facial features are properly aligned.

3. Visualizing the Dataset:
   - The histogram of pixel values for the first data point is plotted to visualize the frequency of occurrences.
   - A subset of 20 faces from the dataset is plotted using matplotlib.

4. Training and Evaluating the Random Forest Classifier:
   - A Random Forest Classifier is initialized with 40 estimators.
   - StratifiedShuffleSplit is used for test/train splitting with 10 splits.
   - The classifier is trained and evaluated on each split, and accuracy and confusion matrices are printed.

5. Cross-Validation using Random Forest Classifier:
   - Cross-validation is performed using the `cross_val_score` function from scikit-learn to evaluate the model's performance.

6. Determining the Optimal Number of Clusters:
   - Silhouette scores are calculated for different numbers of clusters using K-Means clustering.
   - The number of clusters with the highest silhouette score is selected as the best number of clusters.

7. Visualizing Clusters:
   - K-Means clustering is applied to the dataset using the best number of clusters.
   - The clusters are visualized using a scatter plot, where each point represents a face.

8. Creating a Pipeline:
   - A pipeline is created with K-Means clustering and Random Forest Classifier as the successive steps.

9. Training and Evaluating the Pipeline:
   - The pipeline is trained and evaluated using StratifiedShuffleSplit with 10 splits.
   - Accuracy and confusion matrices are printed for each split.

10. Saving the Model:
    - The trained pipeline model is saved using joblib to a file named `group_project_model.pkl`.

11. Example API Request:
    - An example code snippet is provided to demonstrate how to obtain a specific data point as JSON for testing purposes.
