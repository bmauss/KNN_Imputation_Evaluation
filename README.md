# KNN_Imputation_Evaluation

![Medium](https://miro.medium.com/max/506/1*HDtq6WgZmWkcreUuW_dnKg.jpeg)

This is an investigation into the effectiveness of using KNN Imputation on different datasets. 

KNN Imputation has become a very popular method for quickly dealing with missing data.  In this project, we'll be looking into the its effectiveness in estimating missing values against continuous and categorical data on two datasets: The Iris dataset and the Seattle Terry Stops Dataset.

## Objectives
* Calculate the RMSE of KNN Imputation on the Iris dataset.
* Calculate Accuracy of KNN Imputation on categorical data in the Terry Stops dataset.

## The Iris Dataset

For part one of this series of reports, we’ll be looking at the classic Iris dataset. This classification dataset makes for a good baseline. While it is small, the three classes (setosa, virginica, and versicolor) are perfectly balanced. The features (which are the measurements of the sepals and petals) are all continuous, which makes measuring accuracy much simpler. Best of all: it comes perfectly clean!

That being said, we still need to preprocess it for use with `KNNImputer`.

### Preprocessing

As far as preprocessing goes, it doesn’t take much. All that needs to be done is ensure that the `target` column is label encoded and then scale the data. Luckily, you can actually load the dataset with the `target` column already label encoded.  After label encoding, it's a good idea to create a copy of the dataset to use later to make an answer key.

The next step is to scale the dataset. You can argue that you don't need to scale the dataset since the continuous features are all measured on the same scale and are relatively close, so the weights wouldn't cause too much of an issue. At the same time, it doesn't hurt.

After scaling the dataset, we'll begin removing data.  We'll start by removing 10% of the total data.  These NaNs would be classified as "Missing Completely at Random" (MCAR), as each point of data has the same chance of being removed.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_removed.PNG)

Now, we'll examine the changes to the dataframe: 

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_missing.PNG)

So removing 9.7% of the data resulted in 30.6% of the rows being affected.  This is a pretty good representation of many of the curated datasets you find on Kaggle.    

The next step is to make the answer key. To do this, subset the dataframe so that it contains only rows with null values. Call the indices of these rows, and store them in a list.  Then make use of the Pandas `.loc[]` property and make a subset of the dataset copy we made earlier using the list of indices.  

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/answer_key_10.PNG)

With the answer key in hand, we can go ahead and instantiate `KNNImputer` and fit it to our dataset! We'll keep the default number of neighbors: 5.  

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_impute.PNG)

In order to interpret our results, we need to invert the scaling.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_inverse.PNG)

Now that our results and answer key are on the same scale, we can begin evaluating the estimations.  We'll do this by subsetting the test dataframe with the same index we used to create the answer key.  Next, we reset the index, this way we can perform iterative functions and processes on them.  Lastly, we'll create a new dataframe where `df[column][i] = test_df[column][i] - answer_key[column][i]`.  

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_results.PNG)

The values, then, represent the amount of error in present in the imputations.  A `0.00`, however, has two possible interpretations: 1). The value in the test table was never missing to begin with, or 2). The value imputed was a perfect estimation (e.g. the actual value). Although we don't have the means to tell the difference between the two cases on an individual basis, we do have the means to determine the number of "perfect imputations".

 ![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_perfect_imputes.PNG)
 
 As you can see above, we were able to find out how many imputations were perfect estimates by first calculating how many were imperfect (e.g. `y_hat - y_actual != 0`). The total number of imperfect *estimations* was 57.  You may remember earlier that the total number of *imputations* overall was 58. Therefore, there was only one instance of a perfect estimation via the KNNImputer. 

With that out of the way, we'll move on to calculating the Root Mean Squared Error. We're choosing to use the RMSE because it penalizes larger errors, giving us a better idea of what kind of errors to be prepared for.  Once again, we don't need to worry about not knowing which zeros to include in our calculations since we know the total number of imputed values and adding a `0` doesn't affect the sum.
