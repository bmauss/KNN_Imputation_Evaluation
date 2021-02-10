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

## Impute the Data

With the answer key in hand, we can go ahead and instantiate `KNNImputer` and fit it to our dataset! We'll keep the default number of neighbors: 5.  

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_impute.PNG)

In order to interpret our results, we need to invert the scaling.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_inverse.PNG)

Now that our results and answer key are on the same scale, we can begin evaluating the estimations.  We'll do this by subsetting the test dataframe with the same index we used to create the answer key.  Next, we reset the index, this way we can perform iterative functions and processes on them.  Lastly, we'll create a new dataframe where `df[column][i] = test_df[column][i] - answer_key[column][i]`.  

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_results.PNG)

The values, then, represent the amount of error in present in the imputations.  A `0.00`, however, has two possible interpretations: 1). The value in the test table was never missing to begin with, or 2). The value imputed was a perfect estimation (e.g. the actual value). Although we don't have the means to tell the difference between the two cases on an individual basis, we do have the means to determine the number of "perfect imputations".

 ![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_perfect_imputes.PNG)
 
 As you can see above, we were able to find out how many imputations were perfect estimates by first calculating how many were imperfect (e.g. `y_hat - y_actual != 0`). The total number of imperfect *estimations* was 57.  You may remember earlier that the total number of *imputations* overall was 58. Therefore, there was only one instance of a perfect estimation via the KNNImputer. 

With that out of the way, we'll move on to calculating the Root Mean Squared Error. We're choosing to use the RMSE because it penalizes larger errors, giving us a better idea of what kind of errors to be prepared for.  Once again, we don't need to worry about not knowing which zeros to include in our calculations since we know the total number of imputed values and adding `0` doesn't affect the sum.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/10_rmse.PNG)

Imputing 10% of our data through KNN Imputation *can* result in an RMSE of 0.43 centimeters. Of course, since the estimates are made based on the data that is *present*, this means that the RMSE will change depending on which data is missing.  If it so happened that the missing 10% had a lot of outliers, the algorithm would have to make an estimate using data that was closer to the mean, resulting in a higher RMSE.  On the otherhand, if most of the outliers were still present, then the missing values would be closer to the mean, making KNN imputer more accurate.

To get a true feel for how well the algorithm works, we would need to remove the random seed and run the experiment 30+ times and find the average RMSE.  For now, we'll just stick with 0.43 centimeters. 

## 20% Missing Data

Let's see what happens when we remove 20% of the data.  We'll change the random seed value so that the datawe remove will be different from the previous experiment.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/20_removed.PNG)

Now let's inspect the state of the dataset.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/20_removed.PNG)

## Imputation and Results

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/iris/20_results.PNG)

So the RMSE actually improved! What could cause this?  Well, one reason could be because we changed the `random seed` value.  This switched up what elements of the dataframe were removed.  As was mentioned earlier, if the outlier data weren't removed, then the algorithm would just need to calculate its estimations based on data that was closer to the mean (of which there is plenty). 

Another reason is the importance of the `target` variable, the only data left untouched during both of the experiments.  As was stated, the Iris dataset is curated for classification tasks.  We know that the `target` variable is composed of three evenly distributed classes. Again, these classes represent three different *species* of Iris flowers.  Since these are different species, they must have characteristics that are unique to each of them (if they had no unique characteristics, they wouldn't be split up into different species). This implies that one can estimate (with a degree of accuracy) qualities of an iris flower as long as they knew what species they belonged to.  Even if the algorithm faced a row with no features, it would still know what class it belonged to and could make calculations based entirely off of that.  Therefore, the consistency in our results is due to the fact that the classes are distinct from one another, with few overlapping qualities.  This made estimating a much simpler task for the algorithm.

## Categorical Data  

Our next dataset is the antithesis of the Iris dataset.  It's bloated, the classes are highly imbalanced, and it's got over 30,000 rows.  We're using the Seattle Terry Stops dataset. Every feature is categorical, and the columns include such details as race of the subjects and officers, time of day that an incident took place, how the call was issued (eg was it from dispatch or did the officer initiate the stop), and what the result of the stop was.  The target variable is the column `arrest_flag` which is comprised of binary classes. 

### Preprocessing

To get the data ready, we first need to remove the unnecessary columns.  These columns are statistically insignificant as they are more or less unique identifiers and are less likely to have any relationship to the target variable.  In terms of our experiment, they would skew our results since we'd be asking the algorithm to estimate a **unique** value based on data present in the table, an impossible task.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/t_stops/drop_cols.PNG)

Next we label encode the entire dataset.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/t_stops/label.PNG)

The following steps are to scale and then remove the data!  This time around, we're going to jump straight to removing 20%.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/t_stops/nans.PNG)

As you can see, the number of rows affected here are drastically different from the Iris dataset under similar circumstances.  99% of the rows have at least a single value missing.  Very few curated datasets come this messy.  So let's see how well KNN works this time around.

### Imputation

Just like with the Iris dataset, we instantiate the KNNImputer and fit it to the dataset.  A word of warning, if you're going to run this notebook, you may want to do so using Google Colab.  The KNN imputation process on this dataset is computationally expensive and may require up to 6 GiB of memory to perform.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/t_stops/knn.PNG)

Again, we'll invert the scaling. 

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/t_stops/invert.PNG)

Before we move on to evaluation, we need to complete one additional step.  You'll notice that many of the values imputed are `floats`, but label data are `integers`.  To properly evaluate we'll need to round our answers to the nearest whole number.

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/t_stops/rounding.PNG)

Now for our results:

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/t_stops/results_5.PNG)

Unlike the Iris dataset, there is no trophy for "Close Enough". Accuracy is determined strictly by what categories were correctly estimated divided by the total number of imputations. So, using 5 neighbors, our accuracy was 55%.  This is just a little bit better than random chance.

Let's see if increasing the number of neighbors helps out.

#### 10 Neighbors

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/t_stops/results_10.PNG)

With 10 neighbors accuracy decreased.  This is proof that more data doesn't always make better results.  This is because as you increase the number of neighbors you're including neighbors that are farther and, therefore, less similar to the value you're trying to impute.  This will pull the mean farther from the actual value.

With that in mind, let's try fewer neighbors.

#### 2 Neighbors

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/t_stops/results_2.PNG)

There's some improvement! 56% accuracy with 2 neighbors. One issue here though is that any odd number divided by two ends in "*n*.5". All of these values would be rounded up, even if it should have been rounded down. To avoid this, it is best practice to use an odd number of neighbors.

#### 1 Neighbor

![GitHub](https://raw.githubusercontent.com/bmauss/KNN_Imputation_Evaluation/main/images/t_stops/results_1.PNG)

