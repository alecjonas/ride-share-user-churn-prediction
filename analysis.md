# Rideshare Customer Churn 

## Introduction
Daenerys of the House Targaryen, the First of Her Name, The Unburnt, Queen of the Andals, the Rhoynar and the First Men, Queen of Meereen, Khaleesi of the Great Grass Sea, Protector of the Realm, Lady Regent of the Seven Kingdoms, Breaker of Chains and Mother of Dragons has instructed us to examine dragon share (think Uber or Lyft) in the 7 kingdoms. 

Given a set of data, can we accurately predict if users will churn (become inactive after 30 days of non platform usage)?



## Data 
The data set contained an initial 50,000 rows with 12 columns. The data set was split into two different files one for training which was 40,000 rows and one for testing which contained the remaining 10,000 rows. Before we ever began looking for the different predictors that might indicate churn versus not churning we took a look at some very basic plots. It is important to note that in the plot below while a large subset of the data is portrayed, there was one super user who had over 120 trips in their first thirty days. 

![](images/num_trips.png)

The second plot compares the average ratings for users given to drivers and the average ratings given to riders. Again here we see large clusterings around the highest three stars. 

![](images/ratings.png)



## EDA 

To get a better understanding of how each attribute of the dataset was affecting whether or not a customer churned, we seperated the data into two sets, churn and not_churn. From there we plotted overlaying histograms of the churn and not churn datasets for each attribute. This allowed us to easily identify which attributes were influencing churn. The below plots show five attributes that were heavily impacting churn. 


![](images/day_of_week.png)
![](images/luxury_call_users.png)
![](images/PhonevChurn.png)
![](images/feature_importance.png)




## Model Selection 
We decided to use a Random Forest Model because it is generalizable, easy to implement, and is a good place to start. Random Forest Models have proven to be successful in classification problems.

We determined that measuring recall would be most appropriate for this problem. Recall is a measure of how many positive instances we are able to identify as such. It's defined as (True Positives)/(True Positives + False Negatives). Since churning is our positive class, a false negative means that we think someone won't leave the service when in reality they will. This is a missed opportunity.

As stated above, the dataset had 12 features with some exhibiting collinearity. To focus on the features making the most impact on customer churn we decided to only include the five features shown in the above histograms. This helped prevent overfitting our model as well as making our model more interpretable. 

After reducing our number of features, we continued to rerun our model and we were seeing a modest increase in our recall score. Our first training recall score yielded a recall score of 84.62% and we were able to increase this to 85.04%. This modest increase can be explained by the way the Random Forest works, since it naturally eliminates correlation between features. Our decision to remove features was motivated by trying to make the model interpretable.

Additionally, we utilized a Grid Search which helped us determine the best hyper parameters for our model.

We compared our model to a decision tree and logistic regression, both of which yielded slightly worse results. The benefit of the Decision Tree is that it helps us intepret the analysis.

![](images/desision_tree.png)

## Results 
1. Recall for Training Set - 85.04% 
2. Recall on Test Set - 84.69%
3. Logistic Regression Recall on Training Set - 78.28%
4. Decision Tree Recall on Training Set - 81.9%

## Conclusion
In conclusion, we determined that a Random Forest Classification model yielded the best results for us to predict ride share user churn. Additionally, we noticed that the amount of trips taken in the first 30 days, the percent of weekday trips, luxury car rides, having an Android phone, and the percent of total rides having surge pricing to be important features.
Alternatively, if the user is more focused on interpretability of the model the decision tree yielded the best results. 
From a business use perspective, we would want to do a cost benefit matrix against our confusion matrix in order to determine if the estimated gain would offset the expense involved in targeting customers who may churn given our recall score.