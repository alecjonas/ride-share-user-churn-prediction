import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree

def clean_data(path):
    df = pd.read_csv(path)
    df.dropna(inplace=True)
    phone_df = pd.get_dummies(df['phone'])
    city_df = pd.get_dummies(df['city'])
    df['Astapor'] = city_df['Astapor']
    df["King's Landing"] = city_df["King's Landing"]
    df['Winterfell'] = city_df['Winterfell']
    df['Android'] = phone_df['Android']
    df['iPhone'] = phone_df['iPhone']
    df.drop('phone',axis=1,inplace=True)
    df.drop('city',axis=1,inplace=True)
    df['signup_date'] = pd.to_datetime(df['signup_date'])
    df['day_of_week'] = df['signup_date'].apply(lambda x: x.dayofweek)
    df['signup_weekend'] = df['day_of_week'].apply(lambda x: x > 4)
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['churn'] = df['last_trip_date'].apply(lambda x: x < pd.to_datetime('2014-06-01')).astype(int)
    df['luxury_car_user'] = df['luxury_car_user']*1
    df['signup_weekend'] = df['signup_weekend']*1
    df['avg_total_rating'] = df['avg_rating_by_driver'] + df['avg_rating_of_driver']
    #df.drop(columns = ['Android', 'Astapor', 'last_trip_date', 'signup_date', 'day_of_week'],inplace=True)
    return df

def plot_hists(df):
    for column in df.columns:
        df_no_churn = df[[column,'churn']].loc[df['churn'] == 0]
        df_churn = df[[column, 'churn']].loc[df['churn'] == 1]
        fig, ax = plt.subplots()
        try:
            bins = np.linspace(df[column].min(),df[column].max(),15)
            ax.hist(df_no_churn[column], label = 'no_churn', density = True,bins = bins)
            ax.hist(df_churn[column], label = 'churn', alpha = .5, density = True, bins = bins)
            ax.legend()    
            ax.set_title(column)
            plt.show()
        except:
            ax.hist(df_no_churn[column], label = 'no_churn', density = True)
            ax.hist(df_churn[column], label = 'churn', alpha = .5, density = True)
            ax.legend()    
            ax.set_title(column)
            plt.close
            #plt.show()

if __name__=='__main__':
    df = clean_data('ride-share/data/churn_train.csv')
    plt.close

    # EDA Plots
    # data_to_plot = [df['trips_in_first_30_days']]
    # fig,ax =plt.subplots(figsize = (12,7))
    # ax.hist(data_to_plot,bins=40)
    # ax.set_xlabel('Trips In First 30 Days')
    # ax.set_title('Number Of Trips')
    # plt.xlim(0,40)
    # plt.savefig('images/num_trips.png')

    # #EDA 
    # data_to_plot = [df['avg_rating_by_driver'], df['avg_rating_of_driver']]
    # fig,ax = plt.subplots(figsize=(12,7))
    # ax.set_title('Ratings')
    # ax.set_ylabel('Average Number of Stars')
    # ax.violinplot(data_to_plot)
    # plt.xticks(np.arange(1,3),['Rating By Driver Of Rider','Rating Of Driver By Rider'])
    # plt.savefig('images/ratings.png')

    #POTENTIAL TRAIN TEST SPLIT CROSS VALIDATION#

    # y = df['churn']
    # X = df.drop(columns = ['churn'])

    # #modeling 
    # rf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
    #                    max_depth=None, max_features='sqrt', max_leaf_nodes=None,
    #                    min_impurity_decrease=0.0, min_impurity_split=None,
    #                    min_samples_leaf=4, min_samples_split=2,
    #                    min_weight_fraction_leaf=0.0, n_estimators=40,
    #                    n_jobs=None, oob_score=False, random_state=1, verbose=0,
    #                    warm_start=False)
    # rf.fit(X, y)
    # print("Accuracy Score:", rf.score(X, y))
    # print("Out of Bag Score:", rf.oob_score_)


    # #gridsearch
    # random_forest_grid = {'max_depth': [3, None],
    #                     'max_features': ['sqrt', 'log2', None],
    #                     'min_samples_split': [2, 4],
    #                     'min_samples_leaf': [1, 2, 4],
    #                     'bootstrap': [True, False],
    #                     'n_estimators': [10, 20, 40, 80],
    #                     'random_state': [1]}

    # model_gridsearch = GridSearchCV(RandomForestClassifier(),
    #                                 random_forest_grid,
    #                                 n_jobs=-1,
    #                                 verbose=True,
    #                                 scoring='neg_mean_squared_error')
    # model_gridsearch.fit(X, y)
    # best_params = model_gridsearch.best_params_ 
    # model_best = model_gridsearch.best_estimator_
    # model_best.score(X, y)
    



    #Feature Importance
    #Partial Dependence Plots

y = df['churn']
#X= df.drop(columns = ['churn'])
X=df[['trips_in_first_30_days','weekday_pct','luxury_car_user','Android','surge_pct']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def rfc_model(X_train, y_train, X_test, y_test):
    rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=40,
                       n_jobs=None, oob_score=True, random_state=1, verbose=0,
                       warm_start=False)
    rfc.fit(X_train, y_train)
    
    y_preds = rfc.predict(X_test)
    recall = recall_score(y_test, y_preds)
    score = rfc.score(X_train, y_train)
    oob = rfc.oob_score_
    
    return (f'recall = {recall}', f'Out of Bag = {oob}')

def rfc_feature_importance(X_train, y_train, X_test, y_test):
    
    rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=40,
                       n_jobs=None, oob_score=True, random_state=1, verbose=0,
                       warm_start=False)
    rfc.fit(X_train, y_train)
    
    #Feature Importance
    importances = rfc.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    print('------------------------------')
    print('Feature headers:')

    headers = list(X.columns.values)
    for idx, val in enumerate(headers):
        print(f'feature {idx}:', val)

    # Plot the feature importances of the forest
    plt.figure(figsize = (12,9))
    plt.title("Feature importances", size=24)
    plt.bar(range(X.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), headers, rotation=25,ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.close
    plt.savefig('images/feature_importance.png')


print(rfc_feature_importance(X_train, y_train, X_test, y_test))


rfc = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='sqrt', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=4, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=40,
                       n_jobs=None, oob_score=True, random_state=1, verbose=0,
                       warm_start=False)
rfc.fit(X_train, y_train)


df2 = clean_data('ride-share/data/churn_test.csv')
y2 = df2['churn']
X2 = df2[['trips_in_first_30_days','weekday_pct','luxury_car_user','Android','surge_pct']]
y_final_preds = rfc.predict(X2)
final_recall = recall_score(y2, y_final_preds)
#print(final_recall)

def decision_tree(X_train,y_train,X_test,y_test):
    model = DecisionTreeClassifier(max_depth = 5)
    model.fit(X_train,y_train)
    y_hat = model.predict(X_test)
    recall = recall_score(y_test,y_hat)
    return model,y_hat,recall 

model,y_hat,recall = decision_tree(X_train,y_train,X_test,y_test)
print(recall)
plt.figure(figsize = (12,12))
tree.plot_tree(model)
plt.savefig('decision_tree.png')

# lr = LogisticRegression(random_state=0).fit(X_train, y_train)
# lr_preds = lr.predict(X_test)
# lr_recall = recall_score(y_test, lr_preds)
# Recall is .783