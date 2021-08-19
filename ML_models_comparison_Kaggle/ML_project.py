"""
Machine Learning and Data Mining - COMP9417 - T2 2020
Project : Home Credit Default Risk
Url: https://www.kaggle.com/c/home-credit-default-risk

Team members:
    Matias Morales Armijo - z5216410
    Pablo Toro Pacheco - z5222810
    Sebastian Castillo Castro -  z5171921
"""

import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

TEST_DATASET_SIZE_FACTOR = 0.2  # percentaje (0 to 1) assigned to test dataset
K_FOLD_CV = 5                   # number of fold for cross validation
USE_GRID_SEARCH_PARAMS = 0      # 1: runs grid search + cv for search best params and train models.
                                # 0: uses best found params to train models.


##########################################################################
######################### HELPER FUNCTIONS ###############################
##########################################################################
"""
PCA function
"""
def pca_func(x_train_ds, x_test_ds, num_components):
    sc = StandardScaler()
    x_train_std = sc.fit_transform(x_train_ds)
    x_test_std = sc.transform(x_test_ds)
    pca = PCA(n_components=num_components)
    x_train_res = pca.fit_transform(x_train_std)
    x_test_res = pca.transform(x_test_std)
    return x_train_res, x_test_res


"""
Data cleansing for sub datasets
"""
def data_cleansing(x_train):
    x_train_num_cols = x_train.select_dtypes(include=['float64', 'int64'])
    x_train_nonnum_cols = x_train.select_dtypes(include=['object'])

    ### Replacing missing values with the median.
    for column in x_train_num_cols.columns:
        med = x_train_num_cols[column].median()
        x_train_num_cols[column] = x_train_num_cols[column].fillna(med)

    # For each column, get value counts in decreasing order and take the index (value) of most common class
    x_train_nonnum_cols = x_train_nonnum_cols.select_dtypes(include=['object']).apply(
        lambda x: x.fillna(x.value_counts().index[0]))

    # Outlier detection (Extreme Value Analysis)
    df_iqr = pd.DataFrame()

    for cols in x_train_num_cols.columns:
        #   if (cols!='SK_ID_CURR') | (cols!='TARGET') |(cols!='SK_ID_BUREAU'):
        if cols not in ['SK_ID_CURR', 'TARGET', 'SK_ID_BUREAU']:
            IQR = x_train_num_cols[cols].quantile(0.75) - x_train_num_cols[cols].quantile(0.25)
            upper_limit = x_train_num_cols[cols].quantile(0.75) + (IQR * 1.5)
            upper_limit_extreme = x_train_num_cols[cols].quantile(0.75) + (IQR * 3)
            lower_limit = x_train_num_cols[cols].quantile(0.25) - (IQR * 1.5)
            lower_limit_extreme = x_train_num_cols[cols].quantile(0.25) - (IQR * 3)
            df_iqr = df_iqr.append(
                {'Column': cols, 'IQR': '{0:.3f}'.format(IQR), 'upper_limit': '{0:.3f}'.format(upper_limit),
                 'upper_limit_extreme': '{0:.3f}'.format(upper_limit_extreme),
                 'lower_limit': '{0:.3f}'.format(lower_limit),
                 'lower_limit_extreme': '{0:.3f}'.format(lower_limit_extreme)
                 },
                ignore_index=True)
            x_train_num_cols.loc[x_train[cols] > upper_limit_extreme, cols] = upper_limit_extreme
            x_train_num_cols.loc[x_train[cols] < lower_limit_extreme, cols] = lower_limit_extreme

    res_ds = pd.concat([x_train_num_cols, x_train_nonnum_cols], axis=1)

    return res_ds, df_iqr

"""
Data cleansing for final base dataset (merged)
"""
def data_cleansing2(app_train):
    x_train_num_cols = app_train.select_dtypes(include=['float64', 'int64'])
    x_train_nonnum_cols = app_train.select_dtypes(include=['object'])

    # Replacing missing values with the median.
    for column in x_train_num_cols.columns:
        med = x_train_num_cols[column].median()
        x_train_num_cols[column] = x_train_num_cols[column].fillna(med)

    # for each column, get value counts in decreasing order and take the index (value) of most common class
    x_train_nonnum_cols = x_train_nonnum_cols.select_dtypes(include=['object']).apply(
        lambda x: x.fillna(x.value_counts().index[0]))

    res_dataset = pd.concat([x_train_num_cols, x_train_nonnum_cols], axis=1)
    return res_dataset


"""
Row elimination function
"""
def row_elimination(x_train):
    df = x_train.copy()

    # first create missing indicator for features with missing data
    for col in df.columns:
        missing = df[col].isnull()
        num_missing = np.sum(missing)

        if num_missing > 0:
            df['{}_ismissing'.format(col)] = missing

    # then based on the indicator, plot the histogram of missing values
    ismissing_cols = [col for col in df.columns if 'ismissing' in col]
    df['num_missing'] = df[ismissing_cols].sum(axis=1)

    # Solution 1: Drop the Observation
    # drop rows with a lot of missing values.
    ind_missing = df[df['num_missing'] > 50].index
    df_less_missing_rows = df.drop(ind_missing, axis=0)

    return df_less_missing_rows


"""
Feature creation function
It creates polynomial features for the most correlated columns with the target  
"""
def feature_eng(x_train, x_test, y_train):
    train_sk_id_curr = x_train['SK_ID_CURR']
    test_sk_id_curr = x_test['SK_ID_CURR']
    x_train = x_train.drop(columns=['SK_ID_CURR'])
    x_test = x_test.drop(columns=['SK_ID_CURR'])
    most_correlated_features = ['bureau_DAYS_CREDIT_min', 'DAYS_BIRTH', 'bureau_DAYS_CREDIT_mean', 'EXT_SOURCE_3',
                                'EXT_SOURCE_2', 'EXT_SOURCE_1']

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = MinMaxScaler(feature_range=(0, 1))

    x_train = pd.DataFrame(imputer.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(imputer.transform(x_test), index=x_test.index, columns=x_test.columns)

    x_train = pd.DataFrame(scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

    # Creating polynomial features
    x_train['TARGET'] = y_train
    x_train.insert(0, "SK_ID_CURR", train_sk_id_curr)
    x_test.insert(0, "SK_ID_CURR", test_sk_id_curr)
    most_correlated_features.append('TARGET')
    x_train_poly = x_train[most_correlated_features]
    most_correlated_features.remove('TARGET')
    x_test_poly = x_test[most_correlated_features]
    x_train_poly = x_train_poly.drop(columns=['TARGET'])

    # Create the polynomial object with specified degree
    poly_transformer = PolynomialFeatures(degree=3)

    # Transform the features
    x_train_poly = pd.DataFrame(poly_transformer.fit_transform(x_train_poly),
                                       columns=poly_transformer.get_feature_names(most_correlated_features),
                                       index=x_train_poly.index)

    x_test_poly = pd.DataFrame(poly_transformer.transform(x_test_poly),
                                      columns=poly_transformer.get_feature_names(most_correlated_features),
                                      index=x_test_poly.index)

    # Merge polynomial features into training dataframe
    x_train_poly['SK_ID_CURR'] = x_train['SK_ID_CURR']
    x_train_poly = x_train_poly.drop(columns=most_correlated_features)
    x_train = x_train.merge(x_train_poly, on='SK_ID_CURR', how='left').set_index(x_train.index)

    # Merge polynomial features into testing dataframe
    x_test_poly['SK_ID_CURR'] = x_test['SK_ID_CURR']
    x_test_poly = x_test_poly.drop(columns=most_correlated_features)
    x_test = x_test.merge(x_test_poly, on='SK_ID_CURR', how='left').set_index(x_test.index)

    # Align the dataframes
    x_train, x_test = x_train.align(x_test, join='inner', axis=1)

    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    scaler = MinMaxScaler(feature_range=(0, 1))

    x_train = x_train.drop(columns=['SK_ID_CURR'])
    x_test = x_test.drop(columns=['SK_ID_CURR'])

    # Replacing null values with mean values and Normalizing data bewtween 0 and 1
    x_train = pd.DataFrame(imputer.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(imputer.transform(x_test), index=x_test.index, columns=x_test.columns)

    # Scaling data between 0 and 1
    x_train = pd.DataFrame(scaler.fit_transform(x_train), index=x_train.index, columns=x_train.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), index=x_test.index, columns=x_test.columns)

    return x_train, x_test


""" 
Function that creates columns based on statistical calculations
Name of each column : <dataframe_name>_<original_column_name>_<aggregation_function>. E.g. bureau_DAYS_CREDIT_count
"""
def create_statistical_columns(dataframe, df_name, groupby_column):
    agg_funcs = ['count', 'mean', 'sum', 'min', 'max']

    # creating dataframe base with single column the "groupby column"
    resulting_df = pd.DataFrame(data=dataframe[groupby_column].unique(), columns=[groupby_column])

    # going through each column to create the statistical columns
    dataframe = dataframe.select_dtypes(include=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'])
    for col in dataframe.columns:
        if not col.startswith("SK_ID"):
            grouped_df = dataframe.groupby(by=[groupby_column], as_index=False).agg({col: agg_funcs})
            col = df_name + '_' + col
            grouped_df.columns = [groupby_column, col + '_count', col + '_mean', col + '_sum', col + '_min',
                                  col + '_max']
            resulting_df = resulting_df.merge(grouped_df, on=groupby_column, how='left')
    return resulting_df

##########################################################################
###################### GENERATING BASE DATASET ###########################
##########################################################################

# Reading DataSets
app_dataset = pd.read_csv('application_train.csv')
bureau_dataset = pd.read_csv('bureau.csv')

# Data cleansing for dataset Applications
app_dataset, _ = data_cleansing(app_dataset)

# Creating statistical columns for Bureau dataset
bureau_stat_cols = create_statistical_columns(bureau_dataset, "bureau", "SK_ID_CURR")

# Data cleansing for dataset Bureau
bureau_dataset, _ = data_cleansing(bureau_stat_cols)
base_ds = app_dataset.merge(bureau_dataset, on='SK_ID_CURR', how='left')

# Data cleansing for dataset base (Application + Bureau)
base_ds = data_cleansing2(base_ds)

y = base_ds['TARGET']
x = base_ds.drop(columns=['TARGET'])

# Splitting data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_DATASET_SIZE_FACTOR, random_state=0)

# Creating a label encoder object for non numerical columns with only 2 classes
le = LabelEncoder()
for col in x_train:
    if x_train[col].dtype == 'object':
        if len(list(x_train[col].unique())) <= 2:
            le.fit(x_train[col])
            # Transform both training and testing data
            x_train[col] = le.transform(x_train[col])
            x_test[col] = le.transform(x_test[col])

# Applying One-hot encoding for categorical variables
x_train = pd.get_dummies(x_train)
x_test = pd.get_dummies(x_test)

# Aligning training and testing data, keeping only columns present in both dataframes
x_train, x_test = x_train.align(x_test, join='inner', axis=1)



##########################################################################
####################### MACHINE LEARNING MODELS ##########################
##########################################################################
time_start=time.time()

# Logistic Regression - Baseline
time_temp = time.time()
print('--- Baseline - Logistic Regression ---')
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)
predicted_probas = log_reg.predict_proba(x_test)[:, 1]
print('ROC_AUC test:', roc_auc_score(y_test, predicted_probas))
print('--- Checking overfitting ---')
predicted_T_probas= log_reg.predict_proba(x_train)[:,1]
print('ROC_AUC train:',roc_auc_score(y_train, predicted_T_probas))
print("Time (minutes): %.2f" % ((time.time()-time_temp)/60))
print()
print()


# Logistic Regression + PCA
x_train_PCA = x_train.copy()
x_test_PCA = x_test.copy()
time_temp = time.time()
x_train_pca, x_test_pca = pca_func(x_train_PCA,x_test_PCA, 170)
print('--- Logistic_regression + PCA (170 components) ---')
log_reg = LogisticRegression()
log_reg.fit(x_train_pca, y_train)
predicted_probas = log_reg.predict_proba(x_test_pca)[:, 1]
print('ROC_AUC test:', roc_auc_score(y_test, predicted_probas))
print('--- Checking overfitting ---')
predicted_T_probas= log_reg.predict_proba(x_train_pca)[:,1]
print('ROC_AUC train:',roc_auc_score(y_train, predicted_T_probas))
print("Time (minutes): %.2f" % ((time.time()-time_temp)/60))
print()
print()


# Logistic Regression + Feature Engineering
x_train_FE = x_train.copy()
x_test_FE = x_test.copy()
time_temp = time.time()

x_train_FE, x_test_FE = feature_eng(x_train_FE, x_test_FE, y_train)

print(f'SHAPEEE {x_train_FE.shape}')
print('--- Logistic_regression + Feature Engineering ---')
log_reg = LogisticRegression()
log_reg.fit(x_train_FE, y_train)
predicted_probas = log_reg.predict_proba(x_test_FE)[:, 1]
print('ROC_AUC train: ', roc_auc_score(y_test, predicted_probas))
print('--- Checking overfitting ---')
predicted_T_probas= log_reg.predict_proba(x_train_FE)[:,1]
print('ROC_AUC test: ',roc_auc_score(y_train, predicted_T_probas))
print("time (minutes): %.2f" % ((time.time()-time_temp)/60))
print()
print()


# Logistic Regression + Feature Engineering + PCA
time_temp = time.time()
x_train_FE_PCA, x_test_FE_PCA = pca_func(x_train_FE, x_test_FE, 212)
print('---Logistic_regression + Feature Engineering + PCA (212 components) ---')
log_reg = LogisticRegression()
log_reg.fit(x_train_FE_PCA, y_train)
predicted_probas = log_reg.predict_proba(x_test_FE_PCA)[:, 1]
print('ROC_AUC train:', roc_auc_score(y_test, predicted_probas))
print('--- Checking overfitting ---')
predicted_T_probas= log_reg.predict_proba(x_train_FE_PCA)[:,1]
print('ROC_AUC test:',roc_auc_score(y_train, predicted_T_probas))
print("time (minutes): %.2f" % ((time.time()-time_temp)/60))
print()
print()

if (USE_GRID_SEARCH_PARAMS):
    print('--- Decision Tree ---')
    # Creating the dictionary to make GridSearch
    param_grid_tree = {'min_samples_leaf': [0.01,0.05,0.1], 'max_features':["sqrt",0.9,0.8,0.7]}
    grid_tree_clf = GridSearchCV(tree.DecisionTreeClassifier(criterion='entropy',class_weight='balanced',random_state=0),
                                  param_grid=param_grid_tree,cv=K_FOLD_CV,scoring='roc_auc',refit=True)
    best_tree=grid_tree_clf.fit(x_train, y_train)
    print('--- Best params estimator ---')
    best_params_tree = best_tree.best_estimator_.get_params()
    print('max_features:',best_params_tree['max_features'],'min_samples_leaf:',best_params_tree['min_samples_leaf'])
    predicted_probas= best_tree.predict_proba(x_test)[:,1]
    print('ROC_AUC test: ',roc_auc_score(y_test, predicted_probas))
    print('--- Checking overfitting ---')
    predicted_T_probas= best_tree.predict_proba(x_train)[:,1]
    print('ROC_AUC train: ',roc_auc_score(y_train, predicted_T_probas))
    print()
    print()


    # Random Forest: Using the hyperparams obtained in the Decision Tree above
    print('--- Random Forest ---')
    # Creating the dictionary to make GridSearch
    param_grid_RF={'n_estimators':[50,100,150]}
    grid_RF_clf= GridSearchCV(RandomForestClassifier(criterion='entropy',class_weight='balanced_subsample',random_state=0,
                                                min_samples_leaf=best_params_tree['min_samples_leaf'],
                                                max_features=best_params_tree['max_features']),
                              param_grid=param_grid_RF,cv=K_FOLD_CV,scoring='roc_auc',refit=True)
    best_RF=grid_RF_clf.fit(x_train,y_train)
    print('--- Best params estimator ---')
    best_params_RF = best_RF.best_estimator_.get_params()
    print('n_estimators:',best_params_RF['n_estimators'])
    predicted_probas= best_RF.predict_proba(x_test)[:,1]
    print('ROC_AUC test: ',roc_auc_score(y_test, predicted_probas))
    print('--- Checking overfitting ---')
    predicted_T_probas= best_RF.predict_proba(x_train)[:,1]
    print('ROC_AUC train: ',roc_auc_score(y_train, predicted_T_probas))
    print()
    print()


    print('--- AdaBoost ---')
    # Creating the dictionary to make GridSearch
    param_grid_Ada={'n_estimators':[5,10,25,50,100],'learning_rate':[0.5,1.0,1.5]}
    grid_Ada_clf= GridSearchCV(AdaBoostClassifier(random_state=0), param_grid=param_grid_Ada,
                                                 cv=K_FOLD_CV,scoring='roc_auc',refit=True)
    best_Ada=grid_Ada_clf.fit(x_train,y_train)
    print('--- Best params estimator ---')
    best_params_Ada = best_Ada.best_estimator_.get_params()
    print('n_estimators:',best_params_Ada['n_estimators'],
          'learning_rate:',best_params_Ada['learning_rate'])
    predicted_probas= best_Ada.predict_proba(x_test)[:,1]
    print('ROC_AUC test: ',roc_auc_score(y_test, predicted_probas))
    print('--- Checking overfitting ---')
    predicted_T_probas= best_Ada.predict_proba(x_train)[:,1]
    print('ROC_AUC train:',roc_auc_score(y_train, predicted_T_probas))
    print()
    print()


    print('--- Stacking (RF + ADA + LG)---')
    estimators=[('RF',RandomForestClassifier(criterion='entropy',class_weight='balanced_subsample',random_state=0,
                                                min_samples_leaf=best_params_tree['min_samples_leaf'],
                                                max_features=best_params_tree['max_features'],
                                                n_estimators=best_params_RF['n_estimators'])),
                ('Ada',AdaBoostClassifier(n_estimators=best_params_Ada['n_estimators'],
                                          learning_rate=best_params_Ada['learning_rate'], random_state=0))]
    final_clf = StackingClassifier(estimators=estimators,
                                   final_estimator=LogisticRegression(random_state=0,
                                                                      class_weight='balance',
                                                                      penalty='l2',C=1.0),
                                    cv=K_FOLD_CV, stack_method='predict_proba')
    stack_model=final_clf.fit(x_train,y_train)
    predicted_probas= stack_model.predict_proba(x_test)[:,1]
    print('ROC_AUC test:',roc_auc_score(y_test, predicted_probas))
    print('--- Checking overfitting ---')
    predicted_T_probas= stack_model.predict_proba(x_train)[:,1]
    print('ROC_AUC train:',roc_auc_score(y_train, predicted_T_probas))
    print()
    print()

    print('All the process took: ',(time.time()-time_start)/60,' minutes')



else:
    # Training algorithms using optimal found parameters

    print('--- Decision Tree ---')
    
    print('--- Best params estimator ---')
    print('max_features:', 0.8, 'min_samples_leaf:', 0.01)
    clf=tree.DecisionTreeClassifier(criterion='entropy',class_weight='balanced',random_state=0,
                                min_samples_leaf=0.01,max_features=0.8)
    model=clf.fir(x_train,y_train)
    predicted_probas = model.predict_proba(x_test)[:, 1]
    print('ROC_AUC test: ', roc_auc_score(y_test, predicted_probas))
    print('--- Checking overfitting ---')
    predicted_T_probas = model.predict_proba(x_train)[:, 1]
    print('ROC_AUC train: ', roc_auc_score(y_train, predicted_T_probas))
    print()
    print()

    # Random Forest: Using the hyperparams obtained in the Decision Tree above
    print('--- Random Forest ---')
    print('--- Best params estimator ---')
    print('n_estimators:', 150)
    clf=RandomForestClassifier(n_estimators=150,criterion='entropy',class_weight='balanced_subsample',random_state=0,
                                            min_samples_leaf=0.01,
                                            max_features=0.8)
    model=clf.fir(x_train,y_train)
    predicted_probas = model.predict_proba(x_test)[:, 1]
    print('ROC_AUC test: ', roc_auc_score(y_test, predicted_probas))
    print('--- Checking overfitting ---')
    predicted_T_probas = model.predict_proba(x_train)[:, 1]
    print('ROC_AUC train: ', roc_auc_score(y_train, predicted_T_probas))
    print()
    print()

    print('--- AdaBoost ---')
    print('--- Best params estimator ---')
    print('n_estimators:', 100,
          'learning_rate:', 1)
    clf= AdaBoostClassifier(random_state=0,n_estimators=100, learning_rate=1)
    model=clf.fir(x_train,y_train)
    predicted_probas = model.predict_proba(x_test)[:, 1]
    print('ROC_AUC test: ', roc_auc_score(y_test, predicted_probas))
    print('--- Checking overfitting ---')
    predicted_T_probas = model.predict_proba(x_train)[:, 1]
    print('ROC_AUC train:', roc_auc_score(y_train, predicted_T_probas))
    print()
    print()

    print('--- Stacking (RF + ADA + LG)---')
    estimators=[('RF',RandomForestClassifier(criterion='entropy',class_weight='balanced_subsample',random_state=0,
                                            min_samples_leaf=0.01,
                                            max_features=0.8,
                                            n_estimators=150)),
            ('Ada',AdaBoostClassifier(n_estimators=100,
                                      learning_rate=1, random_state=0))]
    final_clf = StackingClassifier(estimators=estimators,
                                   final_estimator=LogisticRegression(random_state=0,
                                                                      class_weight='balance',
                                                                      penalty='l2', C=1.0),
                                   cv=K_FOLD_CV, stack_method='predict_proba')
    stack_model = final_clf.fit(x_train, y_train)
    predicted_probas = stack_model.predict_proba(x_test)[:, 1]
    print('ROC_AUC test:', roc_auc_score(y_test, predicted_probas))
    print('--- Checking overfitting ---')
    predicted_T_probas = stack_model.predict_proba(x_train)[:, 1]
    print('ROC_AUC train:', roc_auc_score(y_train, predicted_T_probas))
    print()
    print()

    print('All the process took: ', (time.time() - time_start) / 60, ' minutes')