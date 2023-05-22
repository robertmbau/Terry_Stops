# relevant imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
plt.style.use('seaborn')
import warnings
warnings.filterwarnings("ignore")


# sklearn imports
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, recall_score, roc_curve, auc, confusion_matrix,
                             classification_report, mean_squared_error, precision_score, ConfusionMatrixDisplay)
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, RFE, RFECV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.svm import SVC



# function to show column details
def info(dataframe):
    info_df = pd.DataFrame(columns=['Column', 'Missing Percentage', 'Missing Values', 'Length', 'Data type'])
    for col in dataframe.columns:
        percent_missing = round((dataframe[col].isna().sum() / len(dataframe[col])) * 100, 2)
        missing = dataframe[col].isna().sum()
        data_type = dataframe[col].dtype
        total_len = len(dataframe[col])
        info_df = info_df.append({'Column': col, 'Missing Percentage': percent_missing, 'Length':total_len,
                                  'Missing Values': missing, 'Data type': data_type}, ignore_index=True)
    info_df = info_df.sort_values(by='Missing Percentage', ascending=False)
    return info_df


# function to replace a value
def replace(dataframe, column, og_value, rep_value):
  '''
  dataframe - dataframe name
  column - column name
  og_value - value to be replaced
  rep_value - replacing value '''

  dataframe[column].replace(to_replace=og_value, value=rep_value, inplace=True)

  return dataframe[column].value_counts()


# function for dropping columns
def drop(df, col):
  '''
  df - dataframe name
  col - column name
  '''
  print(f'Number of columns before dropping: {df.shape[1]}')
  df.drop(col, axis=1, inplace=True)
  return print(f'Number of columns after dropping: {df.shape[1]}')


  # function to round off time
def round_time(col):
    hours, minutes = col.split(':')

    hours = int(hours)
    minutes = int(minutes)

    if minutes > 30:
        hours += 1
        minutes = 0

    rounded_col = f'{hours:02d}'

    return rounded_col


# function to create age groups
def map_age(x):
    age_groups = {
        (18, 25): '21 - 25',
        (26, 35): '26 - 35',
        (36, 45): '36 - 45',
        (46, 55): '46 - 55'
    }
    
    for age_range, age_group in age_groups.items():
        if age_range[0] <= x <= age_range[1]:
            return age_group
    
    return '56 and Above'


# function to group time
def map_time(x):
    time_groups = {
        ('00', '04'): 'After Midnight',
        ('05', '08'): 'Early Morning',
        ('09', '12'): 'Late Morning',
        ('13', '18'): 'Afternoon',
        ('19', '21'): 'Night'
    }
    
    for time_range, time_group in time_groups.items():
        if time_range[0] <= x <= time_range[1]:
            return time_group
    
    return 'Late Night'



# Function to assign a group to an incident
def call_group(incident):
    groups = {
    'Burglary_Theft': ['BURGLARY', 'THEFT', 'FRAUD', 'BURG'],
    'Assault': ['ASSAULTS', 'FIGHT', 'ASLT'],
    'Narcotics': ['NARCOTICS'],
    'Property': ['PROPERTY', 'DAMAGE'],
    'Traffic': ['TRAFFIC', 'AUTOMOBILES'],
    'Disturbances': ['SUSPICIOUS', 'DISTURBANCE', 'NOISE', 'HAZARDS'],
    'Domestic_Violence': ['DV'],
    'Sexual_Offenses': ['SEX OFFENSES', 'VICE', 'LEWD'],
    'Emergency Calls': ['ALARM'],
    'Miscellaneous': ['UNKNOWN', 'ASSIST', 'INTOX', 'LIQ VIOLS', 'CHILD', 'PERSON']
}
    for group, keywords in groups.items():
        if any(keyword in incident for keyword in keywords):
            return group
    return 'Unknown'


# Function to assign a group to an incident
def squad_groups(incident):
    squad_group = {
    'NORTH PCT': ['NORTH PCT'],
    'SOUTH PCT': ['SOUTH', 'SOUTHWEST'],
    'TRAINING': ['TRAINING'],
    'WEST PCT': ['WEST PCT'],
    'EAST PCT': ['EAST PCT'],
    'SWAT': ['SWAT', 'GUN', 'CCI', 'OPS'],
    'HUMAN_TRAFFICKING': ['HUMAN'],
    'TRAFFIC': ['TRAF'],
    'THEFT': ['BURG'],
    'OTHER': ['CANINE', 'HARBOR', 'CRISIS', 'NARC', 'HR', 'DV', 'ALTERNATIVE', 'COMMUNITY']
}
    for group, keywords in squad_group.items():
        if any(keyword in incident for keyword in keywords):
            return group
    return 'UNKNOWN'



# function  to calculate scores
def calc_scores(y_train, y_pred_train, y_test, y_pred_test):
    '''
    y_train: True target values of the training set.
    y_pred_train: Predicted target values for the training set.
    y_test: True target values of the test set.
    y_pred_test: Predicted target values for the test set.
    '''
    f1_train = f1_score(y_train, y_pred_train)
    f1_test = f1_score(y_test, y_pred_test)

    recall_train = recall_score(y_train, y_pred_train)
    recall_test = recall_score(y_test, y_pred_test)

    precision_train = precision_score(y_train, y_pred_train)
    precision_test = precision_score(y_test, y_pred_test)

    print(f'F1 score for training data: {f1_train}')
    print(f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
    print(f'F1 score for testing data: {f1_test}')
    print(f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
    print(f'Recall score for training data: {recall_train}')
    print(f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
    print(f'Recall score for testing data: {recall_test}')
    print(f'+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
    print(f'Precision score for training data: {precision_train}')
    print('+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+')
    print(f'Precision score for testing data: {precision_test}')


# function to plot confusion matrix
def plot_cm(y_train, y_pred_train, y_test, y_pred_test):
    '''
    y_train: True target values of the training set.
    y_pred_train: Predicted target values for the training set.
    y_test: True target values of the test set.
    y_pred_test: Predicted target values for the test set.
    '''
    label_dict = {0: 'No Arrest', 1: 'Arrest'}
    cf = confusion_matrix(y_train, y_pred_train)
    cf2 = confusion_matrix(y_test, y_pred_test)

    labels = [label_dict[key] for key in sorted(label_dict.keys())]
    disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=labels)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cf2, display_labels=labels)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

    disp.plot(ax=ax1)
    ax1.grid(False)

    disp2.plot(ax=ax2)
    ax2.grid(False)
    plt.subplots_adjust(wspace=0.7)

    plt.show()


def plot_roc(X_train, y_train, X_pred, y_pred, model):
    '''
    X_train - training set of X
    y_train - training set of y
    X_pred - predicted X
    y_pred - predicted y
    model - model object with a predict_proba() method
    '''
    y_score = model.fit(X_train, y_train).predict_proba(X_pred)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_pred, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.yticks([i/10.0 for i in range(11)])
    plt.xticks([i/10.0 for i in range(11)])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()
    print(f'AUC: {roc_auc}')
