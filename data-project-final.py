import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.metrics import mean_squared_error

import random
import numpy
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.head()

# Columns that will be droped and will not be included in the calculations
ignore_list=['Year_Factor','State_Factor','building_class'
             ,'facility_type','floor_area','year_built',
             'direction_max_wind_speed','direction_peak_wind_speed'
             ,'max_wind_speed','days_with_fog','site_eui','id']
target=train_df['site_eui']
no_usar=['site_eui','id']

#replacing rest of the values with mean
train_df['year_built'] = train_df['year_built'].fillna(2013)
train_df['energy_star_rating'] = train_df['energy_star_rating'].fillna(train_df['energy_star_rating'].mean())
train_df['direction_max_wind_speed'] = train_df['direction_max_wind_speed'].fillna(train_df['direction_max_wind_speed'].mean())
train_df['direction_peak_wind_speed'] = train_df['direction_peak_wind_speed'].fillna(train_df['direction_peak_wind_speed'].mean())
train_df['max_wind_speed'] = train_df['max_wind_speed'].fillna(train_df['max_wind_speed'].mean())
train_df['days_with_fog'] = train_df['days_with_fog'].fillna(train_df['days_with_fog'].mean())

##for test_dfdata

# year_built: replace with current year.
test_df['year_built'] = test_df['year_built'].fillna(2013)
#replacing rest of the values with mean
test_df['energy_star_rating'] = test_df['energy_star_rating'].fillna(test_df['energy_star_rating'].mean())
test_df['direction_max_wind_speed'] = test_df['direction_max_wind_speed'].fillna(test_df['direction_max_wind_speed'].mean())
test_df['direction_peak_wind_speed'] = test_df['direction_peak_wind_speed'].fillna(test_df['direction_peak_wind_speed'].mean())
test_df['max_wind_speed'] = test_df['max_wind_speed'].fillna(test_df['max_wind_speed'].mean())
test_df['days_with_fog'] = test_df['days_with_fog'].fillna(test_df['days_with_fog'].mean())

for i in train_df.columns:
    if i in ignore_list:
         train_df.drop(i, axis=1, inplace=True)




train_df.shape
(75757, 64)
train_df.info()

train_df.describe()


train_df.dtypes.value_counts()

cat_features = train_df.select_dtypes(include=['object']).columns.to_list()
cat_features
['State_Factor', 'building_class', 'facility_type']
num_features = [i for i in train_df.columns if i not in cat_features]


cat_df = train_df.loc[:, cat_features]
cat_df['site_eui'] = train_df['site_eui']
cat_df


for i in cat_df.columns:
    if i != 'site_eui':
        print(i.upper())
        print(cat_df[i].value_counts(), '\n')





def cat_plots(df):
    for i in cat_features:
        plt.figure(figsize=(20,5))
        sns.countplot(df[i])
        plt.xticks(rotation=90)
        plt.show()

cat_plots(train_df)





cat_plots(test_df)




def missing(df, hex1, hex2, text):
    plt.figure(figsize=(15,5))
    sns.heatmap(df[cat_features].isna().values, cmap = [hex1, hex2], xticklabels=cat_features)
    plt.title('Missing values in {}'.format(text), fontsize=20)
    plt.show()
missing(train_df, '#ffff99', '#009933', 'Training Set')

missing(test_df, '#ffccff',  '#66ccff', 'Test Set')








from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler,OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

def encoder(train, test):
    le = LabelEncoder()
    for col in train.columns:
        if train[col].dtypes == 'object':
            train[col] = le.fit_transform(train[col])
            le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
            test[col] = test[col].apply(lambda x: le_dict.get(x, -1))
    return train, test
train_encoded, test_encoded = encoder(train_df, test_df)
train_encoded


test_encoded


final_train = train_encoded.copy()
final_test = test_encoded.copy()

num_df = train_df[num_features]
num_df['site_eui'] = train_df['site_eui']


corr_dict = num_df.corr(method='spearman').loc[:,'site_eui'].to_dict()

num_df.head()


for i, val in corr_dict.items():
    if i != 'id':
        if (val < 0.3 and val>0.0) or (val > -0.3 and val<0):
            final_train.drop(i, axis=1, inplace=True)
            final_test.drop(i, axis=1, inplace=True)
final_train.head()

final_test.head()

final = final_train.drop('id', axis=1)
sns.heatmap(final.corr())



final_train.isnull().sum()

final_train.info()

plt.figure(figsize=(15,5))
sns.heatmap(final_train.isna().values, cmap = ['#ffff99', '#009933'], xticklabels=final_train.columns)
plt.title('Missing values in {}'.format('Final Training Set'), fontsize=20)
plt.show()

plt.figure(figsize=(15,5))
sns.heatmap(final_test.isna().values, cmap = ['#99ccff', '#ff99ff'], xticklabels=final_test.columns)
plt.title('Missing values in {}'.format('Final Test Set'), fontsize=20)
plt.show()


final_train['energy_star_rating'].describe()

sns.boxplot(final_train['energy_star_rating'], color = 'lightgreen')



final_train1 = final_train.copy()
final_train1['energy_star_rating']=final_train1['energy_star_rating'].fillna(final_train1['energy_star_rating'].mean())
final_train1.isnull().sum()

final_test1 = final_test.copy()
final_test1['energy_star_rating']=final_test1['energy_star_rating'].fillna(final_test1['energy_star_rating'].mean())
final_test1.isnull().sum()

scaler = StandardScaler()
num_feat = ['energy_star_rating']
final_train1[num_feat] = scaler.fit_transform(final_train1[num_feat])
final_test1[num_feat] = scaler.transform(final_test1[num_feat])
final_train1.head()

final_test1.head()


final_train1['site_eui'].hist()
plt.show()



plt.hist(np.log(final_train1['site_eui']), color='green');
plt.title('Log Transformation', fontsize=20)
plt.show()

plt.hist(np.sqrt(final_train1['site_eui']), color='purple');
plt.title('Sqaure Root Transformation', fontsize=20)
plt.show()


from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, 
                                   PolynomialFeatures)
from sklearn.metrics import mean_squared_error



from sklearn.model_selection import train_test_split

X= final_train1.drop('site_eui',axis=1)
y= final_train1['site_eui']
x_train, x_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=42)

def model(obj):
    obj.fit(x_train, y_train)
    obj_y_pred = lreg.predict(x_test)
    mean_squared_error = np.mean((obj_y_pred - y_test)**2)
    obj_coefficient = pd.DataFrame()
    obj_coefficient["Columns"] = x_train.columns
    obj_coefficient['Coefficient Estimate'] = pd.Series(lreg.coef_)
    return obj_coefficient, mean_squared_error

def visualize(obj_coeff, title, color):
    fig, ax = plt.subplots(figsize =(20, 8))

    ax.bar(lreg_coefficient["Columns"],
    obj_coeff['Coefficient Estimate'],
    color = color)
    ax.spines['bottom'].set_position('zero')
    plt.title(title, fontsize=25) 
    plt.style.use('ggplot')
    plt.show()

lreg = LinearRegression()
lreg_coefficient, lreg_mse = model(lreg)
visualize(lreg_coefficient, 'Linear Regression with Mean squared Error: {}'.format(lreg_mse), 'lightgreen')


lasso = Lasso(alpha = 100000000000000)
lasso_coefficient, lasso_mse = model(lasso)
visualize(lasso_coefficient, 'Lasso Regression with Mean squared Error: {}'.format(lasso_mse), 'lightblue')


ridge = Ridge(alpha = 1)
ridge_coefficient, ridge_mse = model(lasso)
visualize(ridge_coefficient, 'Ridge Regression with Mean squared Error: {}'.format(ridge_mse), 'pink')


alpha= [10 ** x for x in range(-6,8)]
cv_error_array=[]
for i in alpha:
    reg= Lasso(alpha=i,random_state=42)
    reg.fit(x_train,y_train)
    y_pred= reg.predict(x_test)
    loss= mean_squared_error(y_test,y_pred)
    cv_error_array.append(loss)
    print("For Alpha : ", i ,"Loss :",round(loss,2))
    
sub = pd.read_csv("sample_solution.csv")
sub["site_eui"] = lreg.predict(x_test)
sub.to_csv("submission3.csv", index = False)

print(sub)