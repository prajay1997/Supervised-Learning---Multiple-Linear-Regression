####   Multilinear regression
################## solution for Q1) #################

import pandas as pd
import numpy as np
import seaborn as sns
 # load the data
data = pd.read_csv(r"C:\Users\praja\Desktop\Data Science\Supervised Learning Technique\multiple Linear Regression\Datasets\50_Startups.csv")
data.columns
data = data.rename(columns={'R&D Spend':'rnd','Administration':'adm','Marketing Spend':'market', 'State':'state', 'Profit':'profit'})
data.info()
data.describe()

# Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

#profit
  
plt.bar(height = data.profit, x = np.arange(1, 51, 1)) #barplot
plt.hist(data.profit) #histogram
plt.boxplot(data.profit) #boxplot
sns.distplot(data['profit'], bins =5, kde = True)
# there is outlier in the profit datasets

# remove the outliers
from feature_engine.outliers import Winsorizer
 winsor = Winsorizer(capping_method = 'iqr',
                     tail = 'both',
                     fold = 1.5,
                     variables = ['profit'])

data['profit'] = winsor.fit_transform(data[['profit']])
plt.boxplot(data.profit)


# countplot
import seaborn as sns
plt.figure(1, figsize=(16,10))
sns.countplot(data['profit'])

# q-q plot
from scipy import stats
import pylab
stats.probplot(data.profit, dist ='norm', plot =pylab)
plt.show()

# rnd
plt.bar(height = data.rnd, x = np.arange(1, 51, 1)) #barplot
plt.hist(data.rnd) #histogram
plt.boxplot(data.rnd) #boxplot
sns.distplot(data['rnd'], bins =5, kde = True)

# jointplot
import seaborn as sns
sns.jointplot(x = data['rnd'], y = data['profit'])

# countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(data['rnd'])

# Q-Q Plot
stats.probplot(data.rnd, dist = "norm", plot = pylab)
plt.show()

# adm
plt.bar(height = data.adm, x = np.arange(1, 51, 1)) #barplot
plt.hist(data.adm) #histogram
plt.boxplot(data.adm) #boxplot
sns.distplot(data['adm'], bins =5, kde = True)
sns.jointplot(x = data['adm'], y = data['profit'])

plt.figure(1, figsize=(16, 10))
sns.countplot(data['adm'])

stats.probplot(data.adm, dist ='norm', plot = pylab)
plt.show()

# market
plt.bar(height = data.market, x = np.arange(1, 51, 1)) #barplot
plt.hist(data.market) #histogram
plt.boxplot(data.market) #boxplot
sns.distplot(data['market'], bins =5, kde = True)
sns.jointplot(x = data['market'], y = data['profit'])

plt.figure(1, figsize=(16, 10))
sns.countplot(data['market'])

stats.probplot(data.market, dist = 'norm', plot = pylab)
plt.show()

# convert the state column into numerical data 

data['state'].describe()

data = pd.get_dummies(data, columns = ['state'])
data.columns
data = data.rename(columns = {'state_California':'california','state_Florida':'florida', 'state_New York':'newyork'})

# standardize the data 
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
data_std = std.fit_transform(data)
data_std = pd.DataFrame(data_std, columns = data.columns)

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(data_std.iloc[:,:])

# Correlation matrix 
data_std.corr()
plt.figure(figsize=(10,9))
sns.heatmap(data_std.corr().round(2), annot = True)


# from the heatmap we can say that RND spend and profit have the strong correlation.
# marketing_spend and profit having the moderate correlation 
# administration and profit having the very less correlation
# market and rnd having the moderate collinearity
# states column does not correlated with profit  so we dont consider state column for analysis

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
ml1 = smf.ols('profit ~ rnd + adm + market + california + florida + newyork' , data = data_std).fit() # regression model

# summary
ml1.summary()

# p-values for adm and market are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm
sm.graphics.influence_plot(ml1)

# index 46, 48and 49 is showing high influence so we can exclude that entire row
data_new = data_std.drop(data_std.index[[46,48,49]])

# Preparing model  
ml_new = smf.ols('profit ~ rnd + adm + market + california + florida + newyork', data = data_new).fit()
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables

rsq_rnd = smf.ols('rnd ~  adm + market + california + florida + newyork', data = data_std).fit().rsquared  
vif_rnd = 1/(1 - rsq_rnd) 

rsq_adm = smf.ols('adm~  rnd + market + california + florida + newyork', data = data_std).fit().rsquared  
vif_adm = 1/(1 - rsq_adm)

rsq_market = smf.ols('market ~ rnd + adm + california + florida + newyork ', data = data_std).fit().rsquared  
vif_market = 1/(1 - rsq_market) 

rsq_cali = smf.ols('california ~ rnd + adm + market + florida + newyork  ', data = data_std).fit().rsquared  
vif_cali = 1/(1 - rsq_cali) 

rsq_flor = smf.ols('florida ~ rnd + adm + market + california + newyork  ', data = data_std).fit().rsquared  
vif_flor = 1/(1 - rsq_flor) 

rsq_ny = smf.ols('newyork ~ rnd + adm + market + florida + california  ', data = data_std).fit().rsquared  
vif_ny = 1/(1 - rsq_ny) 


# Storing vif values in a data frame
d1 = {'Variables':['rnd', 'adm', 'market', 'california','florida' ,'newyork'], 'VIF':[vif_rnd, vif_adm, vif_market,vif_cali,vif_flor,vif_ny ]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame

# as we can see that VIF for state column is infinite so we confirm that collinearity does not exist between them
# From the above correlation table and VIF values we can say that the States and Administration are not significant variaables for predicting the Profit values.
# We will build a model using R&D Spend(rnd) and Marketing Spend(market)

ml2 = smf.ols('profit ~ rnd + market', data = data_std).fit()
ml2.summary()

# prediction

pred = ml2.predict(data_std)

# 
# Q-Q plot to check the normality of errors

res = ml2.resid
sm.qqplot(res, line ='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()
plt

list(np.where(res<-0.8))
# there is a data point (observation no.49) that is very far away from the straight line


# Residual plot for homscedasticity
# Residuals vs Fitted plot 
sns.residplot(x = pred, y = data_std.profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

# There is no pattern in the Residual plot, 
# so we can say there is a linear relation and no error variance

# detecting influencers? ottliers
sm.graphics.influence_plot(ml2)

# We can see that the 49th observation is an Influencer point and has more distance than other data points. 
# We will delete this data point to further increase our accuracy.

data_new = data_std.drop(data_std.index[[49]], axis=0)
data_new.head()

final_model = smf.ols('profit ~ rnd + market', data = data_new).fit()
final_model.summary()

# Build the model
### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
data_train, data_test = train_test_split(data_new, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("profit ~ rnd + market ", data = data_train).fit()

# prediction on test data set 
test_pred = model_train.predict(data_test)

# test residual values 
test_resid = test_pred - data_test.profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(data_train)

# train residual values 
train_resid  = train_pred - data_train.profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

# the RMSE value for train and test datasets are almost equal the model is evaluated perfectly

# predicting for new data
new_data=pd.DataFrame({'rnd': 200000,'market': 100000}, index = [1])

final_model.predict(new_data)
# for RnD spend 200000 and Marketing spend of 100000 we get profit of 185163 `