#!/usr/bin/env python
# coding: utf-8

# # Load Packages

# In[1]:


import pandas as pd 
import numpy as np 
from matplotlib import rc
import matplotlib.pyplot as plt 
import datetime
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error
from sklearn.metrics import f1_score, precision_score, recall_score, f1_score
import seaborn as sns 
import scipy.stats as ss
from scipy import stats
import random
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy.stats import zscore
from sklearn.model_selection import cross_val_score
from sklearn import naive_bayes
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestRegressor
from imblearn import over_sampling
get_ipython().run_line_magic('matplotlib', 'inline')


# # Data Preparation

# ## Load Data

# In[2]:


# LOAD DATA
in_df = pd.read_csv('../data/1indicator_data.csv', low_memory=False)
bn_df = pd.read_csv('../data/2BN_Data.csv', low_memory=False)
em_df = pd.read_csv('../data/3employee_data.csv',low_memory=False)
inpair_df = pd.read_csv('../data/4indicator_pairs_data.csv',low_memory=False)
heat_df = pd.read_csv('../data/5heat_scores.csv',low_memory=False)


# In[3]:


bn_df.classification.value_counts()


# ## Employee Data

# ### Prepare Emp Data

# In[4]:


# Delete Duplicates
em_df.drop_duplicates(subset ="employee_id", keep = 'first', inplace = True)


# In[5]:


# Create index column for groupby function
em_df['tt'] = em_df.index


# ### Fill Country Name

# In[6]:


# change No_Data value to NaN
em_df['country_name'] = em_df['country_name'].replace('No_Data', np.nan)


# In[7]:


US_States = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
            "Florida", "Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas","Kentucky","Louisiana",
            "Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi","Missouri","Montana","Nebraska",
            "Nevada","New Hampshire","New Jersey","New Mexico","New York","North Carolina","North Dakota","Ohio",
            "Oklahoma","Oregon","Pennsylvania","Rhode Island","South Carolina","South Dakota","Tennessee","Texas",
            "Utah","Vermont","Virginia","Washington","West Virginia","Wisconsin","Wyoming"]


# In[8]:


# If state is one of US States:
em_df['country'] = em_df['state_name'].apply(lambda x: 'United States' if x in US_States else "")
em_df['country_name'] = em_df['country_name'].fillna(em_df['country'])
em_df = em_df.drop('country',axis=1)
em_df = em_df.replace(["",' '],np.NaN)

# If missing country_name has the same state_name or city with the above country_name:
h = em_df.dropna(subset=['country_name']).drop_duplicates('state_name').set_index('state_name')['country_name']
em_df['country_name'] = em_df['country_name'].fillna(em_df['city'].map(h))
g = em_df.dropna(subset=['country_name']).drop_duplicates('city').set_index('city')['country_name']
em_df['country_name'] = em_df['country_name'].fillna(em_df['city'].map(g))


# In[9]:


# pd.DataFrame(em_df.isnull().sum())


# In[10]:


# Fill NA Country with "other"
em_df['country_name'] = em_df['country_name'].fillna('other')


# In[11]:


#     In fact, we do not need to try to analyze the city or states for several reasons. 
#     Firstly, there are many cities and states in the world, amounting to hundreds of thousands, 
#     which will create a fragmentary analysis. 
#     Second, it is difficult to gather enough information for these two variables. 
#     Country is enough for analyzing.
#     Thereby, I would eliminate city and state when performing the data combining.


# ### Predict Function Group

# ##### Feature Selection using Chi-square test of Independence

# In[12]:


def chi_square_independence(data, factor_1, factor_2):
    # Contingency table:
    contigency_table = data.groupby([str(factor_1),str(factor_2)])['tt'].nunique().unstack(fill_value=0)
    # Get chi-square value , p-value, degrees of freedom, expected frequencies:
    stat, p, dof, expected = chi2_contingency(np.array(contigency_table))
    # select significance value
    alpha = 0.05
    # Determine whether to reject or keep your null hypothesis
    print('significance=%.3f, p=%.3f' % (alpha, p))
    if p <= alpha:
        print('Variables are associated (reject H0)')
    else:
        print('Variables are not associated(fail to reject H0)')


# In[13]:


for i in range(0, len(em_df.drop(["ge_hire_date","city","state_name","tt"],axis=1).columns)):
    print("function_group ~ ", str(em_df.drop(["ge_hire_date","city","state_name","tt"],axis=1).columns[i]))
    chi_square_independence(em_df, "function_group", str(em_df.drop(["ge_hire_date","city","state_name","tt"],axis=1).columns[i]))
    print("=====")


# In[ ]:





# ##### Feature Selection using Cramer's V

# In[14]:


def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


# In[15]:


cramers_v(em_df.person_status,em_df.function_group)


# In[16]:


for i in range(0, len(em_df.drop(["ge_hire_date","city","state_name","tt"],axis=1).columns)):
    print("function_group ~ ", str(em_df.drop(["ge_hire_date","city","state_name","tt"],axis=1).columns[i]))
    a = cramers_v(em_df.function_group,em_df[str(em_df.drop(["ge_hire_date","city","state_name","tt"],axis=1).columns[i])])
    print(a)
    print("=====")


# In[ ]:





# ##### Predict missing values

# In[17]:


#     As can be seen from the above results, 6 variables including person_type, person_status, country_name,
#     job_function, career_band, industry_focus_name are associated to function_group. So I use those 6 variables
#     in predicting the missing value of function_group.
#     Logistic Regression / Simple Regression / Simple Decision Tree / Random Forest


# In[18]:


# Data for function_group prediction:
df_function_group = pd.DataFrame(em_df[["function_group","person_type","person_status","country_name",
                                        "job_function", "career_band", "industry_focus_name"]])


# In[19]:


# train_data
train_fg = pd.get_dummies(df_function_group, prefix_sep="_", columns=["person_type","person_status","country_name",
                                        "job_function", "career_band", "industry_focus_name"])
train_fg = train_fg.dropna()
train_fg = train_fg.drop("function_group",axis=1)
train_label_fg = df_function_group["function_group"].dropna()


# In[20]:


# Split Data
random.seed(1112)
x_train,x_test,y_train,y_test=train_test_split(train_fg,train_label_fg,test_size=.3)


# In[21]:


# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100, 
                               bootstrap = True,
                               max_features = 'sqrt')
# Fit on training data
model.fit(x_train, y_train)
# Accuracy
rf_predictions = model.predict(x_test) # Actual class predictions
print(accuracy_score(y_test,rf_predictions))


# In[22]:


# Predict all of function_group and append the employee_data
rf_predictions_fg = model.predict(pd.get_dummies(df_function_group.drop("function_group",axis=1), prefix_sep="_"))
em_df['function_group_pred'] = rf_predictions_fg
em_df.function_group.fillna(em_df.function_group_pred, inplace=True)


# In[23]:


print(accuracy_score(em_df.function_group,em_df.function_group_pred))


# In[ ]:





# In[ ]:





# ### Predict Tenure

# In[24]:


def chi_independence(data, factor_1, factor_2):
    # Contingency table:
    contigency_table = data.groupby([str(factor_1),str(factor_2)])['tt'].nunique().unstack(fill_value=0)
    # Get chi-square value , p-value, degrees of freedom, expected frequencies:
    stat, p, dof, expected = chi2_contingency(np.array(contigency_table))
    return stat,p


# In[25]:


# Tenure = 2020 - hire year
em_df['tenure'] = 2020 - pd.DatetimeIndex(em_df.ge_hire_date).year
df_tenure = em_df.copy()
df_tenure = pd.get_dummies(df_tenure, prefix_sep="_", columns=list(df_tenure.drop(["tenure",
                                                                                   "employee_id",
                                                                                   "ge_hire_date",
                                                                                   "tt"],axis=1)))


# In[26]:


# Chi-square test of independence
index_chi = np.array([])
p_Value = np.array([])
stats = np.array([])
critical_Value = np.array([])
Decision = np.array([])
Decision_associated = np.array([])
alPha = np.array([])
alpha = 0.05
no_of_rows = len(df_tenure)
no_of_columns = 2
ddof=(no_of_rows-1)*(no_of_columns-1)

for i in range(0, len(df_tenure.columns)):
    stat, p = chi_independence(df_tenure, "tenure", str(df_tenure.columns[i]))
    critical_value = chi2.ppf(q=1-alpha,df=ddof)
    if (p < 0.05) or (stat > critical_value):
        decision = 'Associated'
        Decision_associated = np.append(Decision_associated, str(df_tenure.columns[i]))
    else: decision = 'Independent'
    
    Decision = np.append(Decision, decision)
    alPha = np.append(alPha, alpha)
    critical_Value = np.append(critical_Value, critical_value)
    stats = np.append(stats, stat)
    index_chi = np.append(index_chi, str(df_tenure.columns[i]))
    p_Value = np.append(p_Value, p)

df_tenure_associated = Decision_associated
# pd.DataFrame(zip(index_chi,p_Value,alPha,stats,critical_Value,Decision),
#             columns=['Classification ~ ','p-value','alpha','Chi-square Statistic','Critical Value','Decision'])


# In[27]:


df_tenure_nona = df_tenure.dropna()
model = RandomForestRegressor(n_estimators=100,random_state=10, max_depth=10)
model.fit(df_tenure_nona.drop(['tenure','employee_id','ge_hire_date','tt'],1),df_tenure_nona.tenure)
plt.figure(figsize=(5,6),dpi=100)
features = df_tenure_nona.drop(['tenure','employee_id','ge_hire_date','tt'],1).columns
importances = model.feature_importances_
indices = np.argsort(importances)[-50:]  # top 50 features
# plt.title('Important Features by Random Forest')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.show()


# In[28]:


# print('Important features not in the Chi-squared Associated array: \n')
other_important_features = np.array([])
for i in range(len(features[indices])):
    if features[indices][i] not in df_tenure_associated:
#         print(features[indices][i])
        other_important_features = np.append(other_important_features,features[indices][i])
df_tenure_features = np.append(df_tenure_associated,other_important_features)


# In[29]:


# Data for function_group prediction:
train_tenure = df_tenure[df_tenure_features].drop(['employee_id','ge_hire_date','tt'],1)
train_tenure_nona = train_tenure.dropna()
train_tenure_na = train_tenure[train_tenure.tenure.isnull()]
# Split data
x_train_tn,x_test_tn,y_train_tn,y_test_tn=train_test_split(train_tenure_nona.drop('tenure',1),
                                                           train_tenure_nona.tenure,test_size=.1)


# In[30]:


# Random Forest - Best
model = RandomForestRegressor(n_estimators=100,random_state=10, max_depth=10)
model.fit(x_train_tn,y_train_tn)
tenure_prediction = model.predict(x_test_tn)
tenure_diff = pd.DataFrame(list(zip(tenure_prediction,y_test_tn,tenure_prediction-y_test_tn)),
             columns=['Tenure Prediction','True Tenure','delta'])


# In[31]:


# Average difference per employee
np.absolute(tenure_prediction-y_test_tn).sum()/len(y_test_tn)


# In[32]:


em_df['tenure_pred'] = model.predict(train_tenure.drop('tenure',1))
em_df['tenure']=em_df['tenure'].fillna(em_df['tenure_pred'])
em_df = em_df.drop('tenure_pred',1)
em_df.tenure = em_df['tenure'].apply(lambda x: 1 if x<=0 else x)


# In[ ]:





# ## Join Data

# ### Join BN Data + Employee Data + Heat Value Data

# In[33]:


# duration = pd.DatetimeIndex(bn_df.alert_escalation_date) - pd.DatetimeIndex(bn_df.insert_date)
# bn_df['duration'] = duration.days


# In[34]:


# Join BN + Employee
merged_data = pd.merge(bn_df.drop("indicator_pairs",axis=1),em_df.drop(["ge_hire_date","city","state_name","function_group_pred"],axis=1))


# In[35]:


# Join Data + HeatValue
heat_df.drop_duplicates(subset ="SHARED_INDICATION_NAME", keep = 'first', inplace = True)
merged_data = pd.merge(merged_data,
                       heat_df.drop(["SHARED_INDICATOR_APPLICATION","SHARED_INDICATOR_TYPE","HEAT_VALUE","Unnamed: 6"],axis=1),
                       left_on='indicators',
                       right_on='SHARED_INDICATION_NAME',
                       how = 'left')
merged_data = merged_data.drop(['SHARED_INDICATION_NAME'], axis=1)


# In[36]:


# merged_data.to_csv("merged_data.csv")


# ### Filter Data

# In[37]:


# Fill NA
merged_data['owner_name'] = merged_data['owner_name'].fillna('other')
merged_data['indicator_heat_score'] = merged_data['indicator_heat_score'].fillna('other')
merged_data['alert_type'] = merged_data['alert_type'].fillna('other')
merged_data['indicator_source'] = merged_data['indicator_source'].fillna('other')
merged_data['SHARED_INDICATOR_ACTIVITY'] = merged_data['SHARED_INDICATOR_ACTIVITY'].fillna('other')


# In[38]:


# Filter Atomic
df_atomic = merged_data[merged_data['alert_type'].str.contains("Atomic")]
remove_column = list(merged_data.drop(["alert_escalation_date",
                                       "alert_id_fk",
                                       "insert_date",
                                       "employee_id",
                                       "alert_category",
                                       "tt",
                                       "tenure",
                                       'score',
                                       'risk_factor',
                                       'avg_score'],axis=1).columns)
df_atomic_onehot = pd.get_dummies(df_atomic, prefix_sep="_",columns=remove_column)
df_atomic_onehot['classification'] = df_atomic.classification


# In[39]:


# Filter Daily Heat
df_daily = merged_data[merged_data['alert_type'].str.contains("Daily_Heat")]
df_daily_onehot = pd.get_dummies(df_daily, prefix_sep="_",columns=remove_column)
df_daily_onehot['classification'] = df_daily.classification


# In[40]:


# Filter Weekly Heat
df_weekly = merged_data[merged_data['alert_type'].str.contains("Weekly_Heat")]
df_weekly_onehot = pd.get_dummies(df_weekly, prefix_sep="_",columns=remove_column)
df_weekly_onehot['classification'] = df_weekly.classification


# In[41]:


# Filter Monthly Heat
df_monthly = merged_data[merged_data['alert_type'].str.contains("Monthly_Heat")]
df_monthly_onehot = pd.get_dummies(df_monthly, prefix_sep="_",columns=remove_column)
df_monthly_onehot['classification'] = df_monthly.classification


# In[42]:


# Filter all threshold
df_threshold = merged_data[merged_data['alert_type'].isin(["Monthly_Heat","Daily_Heat","Weekly_Heat"])]
df_threshold_onehot = pd.get_dummies(df_threshold, prefix_sep="_",columns=remove_column)
df_threshold_onehot['classification'] = df_threshold.classification


# # EDA

# In[43]:


# Duration between insert_date and alert_escalate_date


# In[ ]:





# In[ ]:





# # Data Analysis

# ## Unsupervised Learning Algorithms

# ### K-means Clustering

# #### Atomic

# In[44]:


# data for clustering:
df_atomic_processed = df_atomic_onehot.drop(['alert_escalation_date', 'alert_id_fk', 'insert_date', 'employee_id','tt','alert_category','classification'],axis=1)


# In[45]:


# Find K Atomic: WCSS
# wcss = []
# for i in range(1, 14):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=9000, n_init=10)
#     kmeans.fit(df_atomic_processed)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 14), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()


# In[46]:


# K-mean
random.seed(1221)
kmeans = KMeans(n_clusters=10)
k_fitX = kmeans.fit(df_atomic_processed)


# In[47]:


clusters_atomic_df = k_fitX.cluster_centers_
clusters_atomic_df = pd.DataFrame(clusters_atomic_df,columns=list(df_atomic_processed.columns))
round(clusters_atomic_df,3)


# In[48]:


clusters_atomic_df.to_csv('clusters_atomic_df.csv')


# In[49]:


LIST = k_fitX.labels_
counts,values = pd.Series(LIST).value_counts().values, pd.Series(LIST).value_counts().index
df_results = pd.DataFrame(list(zip(values,counts)),columns=["value","count"])
df_results.sort_values(["value","count"],ascending=[1,0])


# In[ ]:





# #### Daily Heat

# In[50]:


# data for clustering:
df_daily_processed = df_daily_onehot.drop(['classification','alert_escalation_date', 'alert_id_fk', 'insert_date', 'employee_id','tt','alert_category'],axis=1)


# In[51]:


# Find K Atomic: WCSS
# wcss = []
# for i in range(1, 14):
#     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=9000, n_init=10)
#     kmeans.fit(df_daily_processed)
#     wcss.append(kmeans.inertia_)
# plt.plot(range(1, 14), wcss)
# plt.title('Elbow Method')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()


# In[52]:


# K-mean
random.seed(1221)
kmeans = KMeans(n_clusters=12,init='k-means++', max_iter=9000, algorithm = 'auto')
k_fitX = kmeans.fit(df_daily_processed)


# In[53]:


clusters_daily_df = k_fitX.cluster_centers_
clusters_daily_df = pd.DataFrame(clusters_daily_df,columns=list(df_daily_processed.columns))
# clusters_daily_df.to_csv('clusters_daily_df.csv')
# round(clusters_daily_df,3)
round(clusters_daily_df[['classification_FP',
                         'classification_TP/DE',
                         'classification_TP/HIGH',
                         'classification_TP/LOW',
                         'tenure']],3)


# In[54]:


clusters_daily_df.to_csv('clusters_daily_df.csv')


# In[55]:


LIST = k_fitX.labels_
counts,values = pd.Series(LIST).value_counts().values, pd.Series(LIST).value_counts().index
df_results = pd.DataFrame(list(zip(values,counts)),columns=["value","count"])
df_results.sort_values(["value","count"],ascending=[1,0])


# #### Weekly Heat

# In[56]:


# data for clustering:
df_weekly_processed = df_weekly_onehot.drop(['classification','alert_escalation_date', 'alert_id_fk', 'insert_date', 'employee_id','tt','alert_category'],axis=1)


# In[57]:


# Find K Atomic: WCSS
wcss = []
for i in range(1, 14):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=9000, n_init=10)
    kmeans.fit(df_weekly_processed)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 14), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[58]:


# K-mean
random.seed(1221)
kmeans = KMeans(n_clusters=8,init='k-means++', max_iter=9000, algorithm = 'auto')
k_fitX = kmeans.fit(df_weekly_processed)


# In[59]:


clusters_weekly_df = k_fitX.cluster_centers_
clusters_weekly_df = pd.DataFrame(clusters_weekly_df,columns=list(df_weekly_processed.columns))
# clusters_weekly_df.to_csv('clusters_weekly_df.csv')
round(clusters_weekly_df[['classification_FP',
                         'classification_TP/DE',
                         'classification_TP/HIGH',
                         'classification_TP/LOW']],3)


# In[60]:


LIST = k_fitX.labels_
counts,values = pd.Series(LIST).value_counts().values, pd.Series(LIST).value_counts().index
df_results = pd.DataFrame(list(zip(values,counts)),columns=["value","count"])
df_results.sort_values(["value","count"],ascending=[1,0])


# In[61]:


clusters_weekly_df.to_csv('clusters_weekly_df.csv')


# #### Monthly Heat

# In[62]:


# data for clustering:
df_monthly_processed = df_monthly_onehot.drop(['classification','alert_escalation_date', 'alert_id_fk', 'insert_date', 'employee_id','tt','alert_category'],axis=1)


# In[63]:


# Find K Atomic: WCSS
wcss = []
for i in range(1, 14):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=9000, n_init=10)
    kmeans.fit(df_monthly_processed)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 14), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[64]:


# K-mean
random.seed(1221)
kmeans = KMeans(n_clusters=8,init='k-means++', max_iter=9000, algorithm = 'auto')
k_fitX = kmeans.fit(df_monthly_processed)


# In[65]:


clusters_monthly_df = k_fitX.cluster_centers_
clusters_monthly_df = pd.DataFrame(clusters_monthly_df,columns=list(df_monthly_processed.columns))
# clusters_monthly_df.to_csv('clusters_monthly_df.csv')
round(clusters_monthly_df[['classification_FP',
                         'classification_TP/DE',
                         'classification_TP/HIGH',
                         'classification_TP/LOW']],3)


# In[66]:


LIST = k_fitX.labels_
counts,values = pd.Series(LIST).value_counts().values, pd.Series(LIST).value_counts().index
df_results = pd.DataFrame(list(zip(values,counts)),columns=["value","count"])
df_results.sort_values(["value","count"],ascending=[1,0])


# In[67]:


clusters_monthly_df.to_csv('clusters_monthly_df.csv')


# In[ ]:





# ## Prediction Models

# ### Additional Functions

# In[68]:


def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


# In[69]:


def mse_cal(actual_result, prediction_result):
    return mean_squared_error( pd.DataFrame(y_test), pd.DataFrame(prediction_result))


# In[ ]:





# In[ ]:





# ### Feature Selection

# #### Atomic

# In[70]:


pd.set_option('display.max_rows', 990)


# In[71]:


def chi_independence(data, factor_1, factor_2):
    # Contingency table:
    contigency_table = data.groupby([str(factor_1),str(factor_2)])['tt'].nunique().unstack(fill_value=0)
    # Get chi-square value , p-value, degrees of freedom, expected frequencies:
    stat, p, dof, expected = chi2_contingency(np.array(contigency_table))
    return stat,p


# In[72]:


# Chi-square test of independence
index_chi = np.array([])
p_Value = np.array([])
stats = np.array([])
critical_Value = np.array([])
Decision = np.array([])
Decision_associated = np.array([])
alPha = np.array([])
alpha = 0.05
no_of_rows = len(df_atomic_onehot)
no_of_columns = 2
ddof=(no_of_rows-1)*(no_of_columns-1)

for i in range(0, len(df_atomic_onehot.columns)):
    stat, p = chi_independence(df_atomic_onehot, "classification", str(df_atomic_onehot.columns[i]))
    critical_value = chi2.ppf(q=1-alpha,df=ddof)
    if (p < 0.05) or (stat > critical_value):
        decision = 'Associated'
        Decision_associated = np.append(Decision_associated, str(df_atomic_onehot.columns[i]))
    else: decision = 'Independent'
    
    Decision = np.append(Decision, decision)
    alPha = np.append(alPha, alpha)
    critical_Value = np.append(critical_Value, critical_value)
    stats = np.append(stats, stat)
    index_chi = np.append(index_chi, str(df_atomic_onehot.columns[i]))
    p_Value = np.append(p_Value, p)

df_atomic_associated = Decision_associated
# pd.DataFrame(zip(index_chi,p_Value,alPha,stats,critical_Value,Decision),
#             columns=['Classification ~ ','p-value','alpha','Chi-square Statistic','Critical Value','Decision'])


# In[73]:


# Random Forest
drop_columns = ['alert_escalation_date',
                'alert_id_fk',
                'insert_date',
                'employee_id',
                'alert_category',
                'tt',
              'classification_FP',
              'classification_TP/DE',
              'classification_TP/HIGH',
              'classification_TP/LOW']
df_atomic_onehot_rand = df_atomic_onehot.drop(drop_columns,axis=1)
df_atomic_onehot_rand.classification.replace({'TP/HIGH':1,'TP/LOW':0,'TP/DE':0,'FP':0},inplace=True)
model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(df_atomic_onehot_rand.drop('classification',axis=1),df_atomic_onehot_rand.classification)


# In[74]:


plt.figure(figsize=(5,6),dpi=100)
features = df_atomic_onehot_rand.drop('classification',axis=1).columns
importances = model.feature_importances_
indices = np.argsort(importances)[-100:]  # top 100 features
# plt.title('Important Features by Random Forest')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.show()


# In[75]:


# print('Important features not in the Chi-squared Associated array: \n')
other_important_features = np.array([])
for i in range(len(features[indices])):
    if features[indices][i] not in df_atomic_associated:
#         print(features[indices][i])
        other_important_features = np.append(other_important_features,features[indices][i])
df_atomic_features = np.append(df_atomic_associated,other_important_features)


# In[76]:


df_atomic_train = df_atomic_onehot[list(df_atomic_features)]
df_atomic_train = df_atomic_train.drop(['alert_escalation_date', 
                                      'alert_id_fk', 
                                      'insert_date',
                                      'classification_FP', 
                                      'classification_TP/DE',
                                      'classification_TP/HIGH', 
                                      'classification_TP/LOW',],axis=1)


# #### Daily

# In[77]:


# Chi-square test of independence
index_chi = np.array([])
p_Value = np.array([])
stats = np.array([])
critical_Value = np.array([])
Decision = np.array([])
Decision_associated = np.array([])
alPha = np.array([])
alpha = 0.05
no_of_rows = len(df_daily_onehot)
no_of_columns = 2
ddof=(no_of_rows-1)*(no_of_columns-1)

for i in range(0, len(df_daily_onehot.columns)):
    stat, p = chi_independence(df_daily_onehot, "classification", str(df_daily_onehot.columns[i]))
    critical_value = chi2.ppf(q=1-alpha,df=ddof)
    if (p < 0.05) or (stat > critical_value):
        decision = 'Associated'
        Decision_associated = np.append(Decision_associated, str(df_daily_onehot.columns[i]))
    else: decision = 'Independent'
    
    Decision = np.append(Decision, decision)
    alPha = np.append(alPha, alpha)
    critical_Value = np.append(critical_Value, critical_value)
    stats = np.append(stats, stat)
    index_chi = np.append(index_chi, str(df_daily_onehot.columns[i]))
    p_Value = np.append(p_Value, p)

df_daily_associated = Decision_associated
# pd.DataFrame(zip(index_chi,p_Value,alPha,stats,critical_Value,Decision),
#             columns=['Classification ~ ','p-value','alpha','Chi-square Statistic','Critical Value','Decision'])


# In[78]:


# Random Forest
drop_columns = ['alert_escalation_date',
                'alert_id_fk',
                'insert_date',
                'employee_id',
                'alert_category',
                'tt',
              'classification_FP',
              'classification_TP/DE',
              'classification_TP/HIGH',
              'classification_TP/LOW']
df_daily_onehot_rand = df_daily_onehot.drop(drop_columns,axis=1)
df_daily_onehot_rand.classification.replace({'TP/HIGH':1,'TP/LOW':0,'TP/DE':0,'FP':0},inplace=True)
model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(df_daily_onehot_rand.drop('classification',axis=1),df_daily_onehot_rand.classification)


# In[79]:


plt.figure(figsize=(5,6),dpi=100)
features = df_daily_onehot_rand.drop('classification',axis=1).columns
importances = model.feature_importances_
indices = np.argsort(importances)[-100:]  # top 100 features
# plt.title('Important Features by Random Forest')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.show()


# In[80]:


# print('Important features not in the Chi-squared Associated array: \n')
other_important_features = np.array([])
for i in range(len(features[indices])):
    if features[indices][i] not in df_daily_associated:
#         print(features[indices][i])
        other_important_features = np.append(other_important_features,features[indices][i])
df_daily_features = np.append(df_daily_associated,other_important_features)


# In[81]:


df_daily_train = df_daily_onehot[list(df_daily_features)]
df_daily_train = df_daily_train.drop(['alert_escalation_date', 
                                      'alert_id_fk', 
                                      'insert_date',
                                      'classification_FP', 
                                      'classification_TP/DE',
                                      'classification_TP/HIGH', 
                                      'classification_TP/LOW',],axis=1)


# In[ ]:





# In[ ]:





# #### Weekly

# In[82]:


# Chi-square test of independence
index_chi = np.array([])
p_Value = np.array([])
stats = np.array([])
critical_Value = np.array([])
Decision = np.array([])
Decision_associated = np.array([])
alPha = np.array([])
alpha = 0.05
no_of_rows = len(df_weekly_onehot)
no_of_columns = 2
ddof=(no_of_rows-1)*(no_of_columns-1)

for i in range(0, len(df_weekly_onehot.columns)):
    stat, p = chi_independence(df_weekly_onehot, "classification", str(df_weekly_onehot.columns[i]))
    critical_value = chi2.ppf(q=1-alpha,df=ddof)
    if (p < 0.05) or (stat > critical_value):
        decision = 'Associated'
        Decision_associated = np.append(Decision_associated, str(df_weekly_onehot.columns[i]))
    else: decision = 'Independent'
    
    Decision = np.append(Decision, decision)
    alPha = np.append(alPha, alpha)
    critical_Value = np.append(critical_Value, critical_value)
    stats = np.append(stats, stat)
    index_chi = np.append(index_chi, str(df_weekly_onehot.columns[i]))
    p_Value = np.append(p_Value, p)

df_weekly_associated = Decision_associated
# pd.DataFrame(zip(index_chi,p_Value,alPha,stats,critical_Value,Decision),
#             columns=['Classification ~ ','p-value','alpha','Chi-square Statistic','Critical Value','Decision'])


# In[83]:


# Random Forest
df_weekly_onehot_rand = df_weekly_onehot.drop(drop_columns,axis=1)
df_weekly_onehot_rand.classification.replace({'TP/HIGH':1,'TP/LOW':0,'TP/DE':0,'FP':0},inplace=True)
model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(df_weekly_onehot_rand.drop('classification',axis=1),df_weekly_onehot_rand.classification)


# In[84]:


plt.figure(figsize=(5,6),dpi=100)
features = df_weekly_onehot_rand.drop('classification',axis=1).columns
importances = model.feature_importances_
indices = np.argsort(importances)[-100:]  # top 100 features
# plt.title('Important Features by Random Forest')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.show()


# In[85]:


# print('Important features not in the Chi-squared Associated array: \n')
other_important_features = np.array([])
for i in range(len(features[indices])):
    if features[indices][i] not in df_weekly_associated:
#         print(features[indices][i])
        other_important_features = np.append(other_important_features,features[indices][i])
df_weekly_features = np.append(df_weekly_associated,other_important_features)


# In[86]:


df_weekly_train = df_weekly_onehot[list(df_weekly_features)]
df_weekly_train = df_weekly_train.drop(['alert_escalation_date', 
                                      'insert_date',
                                      'classification_FP', 
                                      'classification_TP/DE',
                                      'classification_TP/HIGH', 
                                      'classification_TP/LOW',],axis=1)


# In[ ]:





# #### Monthly

# In[87]:


# Chi-square test of independence
index_chi = np.array([])
p_Value = np.array([])
stats = np.array([])
critical_Value = np.array([])
Decision = np.array([])
Decision_associated = np.array([])
alPha = np.array([])
alpha = 0.05
no_of_rows = len(df_monthly_onehot)
no_of_columns = 2
ddof=(no_of_rows-1)*(no_of_columns-1)

for i in range(0, len(df_monthly_onehot.columns)):
    stat, p = chi_independence(df_monthly_onehot, "classification", str(df_monthly_onehot.columns[i]))
    critical_value = chi2.ppf(q=1-alpha,df=ddof)
    if (p < 0.05) or (stat > critical_value):
        decision = 'Associated'
        Decision_associated = np.append(Decision_associated, str(df_monthly_onehot.columns[i]))
    else: decision = 'Independent'
    
    Decision = np.append(Decision, decision)
    alPha = np.append(alPha, alpha)
    critical_Value = np.append(critical_Value, critical_value)
    stats = np.append(stats, stat)
    index_chi = np.append(index_chi, str(df_monthly_onehot.columns[i]))
    p_Value = np.append(p_Value, p)

df_monthly_associated = Decision_associated
# pd.DataFrame(zip(index_chi,p_Value,alPha,stats,critical_Value,Decision),
#             columns=['Classification ~ ','p-value','alpha','Chi-square Statistic','Critical Value','Decision'])


# In[88]:


# Random Forest
df_monthly_onehot_rand = df_monthly_onehot.drop(drop_columns,axis=1)
df_monthly_onehot_rand.classification.replace({'TP/HIGH':1,'TP/LOW':0,'TP/DE':0,'FP':0},inplace=True)
model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(df_monthly_onehot_rand.drop('classification',axis=1),df_monthly_onehot_rand.classification)


# In[89]:


plt.figure(figsize=(5,6),dpi=100)
features = df_monthly_onehot_rand.drop('classification',axis=1).columns
importances = model.feature_importances_
indices = np.argsort(importances)[-100:]  # top 100 features
# plt.title('Important Features by Random Forest')
# plt.barh(range(len(indices)), importances[indices], color='b', align='center')
# plt.yticks(range(len(indices)), [features[i] for i in indices])
# plt.xlabel('Relative Importance')
# plt.show()


# In[90]:


# print('Important features not in the Chi-squared Associated array: \n')
other_important_features = np.array([])
for i in range(len(features[indices])):
    if features[indices][i] not in df_monthly_associated:
#         print(features[indices][i])
        other_important_features = np.append(other_important_features,features[indices][i])
df_monthly_features = np.append(df_monthly_associated,other_important_features)


# In[91]:


df_monthly_train = df_monthly_onehot[list(df_monthly_features)]
df_monthly_train = df_monthly_train.drop(['alert_escalation_date', 
                                      'insert_date',
                                      'classification_FP', 
                                      'classification_TP/DE',
                                      'classification_TP/HIGH', 
                                      'classification_TP/LOW',],axis=1)


# In[ ]:





# In[ ]:





# ## High risks

# ### standardize data

# In[92]:


# Assign TP/HIGH = "1", others = "0"
df_atomic_trainH = df_atomic_train.classification.replace({'TP/HIGH':1,'TP/LOW':0,'TP/DE':0,'FP':0})
df_daily_trainH = df_daily_train.classification.replace({'TP/HIGH':1,'TP/LOW':0,'TP/DE':0,'FP':0})
df_weekly_trainH = df_weekly_train.classification.replace({'TP/HIGH':1,'TP/LOW':0,'TP/DE':0,'FP':0})
df_monthly_trainH = df_monthly_train.classification.replace({'TP/HIGH':1,'TP/LOW':0,'TP/DE':0,'FP':0})


# In[93]:


mm_scaler = preprocessing.MinMaxScaler()
np_scaled = mm_scaler.fit_transform(df_atomic_train.drop('classification',axis=1))
df_atomic_train_n = pd.DataFrame(np_scaled,columns=list(df_atomic_train.drop('classification',axis=1).columns))
np_scaled = mm_scaler.fit_transform(df_daily_train.drop('classification',axis=1))
df_daily_train_n = pd.DataFrame(np_scaled,columns=list(df_daily_train.drop('classification',axis=1).columns))
np_scaled = mm_scaler.fit_transform(df_weekly_train.drop('classification',axis=1))
df_weekly_train_n = pd.DataFrame(np_scaled,columns=list(df_weekly_train.drop('classification',axis=1).columns))
np_scaled = mm_scaler.fit_transform(df_monthly_train.drop('classification',axis=1))
df_monthly_train_n = pd.DataFrame(np_scaled,columns=list(df_monthly_train.drop('classification',axis=1).columns))


# In[94]:


df_daily_train_z = zscore(df_daily_train.drop('classification',axis=1))
df_daily_train_z = pd.DataFrame(df_daily_train_z,columns=list(df_daily_train.drop('classification',axis=1).columns))


# ### Split Data

# In[95]:


# Atomic
xAtomicTrain, xAtomicTest, yAtomicTrain, yAtomicTest = train_test_split(df_atomic_train_n,
                                                                    df_atomic_trainH,
                                                                    test_size=0.2, 
                                                                    random_state=1)


# In[96]:


# Daily
xDailyTrain, xDailyTest, yDailyTrain, yDailyTest = train_test_split(df_daily_train_n,
                                                                    df_daily_trainH,
                                                                    test_size=0.2, 
                                                                    random_state=1)


# In[97]:


# Weekly
xWeeklyTrain, xWeeklyTest, yWeeklyTrain, yWeeklyTest = train_test_split(df_weekly_train_n,
                                                                        df_weekly_trainH,
                                                                        test_size=0.2, 
                                                                        random_state=1)


# In[98]:


# Monthly
xMonthlyTrain, xMonthlyTest, yMonthlyTrain, yMonthlyTest = train_test_split(df_monthly_train_n,
                                                                            df_monthly_trainH,
                                                                            test_size=0.2, 
                                                                            random_state=1)


# ### Resampling

# In[99]:


over = over_sampling.SMOTE(random_state=2, sampling_strategy=0.5)
xAtomicTrain, yAtomicTrain = over.fit_resample(xAtomicTrain, yAtomicTrain)


# In[100]:


over = over_sampling.SMOTE(random_state=2, sampling_strategy=0.5)
xDailyTrain, yDailyTrain = over.fit_resample(xDailyTrain,yDailyTrain)


# In[101]:


over = over_sampling.SMOTE(random_state=2, sampling_strategy=0.5)
xWeeklyTrain, yWeeklyTrain = over.fit_resample(xWeeklyTrain, yWeeklyTrain)


# In[102]:


over = over_sampling.SMOTE(random_state=2, sampling_strategy=0.5)
xMonthlyTrain, yMonthlyTrain = over.fit_resample(xMonthlyTrain, yMonthlyTrain)


# ### Atomic

# #### Logistic Regression Model

# In[103]:


#LogisticRegression for Daily Heat
l_clf = LogisticRegression(max_iter=90000)
l_clf.fit(xAtomicTrain,yAtomicTrain)
l_prediction = l_clf.predict(xAtomicTest)
l_cm = confusion_matrix(yAtomicTest,l_prediction)
print('Accuracy',metrics.accuracy_score(yAtomicTest, l_prediction))
print('Mean Squared Error:',mean_squared_error(yAtomicTest,l_prediction))
# l_cv = np.mean(cross_val_score(l_clf,xDailyTrain,yDailyTrain,cv=10))
# print('Logistic Regression CV-Score: ',l_cv)


# In[104]:


labels = ['True Negative','False Positive','False Negative','True Positive']
categories = ["Low", "High"]
# make_confusion_matrix(l_cm, 
#                       group_names=labels,
#                       categories=categories,
#                       title="Confusion Matrix",
#                       cmap="Blues")


# In[ ]:





# #### Decision tree Model

# In[105]:


#DecisionTreeClassifier
dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(xAtomicTrain,yAtomicTrain)
dtc_prediction = dtc_clf.predict(xAtomicTest)
dtc_cm = confusion_matrix(yAtomicTest,dtc_prediction)
print('Accuracy: ',metrics.accuracy_score(yAtomicTest, dtc_prediction))
print('MSE:', mean_squared_error(yAtomicTest,dtc_prediction))


# In[106]:


# make_confusion_matrix(dtc_cm, 
#                       group_names=labels,
#                       categories=categories,
#                       title="Confusion Matrix",
#                       cmap="Blues")


# #### Random Forest

# In[107]:


#RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(xAtomicTrain,yAtomicTrain)
rfc_prediction = rfc_clf.predict(xAtomicTest)
rfc_cm = confusion_matrix(yAtomicTest, rfc_prediction)
print('Accuracy: ',metrics.accuracy_score(yAtomicTest, rfc_prediction))
print('MSE:',mean_squared_error(yAtomicTest,rfc_prediction))


# In[108]:


# make_confusion_matrix(rfc_cm, 
#                       group_names=labels,
#                       categories=categories,
#                       title="Confusion Matrix",
#                       cmap="Blues")


# In[ ]:





# #### Naive Bayes

# In[109]:


gnb = naive_bayes.MultinomialNB(alpha=1) # Laplace smoothing
nb_pred = gnb.fit(xAtomicTrain,yAtomicTrain).predict(xAtomicTest)
nvb_cm = confusion_matrix(yAtomicTest,nb_pred)
print('Accuracy: ',metrics.accuracy_score(yAtomicTest, nb_pred))
print('MSE for Naive Bayes:',mean_squared_error(yAtomicTest,nb_pred))


# In[110]:


# make_confusion_matrix(nvb_cm, 
#                       group_names=labels,
#                       categories=categories,
#                       title="Confusion Matrix",
#                       cmap="Blues")


# In[ ]:





# #### Test

# In[111]:


model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(xAtomicTrain,yAtomicTrain)
plt.figure(figsize=(5,6),dpi=100)
features = xAtomicTrain.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # top 100 features
plt.title('Important Features of Atomic Heat')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[112]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
accuracy = np.array([metrics.accuracy_score(yAtomicTest, l_prediction), 
                     metrics.accuracy_score(yAtomicTest, dtc_prediction),
                     metrics.accuracy_score(yAtomicTest, rfc_prediction),
                     metrics.accuracy_score(yAtomicTest, nb_pred)])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc]+' has the highest accuracy: '+str(round(accuracy[max_acc]*100,2))+'%')


# In[113]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
fn = np.array([l_cm[1,0], 
                     dtc_cm[1,0],
                     rfc_cm[1,0],
                     nvb_cm[1,0]])
min_fn = np.argmin(fn)
print(classifiers[min_fn]+' has the lowest False Negative: '+str(fn[min_fn]))


# In[114]:


# obtain optimal threshold from roc
l_probs = l_clf.predict_proba(xAtomicTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yAtomicTest)), l_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yAtomicTest)), l_probs)
optimal_proba_cutoff = sorted(list(zip(np.abs(lr_tpr - lr_fpr), _)), key=lambda i: i[0], reverse=True)[0][1]
roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in l_clf.predict_proba(xAtomicTest)[:, -1]]


# In[115]:


print(optimal_proba_cutoff)


# In[116]:


print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(yAtomicTest, l_prediction), accuracy_score(yAtomicTest, roc_predictions)))
print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(yAtomicTest, l_prediction), precision_score(yAtomicTest, roc_predictions)))
print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(yAtomicTest, l_prediction), recall_score(yAtomicTest, roc_predictions)))
print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(yAtomicTest, l_prediction), f1_score(yAtomicTest, roc_predictions)))
print("Mean Squared Error: Before and After Thresholding: {}, {}".format(mean_squared_error(yAtomicTest, l_prediction), mean_squared_error(yAtomicTest, roc_predictions)))


# In[117]:


make_confusion_matrix(confusion_matrix(yAtomicTest, roc_predictions), 
                      group_names=labels,
                      categories=categories,
                      title="Atomic",
                      cmap="Blues")


# In[118]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yAtomicTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Atomic')
plt.legend()
plt.show()


# In[ ]:





# ### Daily

# #### Logistic Regression Model

# In[119]:


#LogisticRegression for Daily Heat
l_clf = LogisticRegression(max_iter=90000)
l_clf.fit(xDailyTrain,yDailyTrain)
l_prediction = l_clf.predict(xDailyTest)
l_cm = confusion_matrix(yDailyTest,l_prediction)
print('Accuracy',metrics.accuracy_score(yDailyTest, l_prediction))
print('Mean Squared Error:',mean_squared_error(yDailyTest,l_prediction))
# l_cv = np.mean(cross_val_score(l_clf,xDailyTrain,yDailyTrain,cv=10))
# print('Logistic Regression CV-Score: ',l_cv)


# In[120]:


labels = ['True Negative','False Positive','False Negative','True Positive']
categories = ["Low", "High"]
make_confusion_matrix(l_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Decision tree Model

# In[121]:


#DecisionTreeClassifier
dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(xDailyTrain,yDailyTrain)
dtc_prediction = dtc_clf.predict(xDailyTest)
dtc_cm = confusion_matrix(yDailyTest,dtc_prediction)
print('Accuracy: ',metrics.accuracy_score(yDailyTest, dtc_prediction))
print('MSE:', mean_squared_error(yDailyTest,dtc_prediction))


# In[122]:


make_confusion_matrix(dtc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Random Forest

# In[123]:


#RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(xDailyTrain,yDailyTrain)
rfc_prediction = rfc_clf.predict(xDailyTest)
rfc_cm = confusion_matrix(yDailyTest, rfc_prediction)
print('Accuracy: ',metrics.accuracy_score(yDailyTest, rfc_prediction))
print('MSE:',mean_squared_error(yDailyTest,rfc_prediction))


# In[124]:


# fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
# tree.plot_tree(rfc_clf.estimators_[0],
#                feature_names = xThresholdTrain.columns, 
#                class_names=['High','None'],
#                filled = True);
# fig.savefig('rf_individualtree.png')
# plt.show()


# In[125]:


make_confusion_matrix(rfc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[ ]:





# #### Naive Bayes

# In[126]:


gnb = naive_bayes.MultinomialNB(alpha=1) # Laplace smoothing
nb_pred = gnb.fit(xDailyTrain,yDailyTrain).predict(xDailyTest)
nvb_cm = confusion_matrix(yDailyTest,nb_pred)
print('Accuracy: ',metrics.accuracy_score(yDailyTest, nb_pred))
print('MSE for Naive Bayes:',mean_squared_error(yDailyTest,nb_pred))


# In[127]:


make_confusion_matrix(nvb_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[ ]:





# #### Test

# In[128]:


model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(xDailyTrain,yDailyTrain)
plt.figure(figsize=(5,6),dpi=100)
features = xDailyTrain.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # top 100 features
plt.title('Important Features of Daily Heat')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[129]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
accuracy = np.array([metrics.accuracy_score(yDailyTest, l_prediction), 
                     metrics.accuracy_score(yDailyTest, dtc_prediction),
                     metrics.accuracy_score(yDailyTest, rfc_prediction),
                     metrics.accuracy_score(yDailyTest, nb_pred)])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc]+' has the highest accuracy: '+str(round(accuracy[max_acc]*100,2))+'%')


# In[130]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
fn = np.array([l_cm[1,0], 
                     dtc_cm[1,0],
                     rfc_cm[1,0],
                     nvb_cm[1,0]])
min_fn = np.argmin(fn)
print(classifiers[min_fn]+' has the lowest False Negative: '+str(fn[min_fn]))


# In[ ]:





# In[131]:


# obtain optimal threshold from roc
l_probs = rfc_clf.predict_proba(xDailyTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yDailyTest)), l_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yDailyTest)), l_probs)
optimal_proba_cutoff = sorted(list(zip(np.abs(lr_tpr - lr_fpr), _)), key=lambda i: i[0], reverse=True)[0][1]
roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in rfc_clf.predict_proba(xDailyTest)[:, -1]]


# In[132]:


optimal_proba_cutoff


# In[133]:


print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(yDailyTest, rfc_prediction), accuracy_score(yDailyTest, roc_predictions)))
print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(yDailyTest, rfc_prediction), precision_score(yDailyTest, roc_predictions)))
print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(yDailyTest, rfc_prediction), recall_score(yDailyTest, roc_predictions)))
print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(yDailyTest, rfc_prediction), f1_score(yDailyTest, roc_predictions)))
print("Mean Squared Error: Before and After Thresholding: {}, {}".format(mean_squared_error(yDailyTest, rfc_prediction), mean_squared_error(yDailyTest, roc_predictions)))


# In[134]:


make_confusion_matrix(confusion_matrix(yDailyTest, roc_predictions), 
                      group_names=labels,
                      categories=categories,
                      title="Daily Heat",
                      cmap="Blues")


# In[135]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yAtomicTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Daily Heat')
plt.legend()
plt.show()


# ### Weekly

# #### Logistic Regression Model

# In[136]:


#LogisticRegression for Daily Heat
l_clf = LogisticRegression(max_iter=90000)
l_clf.fit(xWeeklyTrain,yWeeklyTrain)
l_prediction = l_clf.predict(xWeeklyTest)
l_cm = confusion_matrix(yWeeklyTest,l_prediction)
print('Accuracy',metrics.accuracy_score(yWeeklyTest, l_prediction))
print('Mean Squared Error:',mean_squared_error(yWeeklyTest,l_prediction))
# l_cv = np.mean(cross_val_score(l_clf,xDailyTrain,yDailyTrain,cv=10))
# print('Logistic Regression CV-Score: ',l_cv)


# In[137]:


labels = ['True Negative','False Positive','False Negative','True Positive']
categories = ["Low", "High"]
make_confusion_matrix(l_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Decision tree Model

# In[138]:


#DecisionTreeClassifier
dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(xWeeklyTrain,yWeeklyTrain)
dtc_prediction = dtc_clf.predict(xWeeklyTest)
dtc_cm = confusion_matrix(yWeeklyTest,dtc_prediction)
print('Accuracy: ',metrics.accuracy_score(yWeeklyTest, dtc_prediction))
print('MSE:', mean_squared_error(yWeeklyTest,dtc_prediction))


# In[139]:


make_confusion_matrix(dtc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Random Forest

# In[140]:


#RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(xWeeklyTrain,yWeeklyTrain)
rfc_prediction = rfc_clf.predict(xWeeklyTest)
rfc_cm = confusion_matrix(yWeeklyTest, rfc_prediction)
print('Accuracy: ',metrics.accuracy_score(yWeeklyTest, rfc_prediction))
print('MSE:',mean_squared_error(yWeeklyTest,rfc_prediction))


# In[141]:


make_confusion_matrix(rfc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[ ]:





# #### Naive Bayes

# In[142]:


gnb = naive_bayes.MultinomialNB(alpha=1) # Laplace smoothing
nb_pred = gnb.fit(xWeeklyTrain,yWeeklyTrain).predict(xWeeklyTest)
nvb_cm = confusion_matrix(yWeeklyTest,nb_pred)
print('Accuracy: ',metrics.accuracy_score(yWeeklyTest, nb_pred))
print('MSE for Naive Bayes:',mean_squared_error(yWeeklyTest,nb_pred))


# In[143]:


make_confusion_matrix(nvb_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Test

# In[144]:


model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(xWeeklyTrain,yWeeklyTrain)
plt.figure(figsize=(5,6),dpi=100)
features = xWeeklyTrain.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # top 100 features
plt.title('Important Features of Weekly Heat')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[145]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
accuracy = np.array([metrics.accuracy_score(yWeeklyTest, l_prediction), 
                     metrics.accuracy_score(yWeeklyTest, dtc_prediction),
                     metrics.accuracy_score(yWeeklyTest, rfc_prediction),
                     metrics.accuracy_score(yWeeklyTest, nb_pred)])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc]+' has the highest accuracy: '+str(round(accuracy[max_acc]*100,2))+'%')


# In[146]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
fn = np.array([l_cm[1,0], 
                     dtc_cm[1,0],
                     rfc_cm[1,0],
                     nvb_cm[1,0]])
min_fn = np.argmin(fn)
print(classifiers[min_fn]+' has the lowest False Negative: '+str(fn[min_fn]))


# In[147]:


# obtain optimal threshold from roc
l_probs = rfc_clf.predict_proba(xWeeklyTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yWeeklyTest)), l_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yWeeklyTest)), l_probs)
optimal_proba_cutoff = sorted(list(zip(np.abs(lr_tpr - lr_fpr), _)), key=lambda i: i[0], reverse=True)[0][1]
roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in rfc_clf.predict_proba(xWeeklyTest)[:, -1]]


# In[148]:


optimal_proba_cutoff


# In[149]:


print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(yWeeklyTest, rfc_prediction), accuracy_score(yWeeklyTest, roc_predictions)))
print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(yWeeklyTest, rfc_prediction), precision_score(yWeeklyTest, roc_predictions)))
print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(yWeeklyTest, rfc_prediction), recall_score(yWeeklyTest, roc_predictions)))
print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(yWeeklyTest, rfc_prediction), f1_score(yWeeklyTest, roc_predictions)))
print("Mean Squared Error: Before and After Thresholding: {}, {}".format(mean_squared_error(yWeeklyTest, rfc_prediction), mean_squared_error(yWeeklyTest, roc_predictions)))


# In[150]:


make_confusion_matrix(confusion_matrix(yWeeklyTest, roc_predictions), 
                      group_names=labels,
                      categories=categories,
                      title="Weekly Heat",
                      cmap="Blues")


# In[151]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yAtomicTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Weekly Heat')
plt.legend()
plt.show()


# ### Monthly

# #### Logistic Regression Model

# In[152]:


#LogisticRegression for Daily Heat
l_clf = LogisticRegression(max_iter=90000)
l_clf.fit(xMonthlyTrain,yMonthlyTrain)
l_prediction = l_clf.predict(xMonthlyTest)
l_cm = confusion_matrix(yMonthlyTest,l_prediction)
print('Accuracy',metrics.accuracy_score(yMonthlyTest, l_prediction))
print('Mean Squared Error:',mean_squared_error(yMonthlyTest,l_prediction))
# l_cv = np.mean(cross_val_score(l_clf,xDailyTrain,yDailyTrain,cv=10))
# print('Logistic Regression CV-Score: ',l_cv)


# In[153]:


labels = ['True Negative','False Positive','False Negative','True Positive']
categories = ["Low", "High"]
make_confusion_matrix(l_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[154]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yMonthlyTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yMonthlyTest)), ns_probs)
l_probs = l_clf.predict_proba(xMonthlyTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yMonthlyTest)), l_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yMonthlyTest)), ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yMonthlyTest)), l_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:





# #### Decision tree Model

# In[155]:


#DecisionTreeClassifier
dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(xMonthlyTrain,yMonthlyTrain)
dtc_prediction = dtc_clf.predict(xMonthlyTest)
dtc_cm = confusion_matrix(yMonthlyTest,dtc_prediction)
print('Accuracy: ',metrics.accuracy_score(yMonthlyTest, dtc_prediction))
print('MSE:', mean_squared_error(yMonthlyTest,dtc_prediction))


# In[156]:


make_confusion_matrix(dtc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Random Forest

# In[157]:


#RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(xMonthlyTrain,yMonthlyTrain)
rfc_prediction = rfc_clf.predict(xMonthlyTest)
rfc_cm = confusion_matrix(yMonthlyTest, rfc_prediction)
print('Accuracy: ',metrics.accuracy_score(yMonthlyTest, rfc_prediction))
print('MSE:',mean_squared_error(yMonthlyTest,rfc_prediction))


# In[158]:


make_confusion_matrix(rfc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Naive Bayes

# In[159]:


gnb = naive_bayes.MultinomialNB(alpha=1) # Laplace smoothing
nb_pred = gnb.fit(xMonthlyTrain,yMonthlyTrain).predict(xMonthlyTest)
nvb_cm = confusion_matrix(yMonthlyTest,nb_pred)
print('Accuracy: ',metrics.accuracy_score(yMonthlyTest, nb_pred))
print('MSE for Naive Bayes:',mean_squared_error(yMonthlyTest,nb_pred))


# In[160]:


make_confusion_matrix(nvb_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Test

# In[161]:


model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(xMonthlyTrain,yMonthlyTrain)
plt.figure(figsize=(5,6),dpi=100)
features = xMonthlyTrain.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # top 100 features
plt.title('Important Features of Monthly Heat')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[162]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
accuracy = np.array([metrics.accuracy_score(yMonthlyTest, l_prediction), 
                     metrics.accuracy_score(yMonthlyTest, dtc_prediction),
                     metrics.accuracy_score(yMonthlyTest, rfc_prediction),
                     metrics.accuracy_score(yMonthlyTest, nb_pred)])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc]+' has the highest accuracy: '+str(round(accuracy[max_acc]*100,2))+'%')


# In[163]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
fn = np.array([l_cm[1,0], 
                     dtc_cm[1,0],
                     rfc_cm[1,0],
                     nvb_cm[1,0]])
min_fn = np.argmin(fn)
print(classifiers[min_fn]+' has the lowest False Negative: '+str(fn[min_fn]))


# In[164]:


# obtain optimal threshold from roc
l_probs = rfc_clf.predict_proba(xMonthlyTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yMonthlyTest)), l_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yMonthlyTest)), l_probs)
optimal_proba_cutoff = sorted(list(zip(np.abs(lr_tpr - lr_fpr), _)), key=lambda i: i[0], reverse=True)[0][1]
roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in rfc_clf.predict_proba(xMonthlyTest)[:, -1]]


# In[165]:


optimal_proba_cutoff


# In[166]:


print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(yMonthlyTest, rfc_prediction), accuracy_score(yMonthlyTest, roc_predictions)))
print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(yMonthlyTest, rfc_prediction), precision_score(yMonthlyTest, roc_predictions)))
print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(yMonthlyTest, rfc_prediction), recall_score(yMonthlyTest, roc_predictions)))
print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(yMonthlyTest, rfc_prediction), f1_score(yMonthlyTest, roc_predictions)))
print("Mean Squared Error: Before and After Thresholding: {}, {}".format(mean_squared_error(yMonthlyTest, rfc_prediction), mean_squared_error(yMonthlyTest, roc_predictions)))


# In[167]:


make_confusion_matrix(confusion_matrix(yMonthlyTest, roc_predictions), 
                      group_names=labels,
                      categories=categories,
                      title="Monthly Heat",
                      cmap="Blues")


# In[168]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yAtomicTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Monthly Heat')
plt.legend()
plt.show()


# ## Notable risks

# ### standardize data

# In[169]:


# Assign TP/HIGH = "1", others = "0"
df_atomic_trainNR = df_atomic_train.classification.replace({'TP/HIGH':1,'TP/LOW':1,'TP/DE':0,'FP':0})
df_daily_trainNR = df_daily_train.classification.replace({'TP/HIGH':1,'TP/LOW':1,'TP/DE':0,'FP':0})
df_weekly_trainNR = df_weekly_train.classification.replace({'TP/HIGH':1,'TP/LOW':1,'TP/DE':0,'FP':0})
df_monthly_trainNR = df_monthly_train.classification.replace({'TP/HIGH':1,'TP/LOW':1,'TP/DE':0,'FP':0})


# ### Split Data

# In[170]:


# Atomic
xAtomicTrain, xAtomicTest, yAtomicTrain, yAtomicTest = train_test_split(df_atomic_train_n,
                                                                    df_atomic_trainNR,
                                                                    test_size=0.2, 
                                                                    random_state=1)


# In[171]:


# Daily
xDailyTrain, xDailyTest, yDailyTrain, yDailyTest = train_test_split(df_daily_train_n,
                                                                    df_daily_trainNR,
                                                                    test_size=0.2, 
                                                                    random_state=1)


# In[172]:


# Weekly
xWeeklyTrain, xWeeklyTest, yWeeklyTrain, yWeeklyTest = train_test_split(df_weekly_train_n,
                                                                        df_weekly_trainNR,
                                                                        test_size=0.2, 
                                                                        random_state=1)


# In[173]:


# Monthly
xMonthlyTrain, xMonthlyTest, yMonthlyTrain, yMonthlyTest = train_test_split(df_monthly_train_n,
                                                                            df_monthly_trainNR,
                                                                            test_size=0.2, 
                                                                            random_state=1)


# ### Resampling

# In[174]:


over = over_sampling.SMOTE(random_state=2, sampling_strategy=0.5)
xAtomicTrain, yAtomicTrain = over.fit_resample(xAtomicTrain, yAtomicTrain)


# In[175]:


over = over_sampling.SMOTE(random_state=2, sampling_strategy=0.5)
xDailyTrain, yDailyTrain = over.fit_resample(xDailyTrain,yDailyTrain)


# In[176]:


over = over_sampling.SMOTE(random_state=2, sampling_strategy=0.5)
xWeeklyTrain, yWeeklyTrain = over.fit_resample(xWeeklyTrain, yWeeklyTrain)


# In[177]:


over = over_sampling.SMOTE(random_state=2, sampling_strategy=0.5)
xMonthlyTrain, yMonthlyTrain = over.fit_resample(xMonthlyTrain, yMonthlyTrain)


# ### Atomic

# #### Logistic Regression Model

# In[178]:


#LogisticRegression for Daily Heat
l_clf = LogisticRegression(max_iter=90000)
l_clf.fit(xAtomicTrain,yAtomicTrain)
l_prediction = l_clf.predict(xAtomicTest)
l_cm = confusion_matrix(yAtomicTest,l_prediction)
print('Accuracy',metrics.accuracy_score(yAtomicTest, l_prediction))
print('Mean Squared Error:',mean_squared_error(yAtomicTest,l_prediction))
# l_cv = np.mean(cross_val_score(l_clf,xDailyTrain,yDailyTrain,cv=10))
# print('Logistic Regression CV-Score: ',l_cv)


# In[179]:


labels = ['True Negative','False Positive','False Negative','True Positive']
categories = ["Low", "High"]
make_confusion_matrix(l_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[ ]:





# #### Decision tree Model

# In[180]:


#DecisionTreeClassifier
dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(xAtomicTrain,yAtomicTrain)
dtc_prediction = dtc_clf.predict(xAtomicTest)
dtc_cm = confusion_matrix(yAtomicTest,dtc_prediction)
print('Accuracy: ',metrics.accuracy_score(yAtomicTest, dtc_prediction))
print('MSE:', mean_squared_error(yAtomicTest,dtc_prediction))


# In[181]:


make_confusion_matrix(dtc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Random Forest

# In[182]:


#RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(xAtomicTrain,yAtomicTrain)
rfc_prediction = rfc_clf.predict(xAtomicTest)
rfc_cm = confusion_matrix(yAtomicTest, rfc_prediction)
print('Accuracy: ',metrics.accuracy_score(yAtomicTest, rfc_prediction))
print('MSE:',mean_squared_error(yAtomicTest,rfc_prediction))


# In[183]:


make_confusion_matrix(rfc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[ ]:





# #### Naive Bayes

# In[184]:


gnb = naive_bayes.MultinomialNB(alpha=1) # Laplace smoothing
nb_pred = gnb.fit(xAtomicTrain,yAtomicTrain).predict(xAtomicTest)
nvb_cm = confusion_matrix(yAtomicTest,nb_pred)
print('Accuracy: ',metrics.accuracy_score(yAtomicTest, nb_pred))
print('MSE for Naive Bayes:',mean_squared_error(yAtomicTest,nb_pred))


# In[185]:


make_confusion_matrix(nvb_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[ ]:





# #### Testt

# In[186]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
accuracy = np.array([metrics.accuracy_score(yAtomicTest, l_prediction), 
                     metrics.accuracy_score(yAtomicTest, dtc_prediction),
                     metrics.accuracy_score(yAtomicTest, rfc_prediction),
                     metrics.accuracy_score(yAtomicTest, nb_pred)])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc]+' has the highest accuracy: '+str(round(accuracy[max_acc]*100,2))+'%')


# In[187]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
fn = np.array([l_cm[1,0], 
                     dtc_cm[1,0],
                     rfc_cm[1,0],
                     nvb_cm[1,0]])
min_fn = np.argmin(fn)
print(classifiers[min_fn]+' has the lowest False Negative: '+str(fn[min_fn]))


# In[188]:


roc_predictions = [1 if i >= 0.01 else 0 for i in rfc_clf.predict_proba(xAtomicTest)[:, -1]]


# In[189]:


print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(yAtomicTest, rfc_prediction), accuracy_score(yAtomicTest, roc_predictions)))
print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(yAtomicTest, rfc_prediction), precision_score(yAtomicTest, roc_predictions)))
print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(yAtomicTest, rfc_prediction), recall_score(yAtomicTest, roc_predictions)))
print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(yAtomicTest, rfc_prediction), f1_score(yAtomicTest, roc_predictions)))
print("Mean Squared Error: Before and After Thresholding: {}, {}".format(mean_squared_error(yAtomicTest, rfc_prediction), mean_squared_error(yAtomicTest, roc_predictions)))


# In[190]:


make_confusion_matrix(confusion_matrix(yAtomicTest, roc_predictions), 
                      group_names=labels,
                      categories=categories,
                      title="Atomic",
                      cmap="Blues")


# In[191]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yAtomicTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Atomic')
plt.legend()
plt.show()


# In[192]:


model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(xAtomicTrain,yAtomicTrain)
plt.figure(figsize=(5,6),dpi=100)
features = xAtomicTrain.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # top 100 features
plt.title('Important Features of Atomic Heat')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### Daily

# #### Logistic Regression Model

# In[193]:


#LogisticRegression for Daily Heat
l_clf = LogisticRegression(max_iter=90000)
l_clf.fit(xDailyTrain,yDailyTrain)
l_prediction = l_clf.predict(xDailyTest)
l_cm = confusion_matrix(yDailyTest,l_prediction)
print('Accuracy',metrics.accuracy_score(yDailyTest, l_prediction))
print('Mean Squared Error:',mean_squared_error(yDailyTest,l_prediction))
# l_cv = np.mean(cross_val_score(l_clf,xDailyTrain,yDailyTrain,cv=10))
# print('Logistic Regression CV-Score: ',l_cv)


# In[194]:


labels = ['True Negative','False Positive','False Negative','True Positive']
categories = ["Low", "High"]
make_confusion_matrix(l_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[195]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yDailyTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yDailyTest)), ns_probs)
l_probs = l_clf.predict_proba(xDailyTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yDailyTest)), l_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yDailyTest)), ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yDailyTest)), l_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:





# #### Decision tree Model

# In[196]:


#DecisionTreeClassifier
dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(xDailyTrain,yDailyTrain)
dtc_prediction = dtc_clf.predict(xDailyTest)
dtc_cm = confusion_matrix(yDailyTest,dtc_prediction)
print('Accuracy: ',metrics.accuracy_score(yDailyTest, dtc_prediction))
print('MSE:', mean_squared_error(yDailyTest,dtc_prediction))


# In[197]:


make_confusion_matrix(dtc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Random Forest

# In[198]:


#RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(xDailyTrain,yDailyTrain)
rfc_prediction = rfc_clf.predict(xDailyTest)
rfc_cm = confusion_matrix(yDailyTest, rfc_prediction)
print('Accuracy: ',metrics.accuracy_score(yDailyTest, rfc_prediction))
print('MSE:',mean_squared_error(yDailyTest,rfc_prediction))


# In[199]:


make_confusion_matrix(rfc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[ ]:





# #### Naive Bayes

# In[200]:


gnb = naive_bayes.MultinomialNB(alpha=1) # Laplace smoothing
nb_pred = gnb.fit(xDailyTrain,yDailyTrain).predict(xDailyTest)
nvb_cm = confusion_matrix(yDailyTest,nb_pred)
print('Accuracy: ',metrics.accuracy_score(yDailyTest, nb_pred))
print('MSE for Naive Bayes:',mean_squared_error(yDailyTest,nb_pred))


# In[201]:


make_confusion_matrix(nvb_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[ ]:





# #### Test

# In[ ]:





# In[ ]:





# In[202]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
accuracy = np.array([metrics.accuracy_score(yDailyTest, l_prediction), 
                     metrics.accuracy_score(yDailyTest, dtc_prediction),
                     metrics.accuracy_score(yDailyTest, rfc_prediction),
                     metrics.accuracy_score(yDailyTest, nb_pred)])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc]+' has the highest accuracy: '+str(round(accuracy[max_acc]*100,2))+'%')


# In[203]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
fn = np.array([l_cm[1,0], 
                     dtc_cm[1,0],
                     rfc_cm[1,0],
                     nvb_cm[1,0]])
min_fn = np.argmin(fn)
print(classifiers[min_fn]+' has the lowest False Negative: '+str(fn[min_fn]))


# In[ ]:





# In[204]:


# obtain optimal threshold from roc
l_probs = rfc_clf.predict_proba(xDailyTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yDailyTest)), l_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yDailyTest)), l_probs)
optimal_proba_cutoff = sorted(list(zip(np.abs(lr_tpr - lr_fpr), _)), key=lambda i: i[0], reverse=True)[0][1]
roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in rfc_clf.predict_proba(xDailyTest)[:, -1]]


# In[205]:


optimal_proba_cutoff


# In[206]:


print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(yDailyTest, rfc_prediction), accuracy_score(yDailyTest, roc_predictions)))
print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(yDailyTest, rfc_prediction), precision_score(yDailyTest, roc_predictions)))
print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(yDailyTest, rfc_prediction), recall_score(yDailyTest, roc_predictions)))
print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(yDailyTest, rfc_prediction), f1_score(yDailyTest, roc_predictions)))
print("Mean Squared Error: Before and After Thresholding: {}, {}".format(mean_squared_error(yDailyTest, rfc_prediction), mean_squared_error(yDailyTest, roc_predictions)))


# In[207]:


make_confusion_matrix(confusion_matrix(yDailyTest, roc_predictions), 
                      group_names=labels,
                      categories=categories,
                      title="Daily Heat",
                      cmap="Blues")


# In[208]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yAtomicTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Daily Heat')
plt.legend()
plt.show()


# In[209]:


model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(xDailyTrain,yDailyTrain)
plt.figure(figsize=(5,6),dpi=100)
features = xDailyTrain.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # top 100 features
plt.title('Important Features of Daily Heat')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# ### Weekly

# #### Logistic Regression Model

# In[210]:


#LogisticRegression for Daily Heat
l_clf = LogisticRegression(max_iter=90000)
l_clf.fit(xWeeklyTrain,yWeeklyTrain)
l_prediction = l_clf.predict(xWeeklyTest)
l_cm = confusion_matrix(yWeeklyTest,l_prediction)
print('Accuracy',metrics.accuracy_score(yWeeklyTest, l_prediction))
print('Mean Squared Error:',mean_squared_error(yWeeklyTest,l_prediction))
# l_cv = np.mean(cross_val_score(l_clf,xDailyTrain,yDailyTrain,cv=10))
# print('Logistic Regression CV-Score: ',l_cv)


# In[211]:


labels = ['True Negative','False Positive','False Negative','True Positive']
categories = ["Low", "High"]
make_confusion_matrix(l_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[212]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yWeeklyTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yWeeklyTest)), ns_probs)
l_probs = l_clf.predict_proba(xWeeklyTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yWeeklyTest)), l_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yWeeklyTest)), ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yWeeklyTest)), l_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:





# #### Decision tree Model

# In[213]:


#DecisionTreeClassifier
dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(xWeeklyTrain,yWeeklyTrain)
dtc_prediction = dtc_clf.predict(xWeeklyTest)
dtc_cm = confusion_matrix(yWeeklyTest,dtc_prediction)
print('Accuracy: ',metrics.accuracy_score(yWeeklyTest, dtc_prediction))
print('MSE:', mean_squared_error(yWeeklyTest,dtc_prediction))


# In[214]:


make_confusion_matrix(dtc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Random Forest

# In[215]:


#RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(xWeeklyTrain,yWeeklyTrain)
rfc_prediction = rfc_clf.predict(xWeeklyTest)
rfc_cm = confusion_matrix(yWeeklyTest, rfc_prediction)
print('Accuracy: ',metrics.accuracy_score(yWeeklyTest, rfc_prediction))
print('MSE:',mean_squared_error(yWeeklyTest,rfc_prediction))


# In[216]:


make_confusion_matrix(rfc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[ ]:





# #### Naive Bayes

# In[217]:


gnb = naive_bayes.MultinomialNB(alpha=1) # Laplace smoothing
nb_pred = gnb.fit(xWeeklyTrain,yWeeklyTrain).predict(xWeeklyTest)
nvb_cm = confusion_matrix(yWeeklyTest,nb_pred)
print('Accuracy: ',metrics.accuracy_score(yWeeklyTest, nb_pred))
print('MSE for Naive Bayes:',mean_squared_error(yWeeklyTest,nb_pred))


# In[218]:


make_confusion_matrix(nvb_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Test

# In[219]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
accuracy = np.array([metrics.accuracy_score(yWeeklyTest, l_prediction), 
                     metrics.accuracy_score(yWeeklyTest, dtc_prediction),
                     metrics.accuracy_score(yWeeklyTest, rfc_prediction),
                     metrics.accuracy_score(yWeeklyTest, nb_pred)])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc]+' has the highest accuracy: '+str(round(accuracy[max_acc]*100,2))+'%')


# In[220]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
fn = np.array([l_cm[1,0], 
                     dtc_cm[1,0],
                     rfc_cm[1,0],
                     nvb_cm[1,0]])
min_fn = np.argmin(fn)
print(classifiers[min_fn]+' has the lowest False Negative: '+str(fn[min_fn]))


# In[221]:


# obtain optimal threshold from roc
l_probs = rfc_clf.predict_proba(xWeeklyTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yWeeklyTest)), l_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yWeeklyTest)), l_probs)
optimal_proba_cutoff = sorted(list(zip(np.abs(lr_tpr - lr_fpr), _)), key=lambda i: i[0], reverse=True)[0][1]
roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in rfc_clf.predict_proba(xWeeklyTest)[:, -1]]


# In[222]:


optimal_proba_cutoff


# In[223]:


print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(yWeeklyTest, rfc_prediction), accuracy_score(yWeeklyTest, roc_predictions)))
print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(yWeeklyTest, rfc_prediction), precision_score(yWeeklyTest, roc_predictions)))
print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(yWeeklyTest, rfc_prediction), recall_score(yWeeklyTest, roc_predictions)))
print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(yWeeklyTest, rfc_prediction), f1_score(yWeeklyTest, roc_predictions)))
print("Mean Squared Error: Before and After Thresholding: {}, {}".format(mean_squared_error(yWeeklyTest, rfc_prediction), mean_squared_error(yWeeklyTest, roc_predictions)))


# In[224]:


make_confusion_matrix(confusion_matrix(yWeeklyTest, roc_predictions), 
                      group_names=labels,
                      categories=categories,
                      title="Weekly Heat",
                      cmap="Blues")


# In[225]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yAtomicTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Weekly Heat')
plt.legend()
plt.show()


# In[226]:


model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(xWeeklyTrain,yWeeklyTrain)
plt.figure(figsize=(5,6),dpi=100)
features = xWeeklyTrain.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # top 100 features
plt.title('Important Features of Weekly Heat')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:





# In[ ]:





# ### Monthly

# #### Logistic Regression Model

# In[227]:


#LogisticRegression for Daily Heat
l_clf = LogisticRegression(max_iter=90000)
l_clf.fit(xMonthlyTrain,yMonthlyTrain)
l_prediction = l_clf.predict(xMonthlyTest)
l_cm = confusion_matrix(yMonthlyTest,l_prediction)
print('Accuracy',metrics.accuracy_score(yMonthlyTest, l_prediction))
print('Mean Squared Error:',mean_squared_error(yMonthlyTest,l_prediction))
# l_cv = np.mean(cross_val_score(l_clf,xDailyTrain,yDailyTrain,cv=10))
# print('Logistic Regression CV-Score: ',l_cv)


# In[228]:


labels = ['True Negative','False Positive','False Negative','True Positive']
categories = ["Low", "High"]
make_confusion_matrix(l_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[229]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yMonthlyTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yMonthlyTest)), ns_probs)
l_probs = l_clf.predict_proba(xMonthlyTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yMonthlyTest)), l_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yMonthlyTest)), ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yMonthlyTest)), l_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# In[ ]:





# #### Decision tree Model

# In[230]:


#DecisionTreeClassifier
dtc_clf = tree.DecisionTreeClassifier()
dtc_clf = dtc_clf.fit(xMonthlyTrain,yMonthlyTrain)
dtc_prediction = dtc_clf.predict(xMonthlyTest)
dtc_cm = confusion_matrix(yMonthlyTest,dtc_prediction)
print('Accuracy: ',metrics.accuracy_score(yMonthlyTest, dtc_prediction))
print('MSE:', mean_squared_error(yMonthlyTest,dtc_prediction))


# In[231]:


make_confusion_matrix(dtc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[232]:


from sklearn import externals
import io
# from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


# In[233]:


dot_data = io.StringIO()
# export_graphviz(dtc_clf, out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(graph.create_png())


# In[234]:


from sklearn.tree import export_text


# In[235]:


print(export_text(dtc_clf, feature_names=(list(xMonthlyTrain.columns))))


# In[ ]:





# #### Random Forest

# In[236]:


#RandomForestClassifier
rfc_clf = RandomForestClassifier()
rfc_clf.fit(xMonthlyTrain,yMonthlyTrain)
rfc_prediction = rfc_clf.predict(xMonthlyTest)
rfc_cm = confusion_matrix(yMonthlyTest, rfc_prediction)
print('Accuracy: ',metrics.accuracy_score(yMonthlyTest, rfc_prediction))
print('MSE:',mean_squared_error(yMonthlyTest,rfc_prediction))


# In[237]:


make_confusion_matrix(rfc_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# In[238]:


# export_graphviz(rfc_clf.estimators_[4], out_file=dot_data,  
#                 filled=True, rounded=True,
#                 special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
# Image(graph.create_png())


# In[239]:


model = RandomForestRegressor(random_state=12, max_depth=10)
model.fit(xMonthlyTrain,yMonthlyTrain)
plt.figure(figsize=(5,6),dpi=100)
features = xMonthlyTrain.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # top 100 features
plt.title('Important Features of Monthly Heat')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# #### Naive Bayes

# In[240]:


gnb = naive_bayes.MultinomialNB(alpha=1) # Laplace smoothing
nb_pred = gnb.fit(xMonthlyTrain,yMonthlyTrain).predict(xMonthlyTest)
nvb_cm = confusion_matrix(yMonthlyTest,nb_pred)
print('Accuracy: ',metrics.accuracy_score(yMonthlyTest, nb_pred))
print('MSE for Naive Bayes:',mean_squared_error(yMonthlyTest,nb_pred))


# In[241]:


make_confusion_matrix(nvb_cm, 
                      group_names=labels,
                      categories=categories,
                      title="Confusion Matrix",
                      cmap="Blues")


# #### Test

# In[ ]:





# In[242]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
accuracy = np.array([metrics.accuracy_score(yMonthlyTest, l_prediction), 
                     metrics.accuracy_score(yMonthlyTest, dtc_prediction),
                     metrics.accuracy_score(yMonthlyTest, rfc_prediction),
                     metrics.accuracy_score(yMonthlyTest, nb_pred)])
max_acc = np.argmax(accuracy)
print(classifiers[max_acc]+' has the highest accuracy: '+str(round(accuracy[max_acc]*100,2))+'%')


# In[243]:


classifiers = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Naive Bayes','SVC']
fn = np.array([l_cm[1,0], 
                     dtc_cm[1,0],
                     rfc_cm[1,0],
                     nvb_cm[1,0]])
min_fn = np.argmin(fn)
print(classifiers[min_fn]+' has the lowest False Negative: '+str(fn[min_fn]))


# In[244]:


# obtain optimal threshold from roc
l_probs = rfc_clf.predict_proba(xMonthlyTest)[:,1]
l_auc = roc_auc_score(np.array(pd.DataFrame(yMonthlyTest)), l_probs)
lr_fpr, lr_tpr, _ = roc_curve(np.array(pd.DataFrame(yMonthlyTest)), l_probs)
optimal_proba_cutoff = sorted(list(zip(np.abs(lr_tpr - lr_fpr), _)), key=lambda i: i[0], reverse=True)[0][1]
roc_predictions = [1 if i >= optimal_proba_cutoff else 0 for i in rfc_clf.predict_proba(xMonthlyTest)[:, -1]]


# In[245]:


optimal_proba_cutoff


# In[246]:


print("Accuracy Score Before and After Thresholding: {}, {}".format(accuracy_score(yMonthlyTest, rfc_prediction), accuracy_score(yMonthlyTest, roc_predictions)))
print("Precision Score Before and After Thresholding: {}, {}".format(precision_score(yMonthlyTest, rfc_prediction), precision_score(yMonthlyTest, roc_predictions)))
print("Recall Score Before and After Thresholding: {}, {}".format(recall_score(yMonthlyTest, rfc_prediction), recall_score(yMonthlyTest, roc_predictions)))
print("F1 Score Before and After Thresholding: {}, {}".format(f1_score(yMonthlyTest, rfc_prediction), f1_score(yMonthlyTest, roc_predictions)))
print("Mean Squared Error: Before and After Thresholding: {}, {}".format(mean_squared_error(yMonthlyTest, rfc_prediction), mean_squared_error(yMonthlyTest, roc_predictions)))


# In[247]:


make_confusion_matrix(confusion_matrix(yMonthlyTest, roc_predictions), 
                      group_names=labels,
                      categories=categories,
                      title="Monthly Heat",
                      cmap="Blues")


# In[248]:


# generate a no skill prediction (majority class)
ns_probs = [0 for _ in range(len(yAtomicTest))]
ns_auc = roc_auc_score(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (l_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(np.array(pd.DataFrame(yAtomicTest)), ns_probs)
# plot the roc curve for the model
plt.figure(figsize=(6,5),dpi=100)
plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Monthly Heat')
plt.legend()
plt.show()

