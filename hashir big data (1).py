#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from matplotlib import style


# In[2]:


#load the data containing pitches
df_pitch= pd.read_csv("pitches.csv")


# In[3]:


#reduce to meaningful variables for correlation matrix
df_corr=df_pitch.drop(columns=['event_num','ab_id', 'pitch_num', 'code','type',  'on_1b','on_2b', 'on_3b', 'pitch_type', 'type', 'on_1b','on_2b', 'on_3b' ])
df_corr_mat= df_corr.corr()


# In[5]:


#correlation plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df_corr_mat,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df_corr.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(df_corr.columns)
ax.set_yticklabels(df_corr.columns)
plt.rc('xtick', labelsize=7) 
plt.rc('ytick', labelsize=7) 
plt.show()


# In[4]:


#load data about each game
df_game= pd.read_csv("games.csv")


# In[7]:


#does attendance impact score?
plt.plot(df_game['attendance'],df_game['away_final_score'],'ro')
plt.xlabel('Attendence')
plt.ylabel('Away Final Scores')


# In[8]:


#does attendance impact home score?
plt.plot(df_game['attendance'],df_game['home_final_score'],'ro')
plt.xlabel('Attendence')
plt.ylabel('Home Final Scores')


# In[9]:


#does a delay in the start of the game impact score?
plt.plot(df_game['delay'],df_game['away_final_score'],'ro')
plt.xlabel('Waiting time before game')
plt.ylabel('Away Final Scores')


# In[10]:


#does a delay impact score at home games?
plt.plot(df_game['delay'],df_game['home_final_score'],'ro')
plt.xlabel('weather')
plt.ylabel('Home Final Scores')


# In[5]:


#lower data of atbats
df_atbat= pd.read_csv("atbats.csv")


# In[6]:


df_atbat.head()


# In[7]:


#load player names
df_player= pd.read_csv("player_names.csv")


# In[8]:


df_player.head()


# In[9]:


#join pitches and abbat data 
df_pitch_atbat = pd.merge(df_pitch, df_atbat, on='ab_id')


# In[10]:



df_pitch_atbat.head()


# In[37]:


#what events do we have? Do any of the events have missing values?
df_pitch_atbat.groupby('event').count()


# In[63]:


#creating a variable of interest called base
df_pitch_atbat['base'] = " "

base= ['Single','Walk','Double','Home Run','Hit By Pitch','Field Error','Intent Walk','Triple']

df_pitch_atbat['base'] = ["0" if x in base else "1" for x in df_pitch_atbat['event']]
df_pitch_atbat= df_pitch_atbat.dropna(axis=0)


# In[66]:


from sklearn.model_selection import train_test_split 
train,test = train_test_split(df_pitch_atbat,test_size = 0.3, random_state=80000)


# In[69]:


#creating x and y variables for training
y_tr= train['base']
X_tr=train._get_numeric_data()


# In[93]:


#scaling the x to avoid bias 
from sklearn import preprocessing
X = X_tr.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
X_tr_scaled = min_max_scaler.fit_transform(X)
X_tr_scaled = pd.DataFrame(X_tr_scaled)


# In[97]:


#training a logistic regression model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=100000).fit(X_tr_scaled, y_tr)
y_pred_tr=clf.predict(X_tr_scaled)


# In[98]:


#training accuracy for logistic model 
from sklearn.metrics import accuracy_score
accuracy_score( y_pred_tr,y_tr)


# In[99]:


#accuracy matrix for logistic regression 
from sklearn.metrics import confusion_matrix
confusion_matrix( y_pred_tr,y_tr)


# In[106]:


#recurrsive feature selection. Narrowing down to 15 variables from 45
from sklearn.feature_selection import RFE
selector = RFE(clf, 15, step=1)
selector = selector.fit(X_tr_scaled, y_tr)
y_pred_tr2=selector.predict(X_tr_scaled)


# In[107]:


#accuracy score with the
accuracy_score( y_pred_tr2,y_tr)


# In[108]:


#accuracy matrix for logistic regression  for 15 best variables
from sklearn.metrics import confusion_matrix
confusion_matrix( y_pred_tr2,y_tr)


# In[ ]:




