#!/usr/bin/env python
# coding: utf-8

# # Analysis of the impact of social media usage on the number of hours of sleep

# A study pretends to analyze if the intensity of social media usage influences the number of hours of sleep. With this purpose, four distinct groups were selected, each characterizing a level of intensity of social media usage: Low usage, moderate usage, high usage, and very high usage. Each one of these groups is composed of a sample of 20 people who were firstly asked how they would characterize their social media usage (between the four options available) and later asked their average number of hours of sleep.

# ____________

# In[72]:


# Setup
import pandas as pd
import numpy as np
import statistics
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
import plotly.express as px
import pylab
import scipy.stats as st
from scipy.stats import shapiro
from scipy.stats import levene
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pingouin as pg


# In[3]:


# Plot settings
subPlots_Title_fontSize = 12
subPlots_xAxis_fontSize = 10
subPlots_yAxis_fontSize = 10
subPlots_label_fontSize = 10

plots_Title_fontSize = 14
plots_Title_textColour = 'black'

plots_Legend_fontSize = 12
plots_Legend_textColour = 'black'


# In[4]:


# Import data
df = pd.read_excel('Series51.xlsx')


# In[5]:


df.head()


# In[6]:


# Get dataset info
df.info()


# ## Exploratory Data Analysis

# In[7]:


# Get dataset statistics
df.describe()


# In[8]:


# Calculate median

for i in df.columns:
    print(f"Median of {i}: %.3f " % (statistics.median(df[i])))


# In[9]:


# Calculate variance

for i in df.columns:
    print(f"Variance of {i}: %.3f " % (statistics.variance(df[i])))


# In[10]:


# Plot histograms
def plot_histogram(df,col):    
    # Draw
    fig, ax = plt.subplots(figsize=(8,5))
    g = sns.histplot(df[col], kde=False)

    # Decoration
    fmt = "{x:,.0f}"
    tick = ticker.StrMethodFormatter(fmt)
    ax.yaxis.set_major_formatter(tick)
    sns.despine()
    plt.title(col +' of Social Media', fontsize=plots_Title_fontSize)
    plt.xlabel('Average Number of Hours of Sleep')
    plt.ylabel("Frequency")
    plt.rc('axes', labelsize=subPlots_label_fontSize)


# In[11]:


for i in df.columns:
    plot_histogram(df,i)


# In[12]:


# Violin plot
def violin_plot(ds,col,width,height):
    fig = px.violin(ds, y=col, box=True, points= False)
    fig.update_layout(height=height, width=width, title_text=i + ' of social media', template = "plotly_white")
    fig.update_yaxes(title_text='Average Number of Hours of Sleep')
    fig.show()


# In[13]:


for i in df.columns:
    violin_plot(df,i,400,400)


# In[14]:


## Create 95% confidence intervals using the Normal Distribution
## Performed after the Shapiro-Wilk Normality Test
alpha=0.95

# Population mean
print('--- Population Mean ---')
for i in df.columns:
    c1,c2 = st.t.interval(alpha=alpha, df=len(df)-1, loc=np.mean(df[i]), scale=st.sem(df[i]))
    print(i)
    print(f"95 percent confidence interval: (%.3f , %.3f)\n" % (c1,c2))

# Population standard deviation
print('--- Population Standard Deviation ---')
for i in df.columns:
    c1,c2 = st.t.interval(alpha=alpha, df=len(df)-1, loc=np.std(df[i]), scale=st.sem(df[i]))
    print(i)
    print(f"95 percent confidence interval: (%.3f , %.3f)\n" % (c1,c2))

# Population variance
print('--- Population Variance ---')
for i in df.columns:
    c1,c2 = st.t.interval(alpha=alpha, df=len(df)-1, loc=statistics.variance(df[i]), scale=st.sem(df[i]))
    print(i)
    print(f"95 percent confidence interval: (%.3f , %.3f)\n" % (c1,c2))


# ## Testing

# #### Normality

# In[15]:


# Shapiro-Wilk Normality Test
def normality_test(data):
    '''H0: the sample comes from a normal population with µ and σ unknown.
    H1: the sample does not come from a normal population.'''
    
    stat, p = shapiro(data)
    print('stat=%.3f, p=%.3f' % (stat, p))
    print('Shapiro-Wilk Test')
    if p > 0.05:
        print('The sample comes from a normal population with µ and σ unknown.')
    else:
        print('The sample does not come from a normal population.')

    print('\n')


# In[16]:


for i in df.columns:
    print(i)
    normality_test(df[i])


# In[17]:


# Q-Q plot
for i in df.columns:
    st.probplot(df[i], dist="norm", plot=pylab)
    plt.title('Probability Plot of Average Hours of Sleep\n for '+i+' of Social Media', fontsize=plots_Title_fontSize)
    pylab.show()


# #### Homoscedasticity

# In[18]:


#Levene's test centered at the mean
def variance_test(df):
    '''H0: the variances are equal across all samples/groups.
    H1: the variances are not equal across all samples/groups.
    '''
    
    stat, p =  levene(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3] , center='mean')
    print('stat=%.4f, p=%.4f' % (stat, p))
    print("Levene's Test centered at the mean")
    if p > 0.05:
        print('The variances are equal across all samples/groups.')
    else:
        print("The variances are not equal across all samples/groups.")


# In[19]:


variance_test(df)


# #### ANOVA

# In[24]:


## Analysis of Variance Test
# Store values of each sample
vals = []
for i in range(0,len(df.columns)):
    col_vals = df.iloc[:, i].tolist()
    vals = vals + col_vals

data = pd.DataFrame({'weight': vals,
                   'group': np.repeat(df.columns.to_list(), repeats=len(df))}) 
    
mod = ols('weight ~ group',
                data=data).fit()

aov_table = sm.stats.anova_lm(mod, typ=2)

# Effect sizes
esq_sm = aov_table['sum_sq'][0]/(aov_table['sum_sq'][0]+aov_table['sum_sq'][1])
aov_table['EtaSq'] = [esq_sm, 'NaN']

# Totals
aov_table.loc['Total']= aov_table.sum(numeric_only=True, axis=0)
aov_table.at['Total', 'F'] = None
aov_table.at['Total', 'PR(>F)'] = None

# Mean Square
mean_sqr_0 = aov_table['sum_sq'][0]/aov_table['df'][0]
mean_sqr_1 = aov_table['sum_sq'][1]/aov_table['df'][1]

aov_table['mean_sq'] = [mean_sqr_0, mean_sqr_1,'NaN']

print(aov_table)


# #### Multiple comparison test

# In[71]:


# Store values of each sample
vals = []
for i in range(0,len(df.columns)):
    col_vals = df.iloc[:, i].tolist()
    vals = vals + col_vals

#create DataFrame to hold data
df_tukey = pd.DataFrame({'score': vals,
                   'group': np.repeat(df.columns.to_list(), repeats=len(df))}) 

# perform Tukey's test
res2 = pairwise_tukeyhsd(endog=df_tukey['score'],
                          groups=df_tukey['group'],
                          alpha=0.05)
print("summary:", res2.summary())
print("mean diffs:", res2.meandiffs)
print("std pairs:",res2.std_pairs)
print("groups unique: ", res2.groupsunique)
print("df total:", res2.df_total)
p_values = psturng(np.abs(res2.meandiffs / res2.std_pairs), len(res2.groupsunique), res2.df_total)
print()
print("Unadjusted p values:", p_values)


# In[74]:


# perform Tukey's test using pingouin to get statistics
pt = pg.pairwise_tukey(dv='weight', between='group', data=data)
pt

