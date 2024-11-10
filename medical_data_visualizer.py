import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
matplotlib.use('Agg')

# 1
### Import data
df = pd.read_csv("medical_examination.csv")
df.head(10)
df.describe()

# 2
### Add an overweight column to the data. 
### To determine if a person is overweight, first calculate their BMI by dividing their weight in kilograms by the square of their height in meters. 
### If that value is > 25 then the person is overweight. Use the value 0 for NOT overweight and the value 1 for overweight.

BMI = df['weight']/((df['height']/100)**2)
df['overweight'] = list(map(lambda x: 1 if x > 25 else 0, BMI))
df.head(10)

# 3
### Normalize the data by making 0 always good and 1 always bad. If the value of cholesterol or gluc is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = list(map(lambda x: 0 if x == 1 else 1, df['cholesterol']))
df['cholesterol'].value_counts()
df['gluc'] = list(map(lambda x: 0 if x == 1 else 1, df['gluc']))
df['cholesterol'].value_counts()
# 4
### Convert the data into long format and create a chart that shows the value counts of the categorical features using seaborn's catplot().
###  The dataset should be split by 'Cardio' so there is one chart for each cardio value. The chart should look like examples/Figure_1.png
def draw_cat_plot():
    var = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    # 5
    ### Create a DataFrame for the cat plot using pd.melt with values from cholesterol, gluc, smoke, alco, active, and overweight in the df_cat variable.
    df_cat = pd.melt(df, id_vars = 'cardio', value_vars = var)

    # 6
    ### Group and reformat the data in df_cat to split it by cardio. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat_count = df_cat.groupby(by = ['cardio'], axis = 0).count()
    df_cat
    # 7
    ### Convert the data into long format and create a chart that shows the value counts of the categorical features using the following method provided by the seaborn library import : sns.catplot()
    graph = sns.catplot(data = df_cat, kind = 'count', x = 'variable', col = 'cardio', hue = 'value')
    graph.set_ylabels('total', fontsize = 10)
    #graph.figure.savefig('catplot.png')

    # 8
    ### Get the figure for the output and store it in the fig variable
    fig = graph.figure


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
### Clean the data in the df_heat variable by filtering out the following patient segments that represent incorrect data:
### diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
### height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
### height is more than the 97.5th percentile
### weight is less than the 2.5th percentile
### weight is more than the 97.5th percentile

def draw_heat_map():
    # 11

    df_heat = df.loc[(df['ap_lo'] <= df['ap_hi'])
                & (df['height'] >= df['height'].quantile(0.025)) 
                & (df['height'] <= df['height'].quantile(0.975))
                & (df['weight'] >= df['weight'].quantile(0.025)) 
                & (df['weight'] <= df['weight'].quantile(0.975)), :]
    
    # 12
    ### Calculate the correlation matrix and store it in the corr variable
    corr = df_heat.corr()
    # 13
    ### Generate a mask for the upper triangle and store it in the mask variable
    mask = np.triu(corr)
    # 14
    ### Set up the matplotlib figure
    fig, ax = plt.subplots(figsize = (10,15))
    
    # 15
    ### Plot the correlation matrix using the method provided by the seaborn library import: sns.heatmap()
    sns.heatmap(corr, annot = True, cmap = 'coolwarm', fmt = '0.1f', mask = mask)
    plt.savefig('corelation_matrix.png')

    # 16
    fig.savefig('heatmap.png')
    return fig
