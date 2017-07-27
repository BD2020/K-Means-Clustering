# K-Means Clustering with Titanicdata
# 
# Using K-Means Clustering, we create two cluster
# groups of Titanic passenger data:
# Survived
# Didn't Survive
# 
# and determine the probability of survival based on
# passenger feature set data
 
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
from sklearn import preprocessing

style.use('ggplot')

'''
Our passenger class feature set:
pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)  - 3rd class was at bottom of ship
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British Pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourgl Q = Queenstown; S = Southhampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

# Determine whether or not we can find some insights from this data
#
# we have the dataset: whether or not the passenger survived
#
# separate these people into two groups:
# Q1: did they survive or not
# Q2: will K means sep two groups into survived / died

cwd = os.getcwd()
#
os.chdir("C:\\ML_Data\\_Titanic")

df = pd.read_excel('titanic.xls')

#print(df.head(5))

# which features will be important to cluster?
# name could be, but we'd need NLP to check for prestigious names
# sex could be, but it's non-numeric
# cabin is important, but it's non-numeric
# embarked is important, but it's non-numeric
# home.dest may be important, but it's non-numeric

# so what do we do with this data, but ML requires
# numerical data
# generally what you do: take text column: take set
# set of sex.values is going to be female=0, male=1
# set of home.dest we just assign 0=city1, 1=city2
#
# with a lot of values, we may end up with a lot of outliers
#
# plus we have a lot of missing data that needs to be filled in
#

# first let's drop some non important columns
df.drop(['body','name'], 1, inplace=True)

#print(df.head(5))

# converts all of the cols to numeric
#
df.convert_objects(convert_numeric=True) 

df.fillna(0, inplace=True)

#print(df.head(5))


# define a function to handle non numeric data
#
def handle_non_numerical_data(df):
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}  # exm: {'Female':0}
        
        def convert_to_int(val):
            return text_digit_vals[val] # returns numeric equiv to text
        
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x += 1

# reset the values of df{col] by mapping this function
# to the value that's in that column
#
            df[column] = list(map(convert_to_int, df[column])) 
    return df # return our modified dataframe
                    
# call our text to int converter
#
df = handle_non_numerical_data(df)

#print(df.head(5))
            
# we could have run this clustering before the ship
# set sail to determine beforehand who would survive
# and who would not
#
# after we've trained, we can then add new data 
# values to predict whether the outcome would be
# survive or die
#

# we're doing unsupervised learning, so we don't
# need to do cross_validation;
#

# df.drop(['boat'], 1, inplace=True)

print(df.head(5))

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)

y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

# if we had a classifier that consistently 
# was 20% accurate, you could flip the result
# to be 80% accurate

correct = 0

for i in range(len(X)):
# could never use the following in supervised learning
#
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    
    if prediction[0] == y[i]:
        correct += 1   # we predicted a correct result

print("accuracy: ", correct / len(X))

# if we consistently get 10%, we can always flip the result
#
# we're getting: accuracy:  0.48 which is not useful
#
# So - what can we change?
# we can preprocess X to scale which now gives
# us something like 82% 73% accuracy
# Next, we can drop ticket # since it's not useful
# but this doesn't seem to help much
#
# Next, try dropping 'boat' to see if this improves accuracy
# dropping boat also didn't help
#
# Also note: 80% of the women survived, 20% of men survived
#
# In next tutorial, we'll be building our own K-means classifier
#


##########################

# example output with accuracy:
"""
pclass  survived  sex      age  sibsp  parch  ticket      fare  cabin  \
0       1         1    1  29.0000      0      0     755  211.3375    108   
1       1         1    0   0.9167      1      2     532  151.5500     76   
2       1         0    1   2.0000      1      2     532  151.5500     76   
3       1         0    0  30.0000      1      2     532  151.5500     76   
4       1         0    1  25.0000      1      2     532  151.5500     76   

   embarked  boat  home.dest  
0         2     1        178  
1         2    24        162  
2         2     0        162  
3         2     0        162  
4         2     0        162  
accuracy:  0.7043544690603514
""""
