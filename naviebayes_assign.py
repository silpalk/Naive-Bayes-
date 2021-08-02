# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 18:37:04 2021

@author: Amarnadh Tadi
"""

import pandas as pd
##loading of train data
salary_train=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign8\SalaryData_train.csv")
salary_train.head()
salary_train.columns
salary_train.isnull().sum()
##Data preprocessing for train data
salary_train.drop(['education','maritalstatus','relationship','race','capitalgain','capitalloss','native'],axis=1,inplace=True)
salary_train.head()
salary_train.dtypes
salary_train_target=salary_train.iloc[:,[6]]
salary_train_target = salary_train_target.replace([' <=50K',' >50K'],[0,1])
display(salary_train_target.head(n=8))
salary_train.drop(['Salary'],axis=1,inplace=True)
salary_train=pd.get_dummies(salary_train,columns=['workclass','occupation','sex'])

salary_train_input=salary_train.iloc[:,0:26]
salary_train=pd.concat([salary_train_input,salary_train_target],axis=1)




##loading of test data
salary_test=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign8\SalaryData_test.csv")
salary_test.head()
##data preprocessing for test data
salary_test.drop(['education','maritalstatus','relationship','race','capitalgain','capitalloss','native'],axis=1,inplace=True)
salary_test.head()
salary_test_target=salary_test.iloc[:,[6]]

salary_test_target = salary_test_target.replace([' <=50K',' >50K'],[0,1])
display(salary_test_target.head(n=8))
salary_test.columns
salary_test.drop(['Salary'],axis=1,inplace=True)
salary_test=pd.get_dummies(salary_test,columns=['workclass','occupation','sex'])

salary_test_input=salary_test.iloc[:,0:26]
salary_test=pd.concat([salary_test_input,salary_test_target],axis=1)

##building navie bayes model 
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
#train the model
model.fit(salary_train_input,salary_train_target.values.ravel())
##to test the accuracy
model.score(salary_test_input,salary_test_target)
##checking of first 10 samples of test data
salary_test_input[:10]
salary_test_target[:10]
##prediction
model.predict(salary_test_input[:10])
##prediction by probabilty
model.predict_proba(salary_test_input[:10])

####car adv.
import pandas as pd


car_ad1=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign8\NB_Car_Ad.csv")
## data pre processing
car_ad1.columns
car_ad=car_ad1.drop(['User ID'],axis=1)
##getting dummies for gender column

car_ad=pd.get_dummies(car_ad,columns=['Gender'])
car_ad.head()
car_ad_target=car_ad.iloc[:,[2]]
car_ad_input=car_ad.drop(['Purchased'],axis=1)
# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(car_ad_input,car_ad_target, test_size = 0.2)

##building navie bayes model 
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
#train the model
model.fit(x_train,y_train)
##to test the accuracy
model.score(x_test,y_test)
##checking of first 10 samples of test data
x_test[:10]
y_test[:10]
##prediction
model.predict(x_test[:10])
##prediction by probabilty
model.predict_proba(x_test[:10])

##tweets data
import pandas as pd
twitter_data=pd.read_csv(r"C:\Users\Amarnadh Tadi\Desktop\datascience\assign8\Disaster_tweets_NB.csv")
# cleaning data 
import re
stop_words = []
# Load the custom built Stopwords
with open(r"C:\Users\Amarnadh Tadi\Desktop\datascience\nlp_textmin\stopwords_en.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

# testing above function with sample text => removes punctuations, numbers
cleaning_text("Residents Return To Destroyed Homes As Washington Wildfire Burns on http://t.co/UcI8stQUg1")


twitter_data.text = twitter_data.text.apply(cleaning_text)

# removing empty rows
twitter_data = twitter_data.loc[twitter_data.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(twitter_data.text,twitter_data.target)

##word frequencies found by count vectoriser
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]

##bulding navie bayes model
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)
tweets = ['there emergency evacuation happening building across street','damage school multi crash breaking']
tweets_count = v.transform(tweets)
model.predict(tweets_count)
X_test_count = v.transform(X_test)
model.score(X_test_count, y_test)
##score prediction by using pipeline
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

clf.fit(X_train, y_train)

clf.score(X_test,y_test)

clf.predict(tweets)

