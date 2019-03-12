
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.preprocessing import StandardScaler
from time import time
import numpy as np
from sklearn.naive_bayes import MultinomialNB as NB

df = pd.read_csv('E:/wheel.csv', index_col=0)
df.head()
df.info()
df1 = pd.read_csv('E:/wheeltest.csv', index_col=0)
df1.head()

df.dtypes
X_train= df
y_train = df['status']
X_test =df1
y_test =df1['status']

t1 = time()
nb = NB()
nb.fit(X_train, y_train)
t2 = time()
elapsed_time = t2-t1

prediction  = nb.predict(X_test)
accuracy = nb.score(X_test, y_test)
#if 0 no  defect possibility otherwise 1 
print(prediction[0])
