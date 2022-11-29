import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

df = pd.read_csv("./dataset/Training.csv")
df.describe()
df.shape

df.drop('Unnamed: 133', axis=1, inplace=True)
df.columns

df['prognosis'].value_counts()

x = df.drop('prognosis', axis = 1)
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) 

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

pred = tree.predict(x_test)
acc = tree.score(x_test, y_test)

print("Acurray on test set: {:.2f}%".format(acc*100))

fi = pd.DataFrame(tree.feature_importances_*100, x_test.columns, columns=['Importance'])
fi.sort_values(by='Importance',ascending=False, inplace=True)
fi

zeros = np.array(fi[fi['Importance'] <= 2.300000].index)
zeros

training_new = df.drop(columns=zeros, axis=1)
training_new.shape[1]
training_new.columns

def modelling(df1):
    x_new = df1.drop('prognosis', axis = 1)
    y_new = df1.prognosis
    x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(x_new, y_new, test_size=0.3, random_state=42) 
    tree.fit(x_train_new, y_train_new)
    
    pred_new = tree.predict(x_test_new)
    
    acc_new = tree.score(x_test_new, y_test_new)
#     a = mean_absolute_error(y_test_new, pred_new)
    print("Acurray on test set: {:.2f}%".format(acc*100))
#     print("mean_absolute_error of the test set: {:.2f}%".format(a))

test = pd.read_csv("./dataset//Testing.csv")
test_new = test.drop(columns=zeros, axis=1)
test_new.shape[1]

modelling(test_new)