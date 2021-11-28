import pandas as pd
import pickle
from pickle import dump #to save model

data=pd.read_csv('/Users/KillSwitch/Desktop/footy/epl.csv') #main data
Standings=pd.read_csv('/Users/KillSwitch/Desktop/footy/eplstandings.csv') #standing prev year

data=data.drop(columns=['Unnamed: 0']) #dropped index

X_all = data.drop(['FTR'],1) #independent variable-1 is for column
y_all = data['FTR'] #dependent variable

from sklearn.preprocessing import StandardScaler #to scale all the values
cols=['HTHG', 'HTAG', 'HTP', 'ATP', 'HomeTeamLP',
      'AwayTeamLP', 'DiffPts', 'DiffLP']
scaler = StandardScaler().fit(X_all.loc[:,cols])
X_all.loc[:,cols]=scaler.transform(X_all.loc[:,cols]) #data is scaled

encoded_df = pd.get_dummies(data=X_all, columns=['HomeTeam', 'AwayTeam']) #one hot encoding

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
y_all=le.fit_transform(y_all)

from sklearn.model_selection import train_test_split

# Shuffle and split the dataset into training and testing set.
X_train, X_test, y_train, y_test = train_test_split(encoded_df, y_all,
                                                    test_size = 20,
                                                    random_state = 2,
                                                    stratify = y_all)

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()

clf.fit(X_train,y_train)
# Creating a pickle file for the classifier


filename = 'model.pkl'

pickle.dump(clf, open(filename, 'wb'))

dump(scaler, open('scaler.pkl', 'wb'))


