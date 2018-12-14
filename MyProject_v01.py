# lecture des donnees
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib.pyplot import hist
from sklearn import metrics
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

les_na = na_strings = ['', '?', 'NaN', 'NAN']
telecom_cust = pd.read_csv(('/home/ouattari/Bureau/Fouilles/KDDCup09_churn.csv'), na_values=les_na)

telecom_cust.head()
telecom_cust.columns.values
telecom_cust.dtypes

x_var_names = telecom_cust.columns
x_types = {x_var_name: telecom_cust[x_var_name].dtype for x_var_name in x_var_names}

for x_var_name in x_var_names:
    if x_types[x_var_name] == int:
        x = telecom_cust[x_var_name].astype(float)
        telecom_cust.ix[:, x_var_name] = x
        x_types[x_var_name] = x.dtype
    elif x_types[x_var_name] != float:
        x = telecom_cust[x_var_name].astype('category')
        telecom_cust.ix[:, x_var_name] = x
        x_types[x_var_name] = x.dtype

fd = telecom_cust.dropna(axis='columns', how='all')
telecom_cust_noNArowCol = fd.dropna(axis='rows', how='all')


OF_telecom_filtred = telecom_cust_noNArowCol.select_dtypes(include=float)

Les_plus_importants = OF_telecom_filtred.isnull().sum() / OF_telecom_filtred.__len__()
Les_plus_importants.to_csv("/home/ouattari/Bureau/Fouilles/Importance.csv", sep='\t', encoding='utf-8')

Les_plus_importants_vars = Les_plus_importants[Les_plus_importants <= .4].index

OF_telecom_OnlyWorthit = OF_telecom_filtred[Les_plus_importants_vars]


float_x_var_names = OF_telecom_OnlyWorthit.columns

OF_telecom_OnlyWorthit[Les_plus_importants_vars].std()

float_x_means = OF_telecom_OnlyWorthit.mean()

for float_x_var_name in float_x_var_names:
    x = OF_telecom_OnlyWorthit[float_x_var_name]
    missing_value_row_yesno = x.isnull()
    if missing_value_row_yesno.sum() > 0:
        OF_telecom_OnlyWorthit.ix[missing_value_row_yesno.tolist(), float_x_var_name] = \
            float_x_means[float_x_var_name]

df_dummies = pd.get_dummies(OF_telecom_OnlyWorthit)

df_dummies.to_csv("/home/ouattari/Bureau/Fouilles/MaBase.csv", sep='\t', encoding='utf-8')
#
#Churn vs tous les autres variables (correlation)
Result_for_chart1 = df_dummies.corr()['CHURN'].sort_values(ascending=True)
hist(Result_for_chart1)
#-------resultat Churn vs tous les autres variables dans fichier CSV
Result_for_chart1.to_csv("/home/ouattari/Bureau/Fouilles/churnVSall.csv", sep='\t', encoding='utf-8')
#pourcentage du Churn dans notre dataset
Result_for_chart2 = (telecom_cust['CHURN'].value_counts()*100.0 /len(telecom_cust))
#-------pourcentage du Churn dans notre dataset dans fichier CSV
Result_for_chart2.to_csv("/home/ouattari/Bureau/Fouilles/PourcentageChurn.csv", sep='\t', encoding='utf-8')

y = df_dummies['CHURN'].values
X = df_dummies.drop(columns=['CHURN'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)

from  sklearn.naive_bayes  import  BernoulliNB
model = BernoulliNB()
model.fit(X_train,y_train)
predictions = model.predict(X_test)
train_sizes, train_scores, valid_scores = learning_curve(
GaussianNB(), X_train, y_train, train_sizes=[50, 80, 110], cv=5)
# print(train_scores)
# print(valid_scores)
print("Naive Bayes Accuracy")
print(metrics.accuracy_score(y_test, predictions))
# print(metrics.confusion_matrix(y_test,predictions))



from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion = "gini",
max_depth = 4, min_samples_split = 3)
model.fit(X,y)
# print(train_scores)
# print(valid_scores)
predictions = model.predict(X_test)
print("Decision Tree Accuracy")
print (metrics.accuracy_score(y_test, predictions))
# print(metrics.confusion_matrix(y_test,predictions))

from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(X_train, y_train)

prediction_test = model_rf.predict(X_test)
# print(train_scores)
# print(valid_scores)
print("Random Forest Accuracy")
print (metrics.accuracy_score(y_test, prediction_test))
# print(metrics.confusion_matrix(y_test,predictions))
