import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from Data_prep.pre_process import replace_y1n0, graph_mp, attrition_bar_graphs, test_perfomance
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

raw_data = pd.read_csv('/Users/dumisani/Documents/Human_Resources.csv')
# print(raw_data['Age'].describe())
# print(raw_data.info())

# DATA PRE-PROCESSING
reformed_data = replace_y1n0(raw_data)
# print(raw_data[['Attrition', 'Over18', 'OverTime']].head())
reformed_data.drop(['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours'], axis=1, inplace=True)
# data_drop.hist(bins=30, figsize=(20,20), color='r')
# plt.show()

# EMPLOYEES ATTRITION
stayed_data = reformed_data[reformed_data['Attrition'] == 0]
left_data = reformed_data[reformed_data['Attrition'] == 1]
# print('Employees stayed: {0}\nEmployees left: {1}'.format(len(stayed_data), len(left_data)))
# graph_mp(reformed_data)
# for i in reformed_data.columns.values:
#    attrition_bar_graphs(i, 'Attrition', reformed_data)
# FEATURE SELECTION
X_categorical = reformed_data[['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']]
X_categorical = pd.DataFrame(OneHotEncoder().fit_transform(X_categorical).toarray())
# print(X_categorical)
X_numerical = reformed_data[['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EnvironmentSatisfaction',
                             'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                             'MonthlyRate', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                             'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
                             'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany',
                             'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
# scaled inputs

X = MinMaxScaler().fit_transform(pd.concat([X_categorical, X_numerical], axis=1))
Y = reformed_data['Attrition']
# Split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# LG CLASSIFIER

LG_model_pred = (LogisticRegression().fit(X_train, Y_train)).predict(X_test)
name = 'logistic Regression'
test_perfomance(name, LG_model_pred, Y_test)

# RANDOM FOREST CLASSIFIER

RF_model_pred = (RandomForestClassifier().fit(X_train, Y_train)).predict(X_test)
name = 'Random Forest'
test_perfomance(name, RF_model_pred, Y_test)

# DEEP LEARNING MODEL
name='Deep Learning'
DL_model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=500, activation='relu'),
    tf.keras.layers.Dense(units=500, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')])

DL_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
DL_model_pred =DL_model.predict(X_test)
epochs_history = DL_model.fit(X_train,
                              Y_train,
                              batch_size=50,
                              epochs=10,
                              callbacks=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=8),
                              verbose=False)
# plt.plot(epochs_history.history['loss'])
# plt.title('Model Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Training Loss')
# plt.legend('Training Loss')
# plt.show()
test_perfomance(name, DL_model_pred>0.5, Y_test)
