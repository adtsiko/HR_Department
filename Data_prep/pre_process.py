import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def replace_y1n0(data):
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='Blues')
    plt.show()
    return data.replace({'Yes': 1, 'No': 0, 'Y': 1, 'N': 0})


def graph_mp(data):
    correlations = data.corr()
    f, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(correlations, annot=True)
    plt.show()
# Age is correlated to job level and subsequently monthly income, monthly income is also strongly correlated to total
    # working hours


def attrition_bar_graphs(m_var, s_var, data):
    plt.figure(figsize=[25, 12])
    sns.countplot(x=m_var, hue=s_var, data=data)
    plt.show()
# The younger the employee the likely the likelihood of them leaving. At 18 years old there a likelihood of 50% of
# attrition

def test_perfomance(name, prediction, test):
    print("{} Model Accuracy: {} %".format(name,round(100*accuracy_score(prediction, test), 4)))
    plt.figure(len(name))
    sns.heatmap(confusion_matrix(prediction, test), annot=True)
    print(classification_report(prediction, test))