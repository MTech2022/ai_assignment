
#from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

data = pd.DataFrame(
    {
        "Age": [17, 64, 18, 20, 38, 49, 55, 25, 29, 31, 33],
        "Loan Default": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    }
)
data

data.sort_values("Age")

age_tree = DecisionTreeClassifier(random_state=17)
abc = age_tree.fit(data["Age"].values.reshape(-1, 1), data["Loan Default"].values)

tree.plot_tree(abc)
plt.title('Data set 1')
plt.savefig("data_set1.png");
plt.show()
print("---------------------------------Data set one done ----------------------------------------");
data2 = pd.DataFrame(
    {
        "Age": [17, 64, 18, 20, 38, 49, 55, 25, 29, 31, 33],
        "Salary": [25, 80, 22, 36, 37, 59, 74, 70, 33, 102, 88],
        "Loan Default": [1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    }
)
data2
data2.sort_values("Age")

age_sal_tree = DecisionTreeClassifier(random_state=17)
abc = age_sal_tree.fit(data2[["Age", "Salary"]].values, data2["Loan Default"].values);

tree.plot_tree(abc)
plt.title('Data set 2')
plt.savefig("data_set2.png");
plt.show()
print("---------------------------------Data set two done ----------------------------------------");