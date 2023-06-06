import pandas as pd
import numpy as np
from ID3 import DecisionTreeClassifier


def main():
    data_df = pd.read_csv('data/car_evaluation.csv', names=['Price', 'Maintenance', 'Doors', 'Persons', 'Lug_boot', 'Safety', 'Acceptability'], header=None)

    X = np.array(data_df.iloc[:,:-1].copy())
    y = np.array(data_df.iloc[:, -1].copy())
    feature_names = data_df.columns[:-1]
    print(feature_names)
    tree_clf = DecisionTreeClassifier(X, feature_names, y)
    tree_clf.id3()
    tree_clf.printTree()
    print(f"Total entropy: {tree_clf.entropy}")
    tree_clf.plotTree()


if __name__ == '__main__':
    main()
