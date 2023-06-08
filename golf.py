import pandas as pd
import numpy as np
from ID3 import DecisionTreeClassifier


def main():
    data_df = pd.read_csv('data/golf.csv')
    X = np.array(data_df.copy().drop(['Jugar'], axis=1))
    y = np.array(data_df.copy()['Jugar'])
    feature_names = data_df.columns[:-1]
    print(feature_names)
    tree_clf = DecisionTreeClassifier(X, feature_names, y)
    tree_clf.id3()
    tree_clf.printTree()
    tree_clf.plotTree()
    test = [["Ll", "M", "A", "N"],["S", "A", "A", "S"]]
    for t in test:
        print(f"La predicción para {t} es: {tree_clf.predict([t])}")


if __name__ == '__main__':
    main()
