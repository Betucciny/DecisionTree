import math
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    """Contains the information of the node and another nodes of the Decision Tree."""

    def __init__(self):
        self.value = None
        self.next = None
        self.childs = None


class DecisionTreeClassifier:
    """Decision Tree Classifier using ID3 algorithm."""

    def __init__(self, X, feature_names, labels):
        self.X = X
        self.feature_names = feature_names
        self.labels = labels
        self.labelCategories = list(set(labels))
        self.labelCategoriesCount = [list(labels).count(x) for x in self.labelCategories]
        self.node = None
        self.entropy = self._get_entropy([x for x in range(len(self.labels))])  # calculates the initial entropy
        self.n_nodes = 0

    def _get_entropy(self, x_ids):
        """ Calculates the entropy.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        __________
        :return: entropy: float, Entropy.
        """
        # sorted labels by instance id
        labels = [self.labels[i] for i in x_ids]
        # count number of instances of each category
        label_count = [labels.count(x) for x in self.labelCategories]
        # calculate the entropy for each category and sum them
        entropy = sum([-count / len(x_ids) * math.log(count / len(x_ids), 2) if count else 0 for count in label_count])
        return entropy

    def _get_information_gain(self, x_ids, feature_id):
        """Calculates the information gain for a given feature based on its entropy and the total entropy of the system.
        Parameters
        __________
        :param x_ids: list, List containing the instances ID's
        :param feature_id: int, feature ID
        __________
        :return: info_gain: float, the information gain for a given feature.
        """
        # calculate total entropy
        info_gain = self._get_entropy(x_ids)
        # store in a list all the values of the chosen feature
        x_features = [self.X[x][feature_id] for x in x_ids]
        # get unique values
        feature_vals = list(set(x_features))
        # get frequency of each value
        feature_vals_count = [x_features.count(x) for x in feature_vals]
        # get the feature values ids
        feature_vals_id = [
            [x_ids[i]
            for i, x in enumerate(x_features)
            if x == y]
            for y in feature_vals
        ]

        # compute the information gain with the chosen feature
        info_gain = info_gain - sum([val_counts / len(x_ids) * self._get_entropy(val_ids)
                                     for val_counts, val_ids in zip(feature_vals_count, feature_vals_id)])

        return info_gain

    def _get_feature_max_information_gain(self, x_ids, feature_ids):
        """Finds the attribute/feature that maximizes the information gain.
        Parameters
        __________
        :param x_ids: list, List containing the samples ID's
        :param feature_ids: list, List containing the feature ID's
        __________
        :returns: string and int, feature and feature id of the feature that maximizes the information gain
        """
        # get the entropy for each feature
        features_entropy = [self._get_information_gain(x_ids, feature_id) for feature_id in feature_ids]
        # find the feature that maximises the information gain
        max_id = feature_ids[features_entropy.index(max(features_entropy))]

        return self.feature_names[max_id], max_id

    def id3(self):
        """Initializes ID3 algorithm to build a Decision Tree Classifier.

        :return: None
        """
        x_ids = [x for x in range(len(self.X))]
        feature_ids = [x for x in range(len(self.feature_names))]
        self.node = self._id3_recv(x_ids, feature_ids, self.node)
        print('')

    def _id3_recv(self, x_ids, feature_ids, node):
        """ID3 algorithm. It is called recursively until some criteria is met.
        Parameters
        __________
        :param x_ids: list, list containing the samples ID's
        :param feature_ids: list, List containing the feature ID's
        :param node: object, An instance of the class Nodes
        __________
        :returns: An instance of the class Node containing all the information of the nodes in the Decision Tree
        """
        if not node:
            self.n_nodes += 1
            node = Node()  # initialize nodes
        # sorted labels by instance id
        labels_in_features = [self.labels[x] for x in x_ids]
        # if all the example have the same class (pure node), return node
        if len(set(labels_in_features)) == 1:
            node.value = self.labels[x_ids[0]]
            return node
        # if there are not more feature to compute, return node with the most probable class
        if len(feature_ids) == 0:
            node.value = max(set(labels_in_features), key=labels_in_features.count)  # compute mode
            return node
        # else...
        # choose the feature that maximizes the information gain
        best_feature_name, best_feature_id = self._get_feature_max_information_gain(x_ids, feature_ids)
        node.value = best_feature_name
        node.childs = []
        # value of the chosen feature for each instance
        feature_values = list(set([self.X[x][best_feature_id] for x in x_ids]))
        # loop through all the values
        for value in feature_values:
            child = Node()
            child.value = value  # add a branch from the node to each feature value in our feature
            self.n_nodes += 1
            node.childs.append(child)  # append new child node to current node
            child_x_ids = [x for x in x_ids if self.X[x][best_feature_id] == value]
            if not child_x_ids:
                child.next = max(set(labels_in_features), key=labels_in_features.count)
                print('')
            else:
                if feature_ids and best_feature_id in feature_ids:
                    to_remove = feature_ids.index(best_feature_id)
                    feature_ids.pop(to_remove)
                # recursively call the algorithm
                child.next = self._id3_recv(child_x_ids, feature_ids, child.next)
        return node

    def predict(self, X):
        if not self.node:
            return
        predictions = []
        for x in X:
            dict_x = dict(zip(self.feature_names, x))
            node = self.node
            while node.childs:
                for child in node.childs:
                    if child.value == dict_x[node.value]:
                        node = child.next
                        break
                else:
                    raise ValueError('Unknown value: {}'.format(dict_x[node.value]))
            predictions.append(node.value)
        return predictions

    def printTree(self):
        if not self.node:
            return

        def printRecursive(node, level):
            if node.childs is None:
                print("\t" * level, end="")
                print(node.value)
                return
            for child in node.childs:
                print("\t" * level, end="")
                print(node.value, ': ', end="")
                print(child.value)
                if child.next is None:
                    continue
                printRecursive(child.next, level + 1)

        printRecursive(self.node, 0)

    def plotTree(self):
        if not self.node:
            return
        graph = nx.DiGraph()
        node_n = 0

        def plotRecursive(node, parent, edge, level):
            nonlocal node_n
            name = str(node.value) + str(node_n)
            node_n += 1
            if node.childs is None:
                graph.add_node(name, label=node.value, layer=level)
                graph.add_edge(parent, name, label=edge)
                return
            graph.add_node(name, label=node.value, layer=level)
            if parent is not None:
                graph.add_edge(parent, name, label=edge)
            for child in node.childs:
                if child.next is None:
                    continue
                plotRecursive(child.next, name, child.value, level + 1)


        plotRecursive(self.node, None, None, 0)
        colors = ['red' if node == self.node.value + '0' else 'lightblue' for node in graph.nodes().keys()]

        plt.figure(3, figsize=(15, 15))
        pos = nx.multipartite_layout(graph, subset_key='layer')
        nx.draw_networkx(graph, pos=pos, node_size=3000, edge_color='k', node_color=colors, with_labels=False, font_size=4)
        edge_labels = nx.get_edge_attributes(graph, 'label')
        nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels)
        node_labels = nx.get_node_attributes(graph, 'label')
        nx.draw_networkx_labels(graph, pos=pos, labels=node_labels)

        plt.axis('off')
        plt.show()

