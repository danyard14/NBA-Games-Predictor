from train_test import train_decision_tree, test_decision_tree
from Utils.names import *

if __name__ == '__main__':
    for criterion in ["gini", "entropy"]:
        for splitter in ["random", "best"]:
            for random_state in [None, 2, 6, 10, 14]:
                print(criterion, splitter, random_state)
                decision_tree = train_decision_tree(TRAIN_DATA_PATH, TRAIN_ALLSTAR_PATH, TRAIN_STANDINGS_PATH, criterion, splitter, random_state)
                test_decision_tree(decision_tree, TEST_DATA_PATH, TEST_ALLSTAR_PATH, TEST_STANDINGS_PATH)
